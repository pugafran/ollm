# llama3-1B/3B/8B-chat

import time, os
from datetime import datetime
import threading
import numpy as np
import torch
from torch import nn
from typing import Callable, Optional, Tuple, Union, Dict, Any, Iterable, List, Unpack
from .utils import _walk_to_parent, _assign_tensor_to_module, _set_meta_placeholder

# shared objects
loader, stats = None, None

#======== rewriting core classes (tested on transformers==4.52.3) ==============
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, eager_attention_forward, LlamaForCausalLM, LlamaAttention, LlamaMLP, LlamaDecoderLayer, LlamaModel, LlamaConfig, create_causal_mask, Cache
from transformers.modeling_outputs import BaseModelOutputWithPast

class MyLlamaMLP(LlamaMLP):
	def forward(self, x): #down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
		chunk_size, chunks = 16384, []
		x = x.squeeze(0)
		for i in range(0, x.shape[0], chunk_size):
			gate_chunk = self.act_fn(self.gate_proj(x[i:i+chunk_size]))			
			up_chunk = self.up_proj(x[i:i+chunk_size])
			out_chunk = self.down_proj(gate_chunk * up_chunk)
			chunks.append(out_chunk)
		down_proj = torch.cat(chunks, dim=0).unsqueeze(0) #T,C->1,T,C
		return down_proj


class loaderLayer1: #legacy GDSWeights
	def _layer_param_manifest_names(self) -> dict:		
		base = f"model.layers.{self.layer_idx}"
		return {
			"self_attn.q_proj.weight": f"{base}.self_attn.q_proj.weight",
			"self_attn.k_proj.weight": f"{base}.self_attn.k_proj.weight",
			"self_attn.v_proj.weight": f"{base}.self_attn.v_proj.weight",
			"self_attn.o_proj.weight": f"{base}.self_attn.o_proj.weight",
			"mlp.gate_proj.weight":  f"{base}.mlp.gate_proj.weight",
			"mlp.up_proj.weight":    f"{base}.mlp.up_proj.weight",
			"mlp.down_proj.weight":  f"{base}.mlp.down_proj.weight",
			"input_layernorm.weight":  f"{base}.input_layernorm.weight",
			"post_attention_layernorm.weight": f"{base}.post_attention_layernorm.weight",
			# add any biases or additional params you need
		}

	def _load_layer_weights(self):
		manifest_map = self._layer_param_manifest_names()
		for attr_path, manifest_name in manifest_map.items():
			try:
				t1 = time.perf_counter()
				tensor = loader.load_param_to_cuda(manifest_name)
				parent, leaf = _walk_to_parent(self, attr_path)
				_assign_tensor_to_module(parent, leaf, tensor)
				if stats: stats.set("layer_load", t1)
			except Exception as e:
				# Be explicit about failures so you can debug missing names
				raise RuntimeError(f"failed to load {manifest_name} into {attr_path}: {e}")
		#if torch.cuda.is_available(): torch.cuda.synchronize()

	def _unload_layer_weights(self):
		"""Replace each loaded attribute with a meta placeholder to free GPU memory."""
		manifest_map = self._layer_param_manifest_names()
		for attr_path in manifest_map.keys():
			parent, leaf = _walk_to_parent(self, attr_path)
			# replace with placeholder (keeps module graph intact)
			_set_meta_placeholder(parent, leaf)


class loaderLayer:
	def _load_layer_weights(self):
		t1 = time.perf_counter()
		base = f"model.layers.{self.layer_idx}."
		loader.preload_layer_safetensors(base)
		d = loader.load_dict_to_cuda(base)
		for attr_path, tensor in d.items():
			parent, leaf = _walk_to_parent(self, attr_path)
			_assign_tensor_to_module(parent, leaf, tensor)
		if stats: stats.set("layer_load", t1)
			
	def _unload_layer_weights(self):
		base = f"model.layers.{self.layer_idx}."
		for attr_path in loader.manifest[base]:
			parent, leaf = _walk_to_parent(self, attr_path)
			_set_meta_placeholder(parent, leaf)


class MyLlamaDecoderLayer(LlamaDecoderLayer, loaderLayer):
	def __init__(self, config: LlamaConfig, layer_idx: int):
		self.layer_idx = layer_idx
		super().__init__(config, layer_idx)

	def forward(self, *args, **kwargs):
		self._load_layer_weights()
		out = super().forward(*args, **kwargs)
		self._unload_layer_weights()
		return out


class MyLlamaModel(LlamaModel):
	def __init__(self, config):
		super().__init__(config)		
		self.layers = nn.ModuleList()
		for layer_idx in range(config.num_hidden_layers):
			self.layers.append(MyLlamaDecoderLayer(config, layer_idx))
			self.layers[-1]._unload_layer_weights()		

	def forward(
		self,
		input_ids: Optional[torch.LongTensor] = None,
		attention_mask: Optional[torch.Tensor] = None,
		position_ids: Optional[torch.LongTensor] = None,
		past_key_values: Optional[Cache] = None,
		inputs_embeds: Optional[torch.FloatTensor] = None,
		cache_position: Optional[torch.LongTensor] = None,
		use_cache: Optional[bool] = None,
		**kwargs: Unpack, #[TransformersKwargs]
	) -> BaseModelOutputWithPast:
		if (input_ids is None) ^ (inputs_embeds is not None):
			raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

		if inputs_embeds is None:
			inputs_embeds: torch.Tensor = self.embed_tokens(input_ids)

		if use_cache and past_key_values is None:
			past_key_values = DynamicCache()

		if cache_position is None:
			past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
			cache_position: torch.Tensor = torch.arange(
				past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
			)

		if position_ids is None:
			position_ids = cache_position.unsqueeze(0)

		causal_mask = create_causal_mask(
			config=self.config,
			input_embeds=inputs_embeds,
			attention_mask=attention_mask,
			cache_position=cache_position,
			past_key_values=past_key_values,
			position_ids=position_ids,
		)

		hidden_states = inputs_embeds
		position_embeddings = self.rotary_emb(hidden_states, position_ids)

		#============= meine ==============		
		self.embed_tokens.cpu(); self.parent_lm_head.cpu()		

		for layer_idx, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
			hidden_states = decoder_layer(
				hidden_states,
				attention_mask=causal_mask,
				position_ids=position_ids,
				past_key_value=past_key_values,
				cache_position=cache_position,
				position_embeddings=position_embeddings,
				**kwargs,
			)			

		hidden_states = self.norm(hidden_states)
		self.embed_tokens.to(hidden_states.device); self.parent_lm_head.to(hidden_states.device)
		if stats: print("./Llama.forward.", datetime.now().strftime("%H:%M:%S"), stats.print_and_clean() if stats else "")
		#====================================
		
		return BaseModelOutputWithPast(
			last_hidden_state=hidden_states,
			past_key_values=past_key_values,
		)

# Monkey-patch
import transformers.models.llama.modeling_llama as llama_modeling
#llama_modeling.LlamaAttention = MyLlamaAttention #replaced to stable attn_implementation="flash_attention_2"
llama_modeling.LlamaMLP = MyLlamaMLP
llama_modeling.LlamaDecoderLayer = MyLlamaDecoderLayer
llama_modeling.LlamaModel = MyLlamaModel
#===============================================

class oForGeneration: #copied from gemma3 prefix=language_model.
	def generate(self, **args):
		with torch.no_grad():			
			return super().generate(**args)

	def offload_layers_to_cpu(self, layers_num=2):
		print(f"offloading layers to CPU {layers_num}/{self.num_hidden_layers}...")
		for layer_idx in range(min(layers_num, self.num_hidden_layers)):
			base = f"model.layers.{layer_idx}."
			loader.preload_layer_safetensors(base)
			loader.offload_dict_to_gpu_cpu(base, gpu=False)		
		print(f"./finished offloading layers to CPU {layers_num}/{self.num_hidden_layers}")


class MyLlamaForCausalLM(LlamaForCausalLM, oForGeneration):
	def __init__(self, config):
		super().__init__(config)
		self.model.parent_lm_head = self.lm_head #link
		self.num_hidden_layers = config.num_hidden_layers

	def generate(self, **args):
		with torch.no_grad():
			return super().generate(**args)

	def offload_layers_to_cpu1(self, layers_num=2): #GDSWeights version
		for layer_idx in range(min(layers_num, self.num_hidden_layers)):			
			for name, attr in loader.manifest.items():
				if name.startswith(f"model.layers.{layer_idx}."):
					loader.offload_param_to_cpu(name)
		print("./Llama offloading layers to CPU. Done.")
