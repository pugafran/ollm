import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import math
from typing import Optional, Tuple

import torch
from ollm import Inference, TextStreamer
from torch import nn
from transformers.models.qwen2_moe import modeling_qwen2_moe
from transformers.models.qwen2_moe.modeling_qwen2_moe import apply_rotary_pos_emb, repeat_kv

# Monkey patch Qwen2MoeAttention to derive head_dim from the loaded weights instead of
# relying on a potentially mismatched config value.
original_init = modeling_qwen2_moe.Qwen2MoeAttention.__init__


def new_init(self, config, layer_idx: int | None = None):
    super(modeling_qwen2_moe.Qwen2MoeAttention, self).__init__()
    self.config = config
    self.layer_idx = layer_idx
    if layer_idx is None:
        print(f"Instantiating Qwen2MoeAttention {layer_idx}...")

    self.hidden_size = config.hidden_size
    self.num_heads = config.num_attention_heads
    self.head_dim = self.hidden_size // self.num_heads
    self.num_key_value_heads = config.num_key_value_heads
    self.num_key_value_groups = self.num_heads // self.num_key_value_heads
    self.max_position_embeddings = config.max_position_embeddings
    self.rope_theta = config.rope_theta
    self.is_causal = True
    self.attention_dropout = config.attention_dropout

    self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.qkv_bias)
    self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.qkv_bias)
    self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.qkv_bias)
    self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    self.rotary_emb = modeling_qwen2_moe.Qwen2MoeRotaryEmbedding(config=self.config)


def new_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[modeling_qwen2_moe.Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    proj_head_dim = query_states.shape[-1] // self.num_heads
    kv_head_dim = key_states.shape[-1] // self.num_key_value_heads

    query_states = query_states.view(bsz, q_len, self.num_heads, proj_head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, kv_head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, kv_head_dim).transpose(1, 2)

    cos, sin = self.rotary_emb(value_states, position_ids)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(proj_head_dim)

    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, self.num_heads, q_len, kv_head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, kv_head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.num_heads * kv_head_dim)

    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_values


modeling_qwen2_moe.Qwen2MoeAttention.__init__ = new_init
modeling_qwen2_moe.Qwen2MoeAttention.forward = new_forward


def main():
    model_id = "qwen3-coder"  # Maps to Qwen/Qwen3-Coder-30B-A3B-Instruct in inference.py

    print(f"Initializing {model_id}...")
    o = Inference(model_id, device="cuda:0", logging=True)

    models_dir = "./models/"
    o.ini_model(models_dir=models_dir, force_download=False)

    text_streamer = TextStreamer(o.tokenizer, skip_prompt=True, skip_special_tokens=False)

    print("\nModel loaded. Enter your code prompt (or 'quit' to exit):")

    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in ["quit", "exit"]:
            break

        messages = [
            {"role": "system", "content": "You are Qwen3-Coder, a helpful and expert coding assistant."},
            {"role": "user", "content": user_input},
        ]

        input_ids = o.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(o.device)

        attention_mask = (input_ids != o.tokenizer.pad_token_id).long().to(o.device)

        print("\nAssistant: ", end="", flush=True)

        try:
            o.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=1024,
                streamer=text_streamer,
                pad_token_id=o.tokenizer.eos_token_id,
            )
        except Exception as e:
            print(f"\nError during generation: {e}")


if __name__ == "__main__":
    main()
