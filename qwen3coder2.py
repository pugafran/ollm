import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
import torch
from torch import nn
from ollm import Inference, TextStreamer

# Monkey patch Qwen2MoeAttention to derive head dimensions from loaded weights and fix reshape
from transformers.models.qwen2_moe import modeling_qwen2_moe
from transformers.models.qwen2_moe.modeling_qwen2_moe import apply_rotary_pos_emb, repeat_kv
import math
from typing import Optional, Tuple


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

    head_dim = query_states.shape[-1] // self.num_heads
    kv_head_dim = key_states.shape[-1] // self.num_key_value_heads

    query_states = query_states.view(bsz, q_len, self.num_heads, head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, kv_head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, kv_head_dim).transpose(1, 2)

    cos, sin = self.rotary_emb(value_states, position_ids)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)

    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, self.num_heads, q_len, head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.num_heads * head_dim)

    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_values


modeling_qwen2_moe.Qwen2MoeAttention.forward = new_forward


def main():
    model_id = "qwen3-coder" # Maps to Qwen/Qwen3-Coder-30B-A3B-Instruct in inference.py

    print(f"Initializing {model_id}...")
    # device="cuda:0" is standard
    o = Inference(model_id, device="cuda:0", logging=True)

    # This will download the model if not present
    # You can change models_dir to your preferred location
    models_dir = "./models/"
    o.ini_model(models_dir=models_dir, force_download=False)

    # Offload layers to CPU for speed boost (optional, adjust as needed)
    # Since this is a large model (30B), offloading is likely needed on 8GB VRAM
    # o.offload_layers_to_cpu(layers_num=2)

    # Create streamer
    text_streamer = TextStreamer(o.tokenizer, skip_prompt=True, skip_special_tokens=False)

    print("\nModel loaded. Enter your code prompt (or 'quit' to exit):")

    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in ["quit", "exit"]:
            break

        messages = [
            {"role": "system", "content": "You are Qwen3-Coder, a helpful and expert coding assistant."},
            {"role": "user", "content": user_input}
        ]

        # Apply chat template
        input_ids = o.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(o.device)

        # Create attention mask
        attention_mask = (input_ids != o.tokenizer.pad_token_id).long().to(o.device)

        print("\nAssistant: ", end="", flush=True)

        # Generate
        # Note: past_key_values might need to be handled if we want multi-turn with caching,
        # but for simple one-shot or if the wrapper handles it, we can pass it.
        # The wrapper delegates to model.generate, which handles internal caching if use_cache=True (default).

        try:
            outputs = o.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=1024,
                streamer=text_streamer,
                pad_token_id=o.tokenizer.eos_token_id
            )
        except Exception as e:
            print(f"\nError during generation: {e}")

if __name__ == "__main__":
    main()
