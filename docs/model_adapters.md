# Model adapters in oLLM

This repository keeps a lightweight Hugging Face `transformers` dependency but swaps in small
monkey-patches per architecture to enable on-demand weight streaming. Every model specific module
under `src/ollm/` follows the same pattern:

1. Import the original `transformers` implementation for the target architecture.
2. Subclass the attention/MLP/decoder blocks so that a global `loader` object is consulted to lazily
   pull each tensor from disk (or, in the case of GPT-OSS, from the CuFile backed GDS manifest).
3. Register those subclasses back into `transformers` so any call to `from_pretrained` picks up the
   patched behaviour.
4. Expose a `ForCausalLM`/`ForConditionalGeneration` wrapper that inherits the patched model and
   optionally mixes in helpers such as `offload_layers_to_cpu` for manual memory management.

The modules of interest are:

- `gemma3.py` — replaces `Gemma3MLP`, `Gemma3DecoderLayer`, and `Gemma3TextModel` so each layer is
  loaded just-in-time and discarded again after the forward pass. It also equips the causal LM and
  multimodal conditional generation heads with CPU/GPU offload helpers.
- `llama.py` — mirrors the same idea for the Llama 3 stack. It supports both legacy “single dense”
  checkpoints (`model.safetensors`) and sharded safetensors (`model-00001-of-XXXX.safetensors`) and
  provides optional CPU offloading utilities.
- `qwen3_next.py` — extends the technique for mixture-of-experts (MoE) checkpoints. Besides
  lazy-loading the base layers it handles expert-specific files and ships an on-disk KV-cache
  implementation (`Qwen3NextDiskCache`).
- `voxtral.py` — reuses the patched Llama decoder to power the Voxtral speech model while keeping
  Voxtral’s higher level text/audio plumbing intact.
- `gpt_oss.py` plus `gds_loader.py` — plugs in the experimental GPU Direct Storage (GDS) manifest
  format produced by the GPT-OSS exporter. The manifest references CuFile byte ranges so the
  codepath can DMA tensors directly into GPU memory.

### Why do the architecture files still exist?

Although the registry introduced in `Inference` can discover new Hugging Face repositories without
touching the codebase, the heavy lifting still happens inside these architecture-specific modules.
Each file mirrors the upstream implementation so it can replace individual attention blocks,
mixture-of-experts routers, or audio/text heads with streaming-aware variants.  Hugging Face’s
generic `AutoModelForCausalLM` loader does not expose hooks for overriding just the weight loading
logic, so we keep thin wrappers that:

1. import the upstream layers and re-export subclasses that consult the active `loader` singleton,
2. provide utilities such as KV cache offloading or CPU/GPU streaming that are tailored to the
   architecture,
3. register the subclasses back into `transformers` so that any model of the same `model_type`
   automatically benefits from the patches.

Without these Python files we would fall back to the vanilla Hugging Face behaviour, losing the lazy
weight paging and offload helpers that make oLLM run massive contexts on small GPUs. The new
registry only decides *which* adapter to activate; the adapters themselves remain essential to
provide the memory-optimized execution path.

Because the loader object is stored in a module level global, the `Inference` helper simply needs to
assign the correct implementation before calling `from_pretrained`. The new family-aware registry in
`Inference` automates this step so that any Hugging Face repo advertising a known `model_type`
receives the matching adapter without editing Python files.
