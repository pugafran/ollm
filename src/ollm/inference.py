import os, requests, zipfile
from dataclasses import dataclass
from typing import Callable, Dict, Optional

import torch
from transformers import AutoTokenizer, AutoProcessor, AutoConfig

from .utils import Stats, file_get_contents
from .gds_loader import GDSWeights, DenseWeightsLoader, MoEWeightsLoader, SingleDenseWeightsLoader
from .kvcache import KVCache

def get_attn_implementation():
        try:
                import flash_attn
                return "flash_attention_2"
        except ImportError:
                print("Warning: flash_attention_2 is not imported. The context length will be limited")
                return None


@dataclass
class ModelSpec:
        """Metadata describing how to download and load a model."""

        load_fn: Callable[["Inference", str], None]
        repo_id: Optional[str] = None
        local_dir: Optional[str] = None
        download_fn: Optional[Callable[["Inference", str, str], None]] = None
        disk_cache_fn: Optional[Callable[["Inference", str], Optional[object]]] = None


class Inference:
        model_registry: Dict[str, ModelSpec] = {}
        model_family_registry: Dict[str, Callable[[str, "Inference", "AutoConfig"], ModelSpec]] = {}

        def __init__(self, model_id, device="cuda:0", logging=True, multimodality=False):
                self.model_id = model_id
                self.device = torch.device(device)
                self.multimodality = multimodality
                self.stats = Stats() if logging else None
                self._model_spec: Optional[ModelSpec] = None

        @classmethod
        def register_model(cls, name: str, spec: ModelSpec):
                """Register a custom model loader at runtime."""
                cls.model_registry[name] = spec

        @classmethod
        def register_model_family(
                cls,
                model_type: str,
                builder: Callable[[str, "Inference", "AutoConfig"], ModelSpec],
        ):
                """Register a loader builder that can service any Hugging Face repo for a model family."""
                cls.model_family_registry[model_type] = builder

        def download_and_unpack(self, models_dir: str):
                os.makedirs(models_dir, exist_ok=True)
                urls = {
                        "gpt-oss-20B": "https://ollm.s3.us-east-1.amazonaws.com/models/gpt-oss-20B.zip"
                }
                url = urls[self.model_id]
                
                # Extract filename from URL
                filename = url.split("/")[-1]
                zip_path = os.path.join(models_dir, filename)

                # Download the file
                print(f"Downloading {url} ...")
                response = requests.get(url, stream=True)
                response.raise_for_status()
                with open(zip_path, "wb") as f:
                        for chunk in response.iter_content(chunk_size=8192):
                                f.write(chunk)
                print(f"Downloaded to {zip_path}")

                # Unzip
                print(f"Unpacking {zip_path} ...")
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(models_dir)
                print(f"Unpacked to {models_dir}")

                os.remove(zip_path) # Optional: remove the zip file after extraction

        
        def hf_download(self, repo_id: str, model_dir: str):
                from huggingface_hub import snapshot_download
                print(f"Downloading {repo_id} ...")
                snapshot_download(repo_id=repo_id, local_dir=model_dir, local_dir_use_symlinks=False)

        @staticmethod
        def _safe_local_name(model_id: str) -> str:
                return model_id.replace("/", "__")

        @staticmethod
        def _default_disk_cache(_, cache_dir: str):
                return KVCache(cache_dir=cache_dir)

        @staticmethod
        def _create_default_spec(model_id: str) -> ModelSpec:
                def _load_default(inference: "Inference", model_dir: str):
                        from transformers import AutoModelForCausalLM

                        dtype = torch.bfloat16 if inference.device.type in {"cuda", "mps"} else torch.float32
                        model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=dtype)
                        model.to(inference.device)
                        inference.model = model
                        inference.tokenizer = AutoTokenizer.from_pretrained(model_dir)

                return ModelSpec(
                        repo_id=model_id,
                        local_dir=Inference._safe_local_name(model_id),
                        load_fn=_load_default,
                        disk_cache_fn=Inference._default_disk_cache,
                )


        def _create_spec_from_config(self, model_id: str) -> Optional[ModelSpec]:
                if not self.model_family_registry:
                        return None

                try:
                        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
                except Exception:
                        return None

                builder = self.model_family_registry.get(getattr(config, "model_type", None))
                if builder is None:
                        return None

                spec = builder(model_id, self, config)
                if spec.local_dir is None:
                        spec.local_dir = self._safe_local_name(model_id)
                if spec.repo_id is None:
                        spec.repo_id = model_id
                return spec


        def ini_model(self, models_dir="./models/", force_download=False):
                spec = self.model_registry.get(self.model_id)
                if spec is None:
                        spec = self._create_spec_from_config(self.model_id)
                if spec is None:
                        spec = self._create_default_spec(self.model_id)
                self._model_spec = spec

                local_name = spec.local_dir or self._safe_local_name(self.model_id)
                model_dir = os.path.join(models_dir, local_name)

                if not os.path.exists(model_dir) or force_download:
                        if spec.download_fn:
                                spec.download_fn(self, models_dir, local_name)
                        elif spec.repo_id:
                                self.hf_download(spec.repo_id, model_dir)
                        else:
                                raise ValueError(f"No download method available for model '{self.model_id}'")

                print("loading model from", model_dir)
                spec.load_fn(self, model_dir)

                self.model.eval()
                self.model.to(self.device)
                if not hasattr(self, "tokenizer"):
                        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)


        def offload_layers_to_cpu(self, **args):
                self.model.offload_layers_to_cpu(**args)
        
        def offload_layers_to_gpu_cpu(self, **args):
                self.model.offload_layers_to_gpu_cpu(**args)
        
        def DiskCache(self, cache_dir="./kvcache"):
                spec = getattr(self, "_model_spec", None)
                if spec and spec.disk_cache_fn is not None:
                        return spec.disk_cache_fn(self, cache_dir)
                return KVCache(cache_dir=cache_dir, stats=self.stats)


def _register_builtin_models():
        def _llama_loader(inference: Inference, model_dir: str):
                from . import llama

                index_path = os.path.join(model_dir, "model.safetensors.index.json")
                single_file_path = os.path.join(model_dir, "model.safetensors")
                if os.path.exists(index_path):
                        loader_impl = DenseWeightsLoader
                elif os.path.exists(single_file_path):
                        loader_impl = SingleDenseWeightsLoader
                else:
                        loader_impl = DenseWeightsLoader

                llama.loader = loader_impl(model_dir)
                llama.stats = inference.stats
                inference.model = llama.MyLlamaForCausalLM.from_pretrained(
                        model_dir,
                        torch_dtype=torch.bfloat16,
                        device_map="cpu",
                        attn_implementation=get_attn_implementation(),
                        low_cpu_mem_usage=True,
                        ignore_mismatched_sizes=True,
                )

        def _gemma_loader(inference: Inference, model_dir: str):
                from . import gemma3

                gemma3.loader = DenseWeightsLoader(model_dir)
                gemma3.stats = inference.stats
                automodel = gemma3.MyGemma3ForConditionalGeneration if inference.multimodality else gemma3.MyGemma3ForCausalLM
                inference.model = automodel.from_pretrained(
                        model_dir,
                        torch_dtype=torch.bfloat16,
                        device_map="cpu",
                        attn_implementation=get_attn_implementation(),
                        low_cpu_mem_usage=True,
                        ignore_mismatched_sizes=True,
                )
                inference.processor = AutoProcessor.from_pretrained(model_dir)

        def _voxtral_loader(inference: Inference, model_dir: str):
                from . import voxtral

                voxtral.loader = DenseWeightsLoader(model_dir)
                voxtral.stats = inference.stats
                inference.model = voxtral.MyVoxtralForConditionalGeneration.from_pretrained(
                        model_dir,
                        torch_dtype="auto",
                        device_map="cpu",
                        attn_implementation=get_attn_implementation(),
                        low_cpu_mem_usage=True,
                        ignore_mismatched_sizes=True,
                )
                inference.processor = AutoProcessor.from_pretrained(model_dir)
                inference.tokenizer = inference.processor.tokenizer

        def _qwen_loader(inference: Inference, model_dir: str):
                from . import qwen3_next

                qwen3_next.loader = MoEWeightsLoader(model_dir)
                qwen3_next.stats = inference.stats
                inference.model = qwen3_next.MyQwen3NextForCausalLM.from_pretrained(
                        model_dir,
                        torch_dtype=torch.bfloat16,
                        device_map="cpu",
                        attn_implementation=get_attn_implementation(),
                        low_cpu_mem_usage=True,
                        ignore_mismatched_sizes=True,
                )

        def _gpt_loader(inference: Inference, model_dir: str):
                from . import gpt_oss

                gpt_oss.loader = GDSWeights(os.path.join(model_dir, "gds_export"))
                gpt_oss.stats = inference.stats
                inference.model = gpt_oss.MyGptOssForCausalLM.from_pretrained(
                        model_dir,
                        torch_dtype=torch.bfloat16,
                        device_map="cpu",
                        low_cpu_mem_usage=True,
                        ignore_mismatched_sizes=True,
                )

        def _gpt_disk_cache(inference: Inference, _cache_dir: str):
                print(f"{inference.model_id} DiskCache is not supported at the moment. Using default DynamicCache instead")
                return None

        def _qwen_disk_cache(inference: Inference, cache_dir: str):
                from .qwen3_next import Qwen3NextDiskCache

                return Qwen3NextDiskCache(inference.model.config, cache_dir=cache_dir, stats=inference.stats)

        builtins = {
                "llama3-1B-chat": ModelSpec(
                        repo_id="unsloth/Llama-3.2-1B-Instruct",
                        load_fn=_llama_loader,
                        disk_cache_fn=Inference._default_disk_cache,
                ),
                "llama3-3B-chat": ModelSpec(
                        repo_id="unsloth/Llama-3.2-3B-Instruct",
                        load_fn=_llama_loader,
                        disk_cache_fn=Inference._default_disk_cache,
                ),
                "llama3-8B-chat": ModelSpec(
                        repo_id="unsloth/Meta-Llama-3.1-8B-Instruct",
                        load_fn=_llama_loader,
                        disk_cache_fn=Inference._default_disk_cache,
                ),
                "gpt-oss-20B": ModelSpec(
                        repo_id="AnuarSh/gpt-oss-20B",
                        local_dir="gpt-oss-20B",
                        load_fn=_gpt_loader,
                        download_fn=lambda self, models_dir, _local: self.download_and_unpack(models_dir),
                        disk_cache_fn=_gpt_disk_cache,
                ),
                "qwen3-next-80B": ModelSpec(
                        repo_id="Qwen/Qwen3-Next-80B-A3B-Instruct",
                        load_fn=_qwen_loader,
                        disk_cache_fn=_qwen_disk_cache,
                ),
                "gemma3-12B": ModelSpec(
                        repo_id="google/gemma-3-12b-it",
                        load_fn=_gemma_loader,
                        disk_cache_fn=Inference._default_disk_cache,
                ),
                "voxtral-small-24B": ModelSpec(
                        repo_id="mistralai/Voxtral-Small-24B-2507",
                        load_fn=_voxtral_loader,
                        disk_cache_fn=Inference._default_disk_cache,
                ),
        }

        for name, spec in builtins.items():
                Inference.register_model(name, spec)

        Inference.register_model_family(
                "llama",
                lambda model_id, _inference, _config: ModelSpec(
                        repo_id=model_id,
                        load_fn=_llama_loader,
                        disk_cache_fn=Inference._default_disk_cache,
                ),
        )
        Inference.register_model_family(
                "gemma3",
                lambda model_id, _inference, _config: ModelSpec(
                        repo_id=model_id,
                        load_fn=_gemma_loader,
                        disk_cache_fn=Inference._default_disk_cache,
                ),
        )
        Inference.register_model_family(
                "qwen3_next",
                lambda model_id, _inference, _config: ModelSpec(
                        repo_id=model_id,
                        load_fn=_qwen_loader,
                        disk_cache_fn=_qwen_disk_cache,
                ),
        )
        Inference.register_model_family(
                "voxtral",
                lambda model_id, _inference, _config: ModelSpec(
                        repo_id=model_id,
                        load_fn=_voxtral_loader,
                        disk_cache_fn=Inference._default_disk_cache,
                ),
        )


_register_builtin_models()
