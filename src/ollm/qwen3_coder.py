import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor

# Placeholder for loader and stats to maintain compatibility with other modules
loader, stats = None, None

class Qwen3CoderWrapper:
    """
    Wrapper for Qwen3 Coder model to fit into the ollm Inference structure.
    Uses AutoModelForCausalLM for loading, supporting AWQ and accelerate's offloading.
    """
    def __init__(self, model_dir, device="cuda:0"):
        self.device = device
        print(f"Loading Qwen3 Coder from {model_dir} with AutoModelForCausalLM...")
        
        # Load model with AWQ support (requires autoawq) and device_map="auto" for offloading
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_dir,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype="auto"
            )
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Trying to load without device_map (might OOM on small GPUs)...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_dir,
                trust_remote_code=True,
                torch_dtype="auto"
            ).to(device)

        self.config = self.model.config
        self.num_hidden_layers = self.config.num_hidden_layers

    def generate(self, **kwargs):
        return self.model.generate(**kwargs)

    def offload_layers_to_cpu(self, layers_num=0):
        print("Qwen3 Coder (AWQ) uses accelerate's device_map='auto' for offloading. Manual layer offloading is skipped.")

    def offload_layers_to_gpu_cpu(self, **kwargs):
        pass

    def to(self, device):
        # Model is already distributed via device_map
        pass

    def eval(self):
        self.model.eval()
