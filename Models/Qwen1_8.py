from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
from peft import prepare_model_for_kbit_training
import torch

class Qwen1_8:
    def __init__(self, quantize_weights=True):
        # https://huggingface.co/docs/peft/en/developer_guides/quantization
        quantization_config = None
        if quantize_weights:
            quantization_config = BitsAndBytesConfig(load_in_4bit=True,
                                                     bnb_4bit_quant_type="nf4",
                                                     bnb_4bit_compute_dtype=torch.bfloat16)
        self.model_name = "Qwen/Qwen-1_8B"
        model = AutoModelForCausalLM.from_pretrained(self.model_name, 
                                                     quantization_config=quantization_config, 
                                                     trust_remote_code=True, 
                                                     torch_dtype="auto")
        self.model = prepare_model_for_kbit_training(model)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right" 
        print("Model Initialized.")