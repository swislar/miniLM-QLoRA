from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
from peft import prepare_model_for_kbit_training
import torch

class Qwen1_8:
    def __init__(self, quantize_weights=False):
        # https://huggingface.co/docs/peft/en/developer_guides/quantization
        # No support for MacOS Quantization - https://pypi.org/project/bitsandbytes/
        # EndOfToken/PadToken - https://qwen.readthedocs.io/en/latest/getting_started/concepts.html
        quantization_config = None
        if quantize_weights:
            quantization_config = BitsAndBytesConfig(load_in_4bit = True,
                                                     bnb_4bit_quant_type = "nf4",
                                                     bnb_4bit_compute_dtype = torch.bfloat16)
        self.name = "Qwen/Qwen-1_8B"
        model = AutoModelForCausalLM.from_pretrained(self.name, 
                                                     quantization_config = quantization_config, 
                                                     trust_remote_code = True, 
                                                     dtype='auto')
                
        self.model = prepare_model_for_kbit_training(model)
        self.tokenizer = AutoTokenizer.from_pretrained(self.name, trust_remote_code = True)
        self.tokenizer.pad_token = "<|endoftext|>"
        self.tokenizer.eos_token = "<|im_end|>"
        self.tokenizer.padding_side = "right" 
        print("Model Initialized.")
    
    def get_layers(self):
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                print(name)