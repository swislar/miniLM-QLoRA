from Models.Qwen1_8 import Qwen1_8
from RL.GRPO import GRPO
from peft import PeftModel
from Datasets.Dataset import mental_health_counseling_rl

def run():
    # lora_path = "/home/e/e0958171/miniLM-LoRA/SFT/LoRA/Qwen/Qwen-1_8B/checkpoint-84"
    lora_path = "/Users/swislar/Desktop/miniLM-LoRA/SFT/LoRA/Qwen/Qwen-1_8B/checkpoint-100"
    
    qwen = Qwen1_8()
    model = PeftModel.from_pretrained(qwen.model, lora_path, is_trainable=True)
    model.tokenizer = qwen.tokenizer
    
    ds = mental_health_counseling_rl()
    GRPO().train(model, qwen.tokenizer, ds)
    
if __name__ == "__main__":
    run()