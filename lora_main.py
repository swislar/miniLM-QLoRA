from Models.Qwen1_8 import Qwen1_8
from SFT.LoRA import LoRA
from Datasets.Dataset import mental_health_counseling_sft

def run():
    qwen = Qwen1_8()
    ds = mental_health_counseling_sft()
    LoRA().train(qwen, ds)
    
if __name__ == "__main__":
    run()