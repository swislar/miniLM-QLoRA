from datasets import load_dataset

class Benchmark:
    def __init__(self):
        return
    
    def mental_chat(self):
        # https://huggingface.co/datasets/ShenLab/MentalChat16K
        return load_dataset("ShenLab/MentalChat16K")['train']