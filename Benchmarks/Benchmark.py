from datasets import load_dataset

def mental_chat():
    # https://huggingface.co/datasets/ShenLab/MentalChat16K
    return load_dataset("ShenLab/MentalChat16K")['train']