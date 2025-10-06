from datasets import load_dataset

class Dataset:
    def __init__(self):
        return
    
    def mental_health_counseling(self):
        # https://huggingface.co/datasets/Amod/mental_health_counseling_conversations
        return load_dataset("Amod/mental_health_counseling_conversations")['train']
    