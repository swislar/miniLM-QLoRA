from datasets import load_dataset
from tqdm import tqdm

def mental_chat():
    # https://huggingface.co/datasets/ShenLab/MentalChat16K
    ds = load_dataset("ShenLab/MentalChat16K")['train']
    input = []
    output = []
    for x in tqdm(ds):
        input.append(f"INSTRUCTION: {x['instruction']}\nUSER: {x['input']}\nASSISTANT:")
        output.append(f"{x['output']}")
    return input, output