from datasets import load_dataset


def mental_health_counseling():
    # https://huggingface.co/datasets/Amod/mental_health_counseling_conversations
    # Special Tokens for QWEN - https://qwen.readthedocs.io/en/latest/getting_started/concepts.html
    ds = load_dataset("Amod/mental_health_counseling_conversations")['train']
    return ds.map(lambda x: {'Pair': f"<|im_start|>USER:\n{x['Context']}<|im_end|>\n<|im_start|>ASSISTANT:\n{x['Response']}<|im_end|><|endoftext|>"})
    