import torch
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

class LoRA:
    def __init__(self, r = 16, lora_alpha = 32, lora_dropout = 0.5):
        # https://huggingface.co/docs/peft/en/developer_guides/quantization
        self.config = LoraConfig(
            r = r, 
            lora_alpha = lora_alpha,
            lora_dropout = lora_dropout,
            bias = "none",
            task_type="CAUSAL_LM",
            target_modules = ["c_attn", "c_proj", "w1", "w2"] 
        )
    
    def train(self, model, dataset):
        trainer = SFTTrainer(
            model = model.model,
            train_dataset = dataset,
            # dataset_text_field = 'Pair',
            peft_config = self.config,
            processing_class = model.tokenizer,
            args = SFTConfig(
                output_dir = f"./results/{model.name}",
                max_length = 512,
                per_device_train_batch_size = 1,
                gradient_accumulation_steps = 8,
                max_steps = 100,
                learning_rate = 2e-4,
                logging_steps = 10,
            )
        )
        trainer.train()
        print("LoRA complete!")