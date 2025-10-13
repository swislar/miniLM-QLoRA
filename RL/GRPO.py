from trl import GRPOConfig, GRPOTrainer
from RL.MentalHealthScorer import MentalHealthScorer

# DIR = "/home/e/e0958171/miniLM-LoRA"
DIR = "/Users/swislar/Desktop/miniLM-LoRA"


class GRPO():
    def __init__(self):    
        self.config = GRPOConfig(
            output_dir=f"{DIR}/RL/GRPO",
            num_train_epochs=3,
            per_device_train_batch_size=32,
            gradient_accumulation_steps=16,
            logging_steps=10,
            temperature=0.7,
            generation_kwargs={"use_cache": False}
        )
    
    def train(self, model, tokenizer, dataset):
        # https://huggingface.co/docs/trl/main/en/grpo_trainer
        dataset = dataset.rename_column("Context", "prompt")
        
        def grpo_reward():
            scorer = MentalHealthScorer()
            def reward_fn(prompts, completions, **kwargs):
                scores = []
                for output in completions:
                    score = scorer.calculate_quality_score(output)
                    scores.append(score)
                return scores
            return reward_fn
        
        model.config.use_cache = False
        if hasattr(model, 'base_model'):
            model.base_model.config.use_cache = False
        
        trainer = GRPOTrainer(
            model=model, 
            processing_class=tokenizer,
            reward_funcs=grpo_reward(),
            args=self.config,
            train_dataset=dataset,
        )
        
        print("Training...")
        trainer.train()
    
