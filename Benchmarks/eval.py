import torch
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from Benchmark import mental_chat
from bert_score import score
from rouge_score import rouge_scorer
from vllm import LLM, SamplingParams 
from vllm.lora.request import LoRARequest

def evaluate(model, prompt_input, prompt_output, lora=False):
    model_outputs = []
    
    sampling_params = SamplingParams(
        max_tokens = 726,
        temperature = 0.7,
        top_p = 0.9,
        repetition_penalty = 1.1,
    )
    
    if lora:
        llm = LLM(
            model=model,
            dtype="bfloat16",
            tensor_parallel_size=torch.cuda.device_count(),
            max_num_batched_tokens=8192,
            enable_lora=True,
            trust_remote_code=True
        )
        lora_request = LoRARequest(lora_name="qwen-sft",
                                lora_int_id=1,
                                lora_path="/home/e/e0958171/miniLM-LoRA/SFT/LoRA/Qwen/Qwen-1_8B/checkpoint-100"
        )
        outputs = llm.generate(prompt_input, 
                               sampling_params,
                               lora_request = lora_request)
    else:
        llm = LLM(
            model=model,
            dtype="bfloat16",
            tensor_parallel_size=torch.cuda.device_count(),
            max_num_batched_tokens=8192,
            enable_lora=False,
            trust_remote_code=True
        )
        outputs = llm.generate(prompt_input, 
                               sampling_params)
    
    model_outputs = []
    for output in tqdm(outputs):
        generated_text = output.outputs[0].text
        
        if "\nASSISTANT:" in generated_text:
            generated_text = generated_text.split("\nASSISTANT:")[1]
        
        model_outputs.append(generated_text.strip())

    print("Generation complete. Releasing the VLLM engine from memory...")
    del llm
    torch.cuda.empty_cache()
    print("Memory released.")
    
    print(f"Initial number of prompts loaded:   {len(prompt_input)}")
    print(f"Initial number of references loaded: {len(prompt_output)}")
    print(f"Number of generated predictions: {len(model_outputs)}")

    bert_scores = []
    rouge_scores = []

    print("Processing BERT-scores...")
    P, R, F1 = score(cands=model_outputs, refs=prompt_output, lang="en", batch_size=128)
    bert_scores = F1.tolist()

    print("Processing ROUGE-scores")
    r_scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    for pred, ref in tqdm(zip(model_outputs, prompt_output), total=len(model_outputs)):
        rouge_score = r_scorer.score(pred, ref)
        rouge_scores.append({
            'rouge1': rouge_score['rouge1'].fmeasure,
            'rougeL': rouge_score['rougeL'].fmeasure
        })
    print("BERT score:", round(sum(bert_scores)/len(bert_scores), 3))
    rouge1_scores = [score['rouge1'] for score in rouge_scores]
    rougeL_scores = [score['rougeL'] for score in rouge_scores]

    print("ROUGE-1 score:", round(sum(rouge1_scores) / len(rouge1_scores), 3))
    print("ROUGE-L score:", round(sum(rougeL_scores) / len(rougeL_scores), 3))
    if lora:
        pd.DataFrame({"reference": prompt_output, "prediction": model_outputs, "bert_score": bert_scores, "rouge1_score": rouge1_scores, "rougeL_score": rougeL_scores}).to_parquet(f"/home/e/e0958171/miniLM-LoRA/Benchmarks/evaluation_lora.pq")
    else:
        pd.DataFrame({"reference": prompt_output, "prediction": model_outputs, "bert_score": bert_scores, "rouge1_score": rouge1_scores, "rougeL_score": rougeL_scores}).to_parquet(f"/home/e/e0958171/miniLM-LoRA/Benchmarks/evaluation_original.pq")

if __name__ == "__main__":
    model = "Qwen/Qwen-1_8B"
    dataset = mental_chat()
    prompt_input = dataset[0]
    prompt_output = dataset[1]
    print("----------Base model results----------")
    evaluate("Qwen/Qwen-1_8B", prompt_input, prompt_output, lora=False)
    print("----------SFT model results----------")
    evaluate("Qwen/Qwen-1_8B", prompt_input, prompt_output, lora=True)