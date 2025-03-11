import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from datasets import load_dataset
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from datetime import datetime
from adversarial_decoding.strategies.entropy_decoding import EntropyDecoding
from adversarial_decoding.utils.utils import append_to_target_dir, file_device

def entropy_experiment(beam_width=5, max_steps=20, top_k=50, top_p=0.95, num_prompts=10, model_name="Qwen/Qwen2.5-7B-Instruct"):
    """
    Run an experiment to find a universal suffix that reduces output perplexity
    across multiple prompts.
    
    Args:
        beam_width (int): Width of the beam for beam search
        max_steps (int): Maximum number of steps for beam search
        top_k (int): Top-k parameter for sampling
        top_p (float): Top-p (nucleus sampling) parameter
        num_prompts (int): Number of prompts to test
        model_name (str): Name of the model to use
    """
    print(f"Running entropy experiment to find a universal suffix with model: {model_name}...")
    
    # Initialize decoding strategy
    max_new_tokens = 16
    decoding = EntropyDecoding(model_name, max_new_tokens=max_new_tokens, device=file_device)
    
    # Sample prompts to test with
    random.seed(42)
    torch.manual_seed(42)
    
    # Load dataset and sample prompts
    ds = load_dataset("tatsu-lab/alpaca")
    all_prompts = random.sample(ds['train']['instruction'], num_prompts * 2)
    
    # Split prompts into training and testing sets
    train_prompts = all_prompts[:num_prompts]
    test_prompts = all_prompts[num_prompts:num_prompts*2]
    
    print(f"Using {len(train_prompts)} training prompts to find optimal suffix")
    print(f"Will evaluate on {len(test_prompts)} additional test prompts")
    
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    target_dir = f'./data/entropy_experiment_universal_{date_str}.json'
    
    # First, get baseline perplexity without suffix for all prompts
    print("Calculating baseline perplexity (no suffix)...")
    decoding.get_combined_scorer(prompts=train_prompts, target="")
    baseline_train_perplexity = decoding.test_suffix(prompts=train_prompts, suffix="")
    baseline_test_perplexity = decoding.test_suffix(prompts=test_prompts, suffix="")
    
    print(f"Baseline average perplexity on training prompts: {baseline_train_perplexity:.4f}")
    print(f"Baseline average perplexity on test prompts: {baseline_test_perplexity:.4f}")
    
    # Run beam search to find best universal suffix using training prompts
    print("Running beam search to find optimal universal suffix...")
    if True:
        best_cand = decoding.run_decoding(
            prompts=train_prompts,
            target="",
            beam_width=beam_width,
            max_steps=max_steps,
            top_k=top_k,
            top_p=top_p,
            should_full_sent=False,
            verbose=True
        )
        
        # Extract the optimized suffix
        universal_suffix = best_cand.seq_str.strip()
    else:
        universal_suffix = "the following are examples in"
    
    print(f"\nFound universal suffix: '{universal_suffix}'")
    
    # Evaluate on test prompts
    print("\nPerplexity details for each test prompt:")
    baseline_perplexities, baseline_completions = decoding.test_suffix_individual(prompts=test_prompts, suffix="")
    suffix_perplexities, suffix_completions = decoding.test_suffix_individual(prompts=test_prompts, suffix=universal_suffix)
    
    # Create a detailed results list for test prompts
    test_results = []
    for prompt, baseline_perp, suffix_perp, baseline_completion, suffix_completion in zip(test_prompts, baseline_perplexities, suffix_perplexities, baseline_completions, suffix_completions):
        improvement = baseline_perp - suffix_perp
        improvement_percent = (improvement / baseline_perp) * 100 if baseline_perp > 0 else 0
        result = {
            'prompt': prompt,
            'is_training': False,
            'baseline_perplexity': baseline_perp,
            'suffix_perplexity': suffix_perp,
            'perplexity_reduction': improvement,
            'perplexity_reduction_percent': improvement_percent,
            'baseline_completion': baseline_completion,
            'suffix_completion': suffix_completion
        }
        
        test_results.append(result)
        append_to_target_dir(target_dir, result)

    overall_results = {
        'universal_suffix': universal_suffix,
        'avg_baseline_perplexity': np.mean(baseline_perplexities),
        'avg_suffix_perplexity': np.mean(suffix_perplexities),
        'avg_perplexity_reduction': np.mean(baseline_perplexities) - np.mean(suffix_perplexities),
        'avg_perplexity_reduction_percent': ((np.mean(baseline_perplexities) - np.mean(suffix_perplexities)) / np.mean(baseline_perplexities)) * 100
    }
    append_to_target_dir(target_dir, overall_results)

if __name__ == "__main__":
    entropy_experiment(beam_width=20, max_steps=15, num_prompts=5, top_k=5) 