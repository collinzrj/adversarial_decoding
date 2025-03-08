import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from datasets import load_dataset
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
import json

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
    decoding = EntropyDecoding(model_name=model_name, device=file_device)
    
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
    
    target_dir = f'./data/entropy_experiment_universal.json'
    
    # First, get baseline perplexity without suffix for all prompts
    print("Calculating baseline perplexity (no suffix)...")
    baseline_train_perplexity = decoding.test_suffix(prompts=train_prompts, suffix="")
    baseline_test_perplexity = decoding.test_suffix(prompts=test_prompts, suffix="")
    
    print(f"Baseline average perplexity on training prompts: {baseline_train_perplexity:.4f}")
    print(f"Baseline average perplexity on test prompts: {baseline_test_perplexity:.4f}")
    
    # Run beam search to find best universal suffix using training prompts
    print("Running beam search to find optimal universal suffix...")
    best_cand = decoding.run_decoding(
        prompts=train_prompts,
        target="",
        beam_width=beam_width,
        max_steps=max_steps,
        top_k=top_k,
        top_p=top_p,
        should_full_sent=True,
        verbose=True
    )
    
    # Extract the optimized suffix
    universal_suffix = best_cand.seq_str.strip()
    
    print(f"\nFound universal suffix: '{universal_suffix}'")
    
    # Test the universal suffix on both training and test prompts
    suffix_train_perplexity = decoding.test_suffix(prompts=train_prompts, suffix=universal_suffix)
    suffix_test_perplexity = decoding.test_suffix(prompts=test_prompts, suffix=universal_suffix)
    
    # Calculate improvements
    train_improvement = baseline_train_perplexity - suffix_train_perplexity
    train_improvement_percent = (train_improvement / baseline_train_perplexity) * 100
    
    test_improvement = baseline_test_perplexity - suffix_test_perplexity
    test_improvement_percent = (test_improvement / baseline_test_perplexity) * 100
    
    print(f"Training set perplexity with suffix: {suffix_train_perplexity:.4f}")
    print(f"Training set perplexity reduction: {train_improvement:.4f} ({train_improvement_percent:.2f}%)")
    print(f"Test set perplexity with suffix: {suffix_test_perplexity:.4f}")
    print(f"Test set perplexity reduction: {test_improvement:.4f} ({test_improvement_percent:.2f}%)")
    
    # Get individual perplexity values for each prompt
    print("\nPerplexity details for each training prompt:")
    baseline_individual_train = decoding.test_suffix_individual(prompts=train_prompts, suffix="")
    suffix_individual_train = decoding.test_suffix_individual(prompts=train_prompts, suffix=universal_suffix)
    
    # Create a detailed results list for training prompts
    train_results = []
    for i, prompt in enumerate(train_prompts):
        baseline_perp = baseline_individual_train[i]
        suffix_perp = suffix_individual_train[i]
        improvement = baseline_perp - suffix_perp
        improvement_percent = (improvement / baseline_perp) * 100 if baseline_perp > 0 else 0
        
        print(f"Prompt {i+1}: '{prompt}'")
        print(f"  Baseline perplexity: {baseline_perp:.4f}")
        print(f"  With suffix perplexity: {suffix_perp:.4f}")
        print(f"  Improvement: {improvement:.4f} ({improvement_percent:.2f}%)")
        
        # Generate completions for comparison
        tokenizer = decoding.tokenizer
        model = decoding.llm_wrapper.model
        
        # Generate with original prompt
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            baseline_outputs = model.generate(
                input_ids, 
                max_new_tokens=32,
                do_sample=True,
                temperature=0.7,
                top_p=0.95
            )
        baseline_completion = tokenizer.decode(baseline_outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
        
        # Generate with prompt + suffix
        input_ids = tokenizer.encode(f"{prompt} {universal_suffix}", return_tensors="pt").to(model.device)
        with torch.no_grad():
            suffix_outputs = model.generate(
                input_ids, 
                max_new_tokens=32,
                do_sample=True,
                temperature=0.7,
                top_p=0.95
            )
        suffix_completion = tokenizer.decode(suffix_outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
        
        result = {
            'prompt': prompt,
            'is_training': True,
            'baseline_perplexity': baseline_perp,
            'suffix_perplexity': suffix_perp,
            'perplexity_reduction': improvement,
            'perplexity_reduction_percent': improvement_percent,
            'baseline_completion': baseline_completion,
            'suffix_completion': suffix_completion
        }
        
        train_results.append(result)
        append_to_target_dir(target_dir, result)
    
    # Evaluate on test prompts
    print("\nPerplexity details for each test prompt:")
    baseline_individual_test = decoding.test_suffix_individual(prompts=test_prompts, suffix="")
    suffix_individual_test = decoding.test_suffix_individual(prompts=test_prompts, suffix=universal_suffix)
    
    # Create a detailed results list for test prompts
    test_results = []
    for i, prompt in enumerate(test_prompts):
        baseline_perp = baseline_individual_test[i]
        suffix_perp = suffix_individual_test[i]
        improvement = baseline_perp - suffix_perp
        improvement_percent = (improvement / baseline_perp) * 100 if baseline_perp > 0 else 0
        
        print(f"Prompt {i+1}: '{prompt}'")
        print(f"  Baseline perplexity: {baseline_perp:.4f}")
        print(f"  With suffix perplexity: {suffix_perp:.4f}")
        print(f"  Improvement: {improvement:.4f} ({improvement_percent:.2f}%)")
        
        # Generate completions for comparison
        tokenizer = decoding.tokenizer
        model = decoding.llm_wrapper.model
        
        # Generate with original prompt
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            baseline_outputs = model.generate(
                input_ids, 
                max_new_tokens=32,
                do_sample=True,
                temperature=0.7,
                top_p=0.95
            )
        baseline_completion = tokenizer.decode(baseline_outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
        
        # Generate with prompt + suffix
        input_ids = tokenizer.encode(f"{prompt} {universal_suffix}", return_tensors="pt").to(model.device)
        with torch.no_grad():
            suffix_outputs = model.generate(
                input_ids, 
                max_new_tokens=32,
                do_sample=True,
                temperature=0.7,
                top_p=0.95
            )
        suffix_completion = tokenizer.decode(suffix_outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
        
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
    
    # Save overall stats
    overall_result = {
        'universal_suffix': universal_suffix,
        'training_prompts': len(train_prompts),
        'test_prompts': len(test_prompts),
        'baseline_train_perplexity': baseline_train_perplexity,
        'suffix_train_perplexity': suffix_train_perplexity,
        'train_improvement': train_improvement,
        'train_improvement_percent': train_improvement_percent,
        'baseline_test_perplexity': baseline_test_perplexity,
        'suffix_test_perplexity': suffix_test_perplexity,
        'test_improvement': test_improvement,
        'test_improvement_percent': test_improvement_percent,
    }
    
    append_to_target_dir(target_dir, overall_result)
    
    print("\n" + "=" * 50)
    print(f"Experiment Complete! Universal suffix: '{universal_suffix}'")
    print(f"Training set perplexity reduction: {train_improvement:.4f} ({train_improvement_percent:.2f}%)")
    print(f"Test set perplexity reduction: {test_improvement:.4f} ({test_improvement_percent:.2f}%)")
    print(f"Results saved to {target_dir}")
    
    # Find the prompt where the suffix works best and worst
    best_train = max(train_results, key=lambda x: x['perplexity_reduction_percent'])
    worst_train = min(train_results, key=lambda x: x['perplexity_reduction_percent'])
    
    best_test = max(test_results, key=lambda x: x['perplexity_reduction_percent'])
    worst_test = min(test_results, key=lambda x: x['perplexity_reduction_percent'])
    
    print(f"\nBest improvement on training: {best_train['perplexity_reduction_percent']:.2f}% on prompt: '{best_train['prompt']}'")
    print(f"Worst improvement on training: {worst_train['perplexity_reduction_percent']:.2f}% on prompt: '{worst_train['prompt']}'")
    
    print(f"\nBest improvement on test: {best_test['perplexity_reduction_percent']:.2f}% on prompt: '{best_test['prompt']}'")
    print(f"Worst improvement on test: {worst_test['perplexity_reduction_percent']:.2f}% on prompt: '{worst_test['prompt']}'")

if __name__ == "__main__":
    entropy_experiment(beam_width=5, max_steps=15, num_prompts=10) 