#!/usr/bin/env python3
"""
Script to run the entropy experiment, which finds a universal suffix that reduces
model output perplexity across multiple prompts.
"""

import argparse
from adversarial_decoding.experiments.entropy_experiment import entropy_experiment
from transformers.utils import logging
logging.set_verbosity_error() 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run entropy experiment to find a universal suffix that reduces perplexity across multiple prompts.")
    parser.add_argument("--beam_width", type=int, default=10, help="Width of the beam for beam search")
    parser.add_argument("--max_steps", type=int, default=10, help="Maximum number of steps for beam search")
    parser.add_argument("--top_k", type=int, default=20, help="Top-k parameter for sampling")
    parser.add_argument("--top_p", type=float, default=1, help="Top-p parameter for sampling")
    parser.add_argument("--num_prompts", type=int, default=10, help="Number of prompts in each set (training and testing)")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Name of the model to use")
    
    args = parser.parse_args()
    
    print("Starting entropy experiment to find a universal suffix with the following parameters:")
    print(f"  Beam width: {args.beam_width}")
    print(f"  Max steps: {args.max_steps}")
    print(f"  Top k: {args.top_k}")
    print(f"  Top p: {args.top_p}")
    print(f"  Number of prompts per set: {args.num_prompts}")
    print(f"  Model name: {args.model_name}")
    print("-" * 50)
    
    entropy_experiment(
        beam_width=args.beam_width,
        max_steps=args.max_steps,
        top_k=args.top_k,
        top_p=args.top_p,
        num_prompts=args.num_prompts,
        model_name=args.model_name
    ) 