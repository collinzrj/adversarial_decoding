import pandas as pd
from adversarial_decoding.strategies.llama_guard_decoding import LlamaGuardDecoding
from adversarial_decoding.utils.utils import append_to_target_dir, file_device

def llama_guard_experiment(need_naturalness=False, beam_width=5, max_steps=20, top_k=50, top_p=0.95):
    """
    Run an experiment to test LlamaGuard evasion.
    
    Args:
        need_naturalness (bool): Whether to enforce naturalness in the outputs
        beam_width (int): Width of the beam for beam search
        max_steps (int): Maximum number of steps for beam search
        top_k (int): Top-k parameter for sampling
        top_p (float): Top-p (nucleus sampling) parameter
    """
    print("Running LlamaGuard attack experiment...")
    
    # Set up natural tag based on configuration
    if need_naturalness:
        natural_tag = 'natural'
    else:
        natural_tag = 'unnatural'
    
    # Initialize decoding strategy
    attack = LlamaGuardDecoding(need_naturalness, device=file_device)
    
    # Load dataset
    df = pd.read_csv('./datasets/harmbench.csv')
    prompts = list(df[df['FunctionalCategory'] == 'standard']['Behavior'])
    target_dir = f'./data/llama_guard_attack_{natural_tag}_final.json'
    
    # Run decoding for each prompt
    for prompt in prompts:
        print("-" * 50)
        print(f"Processing prompt: {prompt}")
        
        best_cand = attack.run_decoding(
            prompt=prompt,
            target='',
            beam_width=beam_width,
            max_steps=max_steps,
            top_k=top_k,
            top_p=top_p,
            should_full_sent=False
        )
        
        # Save results
        result_str = best_cand.seq_str
        print(f"Result: {result_str}")
        append_to_target_dir(target_dir, {
            'prompt': prompt,
            'generation': result_str,
            'llama_guard_score': best_cand.llama_guard_score or 0.0,
            'naturalness': best_cand.naturalness or 0.0
        }) 