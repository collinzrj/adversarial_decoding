import pandas as pd
from adversarial_decoding.strategies.llama_guard_decoding import LlamaGuardDecoding
from adversarial_decoding.utils.utils import append_to_target_dir, file_device

def llama_guard_experiment(need_naturalness=True):
    """
    Run LlamaGuard attack experiment on HarmBench dataset.
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
            beam_width=30,
            max_steps=30,
            top_k=20,
            top_p=1,
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