import pandas as pd
from adversarial_decoding.strategies.naturalness_decoding import NaturalnessDecoding
from adversarial_decoding.utils.utils import append_to_target_dir, file_device

def naturalness_experiment(score_target=0.4, beam_width=5, max_steps=20, top_k=50, top_p=0.95):
    """
    Run an experiment to test naturalness optimization.
    
    Args:
        score_target (float): Target naturalness score
        beam_width (int): Width of the beam for beam search
        max_steps (int): Maximum number of steps for beam search
        top_k (int): Top-k parameter for sampling
        top_p (float): Top-p (nucleus sampling) parameter
    """
    print(f"Running naturalness experiment with target score: {score_target}...")
    
    # Initialize decoding strategy
    attack = NaturalnessDecoding(score_target=score_target, device=file_device)
    
    # Sample prompts to test with
    prompts = [
        "Tell me a short story",
        "Explain quantum physics",
        "Write a poem about nature"
    ]
    
    target_dir = f'./data/naturalness_experiment_{score_target}.json'
    
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
            should_full_sent=True
        )
        
        # Save results
        result_str = best_cand.seq_str
        print(f"Result: {result_str}")
        append_to_target_dir(target_dir, {
            'prompt': prompt,
            'generation': result_str,
            'target_naturalness': score_target,
            'naturalness': best_cand.naturalness or 0.0
        })
    
    # Store the score target for future runs
    attack.score_target = score_target 