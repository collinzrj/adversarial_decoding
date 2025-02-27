import pandas as pd
from adversarial_decoding.strategies.naturalness_decoding import NaturalnessDecoding
from adversarial_decoding.utils.utils import append_to_target_dir, file_device

def naturalness_experiment(score_target=0.4):
    """
    Run naturalness experiment targeting a specific naturalness score.
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
            beam_width=30,
            max_steps=30,
            top_k=20,
            top_p=1,
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