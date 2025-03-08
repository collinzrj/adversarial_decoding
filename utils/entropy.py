from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datasets import load_dataset
import random
import numpy as np
from tqdm import tqdm

# Set random seed for reproducibility
random.seed(42)
torch.manual_seed(42)

# Load dataset and sample prompts
ds = load_dataset("tatsu-lab/alpaca")
prompts = random.sample(ds['train']['instruction'], 10)

# Model configuration
model_name = "Qwen/Qwen2.5-7B-Instruct"
# model_name = "meta-llama/Llama-3.1-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="cuda"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token
# Configuration parameters
batch_size = 16  # Process this many prompts at once
max_new_tokens = 32  # Generate this many new tokens for perplexity calculation

def exp(be_certain=True):
    if be_certain:
        print("Be certain")
    else:
        print("Not be certain")
    # Prepare all prompts
    all_prepared_inputs = []
    for prompt in prompts:
        if be_certain:
            suffix = " Be very certain about your results."
            messages = [
                {"role": "system", "content": "You are a helpful assistant. You are very certain about your results."},
                {"role": "user", "content": prompt + suffix}
            ]
        else:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        all_prepared_inputs.append(text)

    # Process prompts in batches to avoid OOM
    perplexity_list = []
    for batch_idx in range(0, len(all_prepared_inputs), batch_size):
        batch_end = min(batch_idx + batch_size, len(all_prepared_inputs))
        batch_texts = all_prepared_inputs[batch_idx:batch_end]
        
        # print(f"\nProcessing batch {batch_idx//batch_size + 1}/{(len(all_prepared_inputs)-1)//batch_size + 1} ({batch_idx}-{batch_end-1})")
        
        # Convert to tensor
        batch_inputs = tokenizer.batch_encode_plus(batch_texts, return_tensors="pt", padding=True).to(model.device)
        input_len = batch_inputs.input_ids.shape[1]
        
        # Generate new tokens
        with torch.no_grad():
            generation_output = model.generate(
                **batch_inputs,
                max_new_tokens=max_new_tokens,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                do_sample=False
            )
        
        # For each prompt in the batch
        for i in range(len(batch_texts)):
            prompt_idx = batch_idx + i
            prompt_text = prompts[prompt_idx]
            
            # Extract scores for this prompt
            scores = torch.stack(generation_output.scores)
            # Get the scores for this prompt
            prompt_scores = scores[:, i, :]
            
            # Get the generated token ids for this prompt
            generated_ids = generation_output.sequences[i, input_len:]
            
            # Calculate perplexity
            token_log_probs = []
            token_probs = []
            top5_ids_list = []
            # print([tokenizer.decode(generated_ids)])
            # print(generated_ids)
            # print([prompt_scores[j][token_id] for j, token_id in enumerate(generated_ids)])
            for j, token_id in enumerate(generated_ids[:1]):
                if token_id == tokenizer.eos_token_id:
                    break
                if j < len(prompt_scores):  # Ensure we don't go out of bounds
                    # print(prompt_scores[j][token_id])
                    log_prob = torch.nn.functional.log_softmax(prompt_scores[j], dim=-1)[token_id]
                    prob = torch.nn.functional.softmax(prompt_scores[j], dim=-1)[token_id]
                    top5_ids = torch.nn.functional.softmax(prompt_scores[j], dim=-1).topk(5).indices
                    token_log_probs.append(log_prob.item())
                    token_probs.append(prob.item())
                    top5_ids_list.append(top5_ids)
            # Calculate perplexity as exp(average negative log likelihood)
            if token_log_probs:
                avg_neg_log_likelihood = -sum(token_log_probs) / len(token_log_probs)
                perplexity = np.exp(avg_neg_log_likelihood)
                perplexity_list.append(perplexity)
                # Print results
                print(f"Prompt {prompt_idx+1}/100: '{prompt_text}'")
                print(f"Perplexity over {len(token_log_probs)} tokens: {perplexity:.4f}")
                print(f"Token log probs: {token_log_probs}")
                print(f"Token probs: {list(zip([tokenizer.decode(id) for id in generated_ids], token_probs))}")
                # print(f"Top 5 next tokens: {[[tokenizer.decode(id) for id in top5_ids] for top5_ids in top5_ids_list]}")
                # # Get top 5 next tokens after the prompt (for comparison with original code)
                # first_token_probs = torch.nn.functional.softmax(prompt_scores[0], dim=-1)
                # top5 = first_token_probs.topk(5)
                # print(f"Top 5 first token probabilities: {top5}")
                # print("---")
            else:
                print(f"Warning: No token log probabilities for prompt {prompt_idx+1}")
        
        # Clear memory
        torch.cuda.empty_cache()

    print(f"Perplexity list: {perplexity_list}")
    print(f"Average perplexity: {np.mean(perplexity_list)}")

exp(be_certain=True)
exp(be_certain=False)