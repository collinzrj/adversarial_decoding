import json
import faiss
import random
from datetime import datetime
from nltk.translate.bleu_score import sentence_bleu
from sentence_transformers import SentenceTransformer
from adversarial_decoding.strategies.emb_inv_decoding import EmbInvDecoding
from adversarial_decoding.utils.utils import append_to_target_dir, file_device

def measure_new_trigger_asr(trig, adv_texts, model, cpu_index):
    """
    Measure attack success rate for RAG poisoning.
    """
    # Use a simpler approach without attack for demonstration
    test_trigger = f"what is {trig}"
    test_emb = model.encode([test_trigger], normalize_embeddings=True)
    
    # Search for nearest neighbors
    D, I = cpu_index.search(test_emb, 3)
    
    # Check if the adversarial texts are retrieved
    is_success = False
    for i in I[0]:
        if i < len(adv_texts):
            is_success = True
            break
            
    return is_success

def emb_inv_experiment(should_natural=False, beam_width=5, max_steps=20, top_k=50, top_p=1):
    """
    Run an experiment to test retrieval-augmented generation.
    
    Args:
        should_natural (bool): Whether to enforce naturalness in the outputs
        beam_width (int): Width of the beam for beam search
        max_steps (int): Maximum number of steps for beam search
        top_k (int): Top-k parameter for sampling
        top_p (float): Top-p (nucleus sampling) parameter
    """
    print("Running Embedding Inversion experiment...")
    
    # Set up trigger and control text
    trigger = "tesla"
    control_text = "I hate {trig}."
    
    # Initialize encoder
    encoder = SentenceTransformer("thenlper/gte-base", device=file_device)
    
    # Initialize decoding strategy
    attack = EmbInvDecoding(
        encoder=encoder, 
        device=file_device,
        should_natural=should_natural
    )
    
    # Set up FAISS index for evaluation
    dimension = 768  # Contriever dimension
    cpu_index = faiss.IndexFlatIP(dimension)
    
    # Sample test prompts
    
    from datasets import load_dataset

    marco_ds = load_dataset("microsoft/ms_marco", "v2.1")
    marco_ds = marco_ds['train'].shuffle(seed=42).select(range(1000))
    random.seed(42)
    target_docs = [random.choice(doc['passages']['passage_text']) for doc in marco_ds]
    
    adv_texts = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target_dir = f'./data/emb_inv_attack_{"natural" if should_natural else "unnatural"}_{timestamp}.json'
    
    # Run decoding for each prompt
    for target in target_docs[:100]:
        print("-" * 50)
        target = encoder.tokenizer.decode(encoder.tokenizer.encode(target, add_special_tokens=False)[:32])
        print(f"Processing prompt: {target}")
        
        # Set up combined scorer
        
        # Run decoding
        prompt = 'tell me a story'
        for _ in range(5):
            best_cand = attack.run_decoding(
                prompt=prompt,
                target=target,
                beam_width=beam_width,
                max_steps=max_steps,
                top_k=top_k,
                top_p=top_p,
                should_full_sent=False,
                verbose=False
            )
            print([best_cand.seq_str])
            prompt = f"generate a similar sentence: {best_cand.seq_str}"
        
        # Save results
        result_str = best_cand.seq_str
        adv_texts.append(result_str)
        print(f"Result: {result_str}")
        attack.llm_wrapper.template_example()
        bleu_score = sentence_bleu([target], result_str)
        
        # Add embeddings to index for evaluation
        emb = encoder.encode([result_str], normalize_embeddings=True)
        cpu_index.add(emb)
        
        append_to_target_dir(target_dir, {
            'target': target,
            'generation': result_str,
            'cos_sim': best_cand.cos_sim or 0.0,
            'naturalness': best_cand.naturalness or 0.0,
            'bleu_score': bleu_score,
            'beam_width': beam_width,
            'max_steps': max_steps,
            'top_k': top_k,
            'top_p': top_p
        })
    
    # Test attack success rate
    asr = measure_new_trigger_asr(trigger, adv_texts, encoder, cpu_index)
    print(f"Attack success rate: {asr}") 