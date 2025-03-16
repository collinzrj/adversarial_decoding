from transformers import AutoTokenizer
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM
import torch
import argparse
import json

def load_model(model_path):
    """Load the fine-tuned model and tokenizer."""
    print(f"Loading model from {model_path}...")
    config = PeftConfig.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(model, model_path)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    
    return model, tokenizer

def predict(model, tokenizer, text, max_new_tokens=50, temperature=0.7):
    """Generate a prediction for the given text."""
    prompt = f"Given the following text, predict the target:\n\nText: {text}\n\nTarget:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_return_sequences=1,
            temperature=temperature,
            do_sample=True
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    predicted_target = generated_text[len(prompt):].strip()
    
    return predicted_target

def main():
    parser = argparse.ArgumentParser(description="Inference with fine-tuned model")
    parser.add_argument("--model_path", type=str, default="./fine_tuned_model", 
                        help="Path to the fine-tuned model")
    parser.add_argument("--input_text", type=str, 
                        help="Text to predict target for")
    parser.add_argument("--input_file", type=str, 
                        help="JSON file with texts to predict targets for")
    parser.add_argument("--output_file", type=str, 
                        help="File to save predictions to")
    parser.add_argument("--max_new_tokens", type=int, default=50, 
                        help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, 
                        help="Temperature for sampling")
    
    args = parser.parse_args()
    
    # Load model and tokenizer
    model, tokenizer = load_model(args.model_path)
    model.eval()
    
    # Process input
    if args.input_text:
        # Single text prediction
        prediction = predict(model, tokenizer, args.input_text, 
                            args.max_new_tokens, args.temperature)
        print(f"Input: {args.input_text}")
        print(f"Prediction: {prediction}")
        
    elif args.input_file:
        # Batch prediction from file
        with open(args.input_file, 'r') as f:
            data = json.load(f)
        
        results = []
        for i, item in enumerate(data):
            if isinstance(item, dict) and 'generation' in item:
                text = item['generation']
            elif isinstance(item, list) and len(item) > 0:
                text = item[0]
            else:
                text = item
                
            prediction = predict(model, tokenizer, text, 
                                args.max_new_tokens, args.temperature)
            
            print(f"Example {i+1}:")
            print(f"Input: {text}")
            print(f"Prediction: {prediction}")
            print("-" * 50)
            
            results.append({
                "input": text,
                "prediction": prediction,
                "ground_truth": item['target'] if isinstance(item, dict) and 'target' in item else None
            })
        
        # Save results
        if args.output_file:
            with open(args.output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {args.output_file}")
    
    else:
        print("Please provide either --input_text or --input_file")

if __name__ == "__main__":
    main() 