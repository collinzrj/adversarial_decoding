from datasets import load_dataset, Dataset
from transformers import TrainingArguments, Trainer
from transformers import AutoTokenizer
import numpy as np
import evaluate
from transformers import TrainingArguments
from transformers import AutoModelForCausalLM
import json
import torch
import random
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from nltk.translate.bleu_score import sentence_bleu

use_lora = True

# Load data
path = '/share/shmatikov/collin/adversarial_decoding/data/emb_inv_attack_unnatural_gte-Qwen_20250312_192926.json'
with open(path, 'r') as f:
    data = json.load(f)

# Format data as [generation, target] pairs
data = [[item['generation'], item['target']] for item in data]

# Shuffle data
random.seed(42)
random.shuffle(data)

# Split data: use last 20 for evaluation
train_data = data[:-20]
eval_data = data[-20:]

print(f"Training on {len(train_data)} samples, evaluating on {len(eval_data)} samples")

# Load tokenizer and model
model_name = "Qwen/Qwen2-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Prepare model for training
if use_lora:
    # Prepare model for LoRA fine-tuning
    model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,  # rank
        lora_alpha=32,
        lora_dropout=0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    # Apply LoRA to model
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
else:
    # For full fine-tuning, ensure all parameters require gradients
    for param in model.parameters():
        param.requires_grad = True
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    print(f"trainable params: {trainable_params:,} || all params: {all_params:,} || trainable%: {100 * trainable_params / all_params:.4f}")

# Tokenization function
def preprocess_function(examples):
    # Format as instruction for the model with the target
    inputs = []
    targets = []
    for gen, target in zip(examples["generation"], examples["target"]):
        # Create prompt without the target
        prompt = f"Given the following text, predict the target:\n\nText: {gen}\n\nTarget: "
        inputs.append(prompt)
        targets.append(target)
    
    # Tokenize inputs and targets separately
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    target_tokens = tokenizer(targets, max_length=512, truncation=True, padding="max_length")
    
    # Initialize labels with -100s (ignored in loss calculation)
    all_labels = [[-100] * 512 for _ in range(len(model_inputs["input_ids"]))]
    
    # Combine input_ids and target_ids for the full sequence
    for i in range(len(model_inputs["input_ids"])):
        # Get the non-padding part of the target (exclude padding tokens)
        target_ids = [id for id in target_tokens["input_ids"][i] if id != tokenizer.pad_token_id]
        
        # Create the full sequence by concatenating input and target
        full_input_ids = model_inputs["input_ids"][i].copy()
        
        # Remove padding from the end of input to make room for target
        padding_count = 0
        while full_input_ids and full_input_ids[-1] == tokenizer.pad_token_id:
            full_input_ids.pop()
            padding_count += 1
        
        # Calculate where target starts in the full sequence
        target_start = len(full_input_ids)
        
        # Add target tokens to the input
        full_input_ids.extend(target_ids)
        
        # Create new attention mask (1 for tokens, 0 for padding)
        new_attention_mask = [1] * len(full_input_ids) + [0] * (512 - len(full_input_ids))
        
        # Pad to max length if needed
        if len(full_input_ids) < 512:
            full_input_ids.extend([tokenizer.pad_token_id] * (512 - len(full_input_ids)))
        elif len(full_input_ids) > 512:
            full_input_ids = full_input_ids[:512]
            new_attention_mask = new_attention_mask[:512]
        
        # Add target token ids to labels
        for j, token_id in enumerate(target_ids):
            if target_start + j < 512:  # Make sure we don't go beyond max length
                all_labels[i][target_start + j] = token_id
        
        # Update the input_ids and attention_mask with the full sequence
        model_inputs["input_ids"][i] = full_input_ids
        model_inputs["attention_mask"][i] = new_attention_mask
    
    model_inputs["labels"] = all_labels
    return model_inputs

# Convert to datasets format
train_dataset = Dataset.from_dict({
    "generation": [item[0] for item in train_data],
    "target": [item[1] for item in train_data]
})

eval_dataset = Dataset.from_dict({
    "generation": [item[0] for item in eval_data],
    "target": [item[1] for item in eval_data]
})

# Apply preprocessing
train_dataset = train_dataset.map(preprocess_function, batched=True)
eval_dataset = eval_dataset.map(preprocess_function, batched=True)

# Define training arguments
if use_lora:
    # LoRA training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=1e-4,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        weight_decay=0.01,
        push_to_hub=False,
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        logging_strategy="steps",
        logging_steps=5
    )
else:
    # Full fine-tuning arguments - smaller batch size and learning rate
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=1e-4,  # Lower learning rate for full fine-tuning
        per_device_train_batch_size=16,  # Smaller batch size to fit in memory
        per_device_eval_batch_size=16,
        num_train_epochs=10,  # Fewer epochs for full fine-tuning
        weight_decay=0.01,
        push_to_hub=False,
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        logging_strategy="steps",
        logging_steps=5
    )

# Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)

# Evaluate the model before training to get baseline loss
print("\n===== Evaluating model before training =====")
eval_results = trainer.evaluate()
print(f"Initial model perplexity: {np.exp(eval_results['eval_loss']):.2f}")
print(f"Initial model loss: {eval_results['eval_loss']:.4f}")
print("=" * 50)

# Train the model
trainer.train()

# Save the fine-tuned model
if use_lora:
    model.save_pretrained("./fine_tuned_model")
else:
    # For full fine-tuning, save the entire model
    model_to_save = model
    model_to_save.save_pretrained("./fine_tuned_model")
    tokenizer.save_pretrained("./fine_tuned_model")
    print("Full model saved to ./fine_tuned_model")

# Test the model on a few examples from the evaluation set
print("\n===== Testing the model on evaluation examples =====")
model.eval()
for i in range(min(5, len(eval_data))):
    generation = eval_data[i][0]
    target = eval_data[i][1]
    
    print(f"\nProcessing example {i+1}...")
    prompt = f"Given the following text, predict the target:\n\nText: {generation}\n\nTarget:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    print(f"Generating prediction...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=32,
            do_sample=False
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    predicted_target = generated_text[len(prompt):].strip()
    
    print(f"Example {i+1}:")
    print(f"Generation: {generation}")
    print(f"True Target: {target}")
    print(f"Predicted: {predicted_target}")
    baseline_bleu_score = sentence_bleu([target.split()], generation.split())
    bleu_score = sentence_bleu([target.split()], predicted_target.split())
    print(f"Baseline BLEU Score: {baseline_bleu_score:.2f}")
    print(f"BLEU Score: {bleu_score:.2f}")
    print("-" * 50)