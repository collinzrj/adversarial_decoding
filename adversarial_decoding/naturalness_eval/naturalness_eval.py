import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_scheduler, GPT2ForSequenceClassification, GPT2Tokenizer
from datasets import load_from_disk, Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def train():
    BERT = False
    FULL = True
    if BERT:
        # Load BERT tokenizer and model
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
        if not FULL:
            for param in model.bert.parameters():
                param.requires_grad = False
        model_name = 'bert_naturalness_model'
    else:
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
        model = GPT2ForSequenceClassification.from_pretrained('gpt2', num_labels=2)
        model.config.pad_token_id = tokenizer.eos_token_id
        if not FULL:
            for param in model.transformer.parameters():
                param.requires_grad = False
        model_name = 'gpt2_naturalness_model'


    # Tokenize the data
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=20, return_tensors="pt")

    # Prepare train and test datasets
    train_dataset = load_from_disk('./data/naturalness_train_dataset').shuffle(seed=42).map(tokenize_function, batched=True)
    test_dataset = load_from_disk('./data/naturalness_test_dataset').shuffle(seed=42).map(tokenize_function, batched=True)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # Convert to PyTorch DataLoader
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=8)
    test_dataloader = DataLoader(test_dataset, batch_size=8)

    # Define optimizer and learning rate scheduler
    optimizer = AdamW(model.parameters(), lr=5e-5)

    num_epochs = 5
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        "linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    
    model.to(device)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        for idx, batch in enumerate(train_dataloader):
            # Move batch to the same device as the model
            batch = {
                'input_ids': torch.stack(batch['input_ids']).to(device).t(),
                'attention_mask': torch.stack(batch['attention_mask']).to(device).t(),
                'labels': batch['label'].to(device)
            }

            outputs = model(**batch)
            loss = outputs.loss
            print(epoch, idx, loss.item())
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        print(f"Epoch {epoch + 1} completed")

    # Evaluation
    model.eval()
    predictions, true_labels = [], []

    for batch in test_dataloader:
        batch = {
            'input_ids': torch.stack(batch['input_ids']).to(device).t(),
            'attention_mask': torch.stack(batch['attention_mask']).to(device).t(),
            'labels': batch['label'].to(device)
        }
        
        with torch.no_grad():
            outputs = model(**batch)
        
        logits = outputs.logits
        predicted_labels = torch.argmax(logits, dim=-1)
        predictions.extend(predicted_labels.cpu().numpy())
        true_labels.extend(batch["labels"].cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average="binary")

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    if FULL:
        style = 'full'
    else:
        style = 'linear'
    model.save_pretrained(f'./models/{style}_{model_name}')


def further_train(unnatural_samples, natural_samples):

    # Load BERT tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('./models/naturalness_model_new', num_labels=2).to('cuda')

    # Tokenize the data
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=16, return_tensors="pt").to('cuda')

    train_dataset = Dataset.from_list([{'text': x, 'label': 0} for x in unnatural_samples] + [{'text': x, 'label': 1} for x in natural_samples]).shuffle(seed=42).map(tokenize_function, batched=True)

    # Convert to PyTorch DataLoader
    train_dataloader = DataLoader(train_dataset, batch_size=8)

    # Define optimizer and learning rate scheduler
    optimizer = AdamW(model.parameters(), lr=5e-5)

    num_epochs = 5
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        "linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    device = 'cuda'

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        for idx, batch in enumerate(train_dataloader):
            # Move batch to the same device as the model
            batch = {
                'input_ids': torch.stack(batch['input_ids']).to(device).t(),
                'attention_mask': torch.stack(batch['attention_mask']).to(device).t(),
                'labels': batch['label'].to(device)
            }

            outputs = model(**batch)
            loss = outputs.loss
            print(epoch, idx, loss.item())
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        print(f"Epoch {epoch + 1} completed")

    model.save_pretrained('./models/naturalness_model_new')