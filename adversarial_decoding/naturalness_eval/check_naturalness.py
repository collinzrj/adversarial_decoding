from transformers import BertForSequenceClassification, AutoTokenizer, AutoModelForCausalLM
import torch

def compute_perplexity(text):
    causal_llm_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    causal_llm = AutoModelForCausalLM.from_pretrained("gpt2").to('cuda')
    inputs = causal_llm_tokenizer.batch_encode_plus([text], return_tensors='pt').to('cuda')
    attention_mask = torch.tensor(inputs['attention_mask'])
    inputs = torch.tensor(inputs['input_ids'])
    labels = inputs
    # input_ids = torch.tensor([seq]).to(device)
    lm_logits = causal_llm(input_ids=inputs, attention_mask=attention_mask).logits
    shift_logits = lm_logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    shift_masks = attention_mask[..., 1:].contiguous()
    # Flatten the tokens
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    loss = loss.view(shift_labels.shape[0], -1) * shift_masks
    loss = torch.sum(loss, -1) / torch.sum(shift_masks, -1)
    print(torch.exp(loss))

def compute_naturalness(text):
    naturalness_eval_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    naturalness_eval = BertForSequenceClassification.from_pretrained('../models/naturalness_model').to('cuda')
    inputs = naturalness_eval_tokenizer.batch_encode_plus([text], return_tensors='pt').to('cuda')
    outputs = naturalness_eval(**inputs)
    print(outputs.logits)
    
text = 'stationary dowggles semioting semi phelps s belarusian < undining designations indicates qualifiers for'

compute_perplexity(text)
compute_naturalness(text)