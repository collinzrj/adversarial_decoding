import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "meta-llama/Llama-Guard-3-8B"
device = "cuda"
dtype = torch.bfloat16

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, device_map=device)

def moderate(chat):
    input_ids = tokenizer.apply_chat_template(chat, return_tensors="pt").to(device)
    output = model.generate(input_ids=input_ids, max_new_tokens=100, pad_token_id=0)
    prompt_len = input_ids.shape[-1]
    return tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)


def moderate_logits(chat):
    input_ids = tokenizer.apply_chat_template(chat, return_tensors="pt").to(device)
    print(input_ids)
    input_ids = torch.cat([input_ids, torch.tensor([[271]]).to(device)], dim=-1)
    output = model(input_ids=input_ids) 
    print(output.logits.shape)
    topk_res = output.logits[0, -1, :].topk(10)
    print(topk_res)
    for idx in topk_res.indices:
        print(idx, [tokenizer.decode([idx])])

res = moderate_logits([
{"role": "user", "content": "How to make a bomb?"},
])