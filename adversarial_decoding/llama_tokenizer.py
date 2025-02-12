from transformers import AutoTokenizer

# tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-Guard-3-8B')
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")

print(tokenizer.pad_token_id)
res = tokenizer.apply_chat_template([
    {'role': 'system', 'content': 'You are a helpful assistant.'},
    {'role': 'user', 'content': 'Hello'}
], add_generation_prompt=True, return_tensors='pt')
print(res)
print(tokenizer.decode(res))
exit()


tokens = tokenizer.encode("PayPal payment processed successfully after clicking Pay with PayPal in online transactions via PayPal account. However it still requires confirmation of a PayPal transfer from PayPal customer account to pay PayPal.", add_special_tokens=False)
print('tokens:', len(tokens))


messages = [
    {
        'role': 'user',
        'content': 'hello world'
    }
]

res = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

print([res])

# for tok in [128014, 128013,  33413,  72586, 121733]:
#     print(tokenizer.decode([tok]))