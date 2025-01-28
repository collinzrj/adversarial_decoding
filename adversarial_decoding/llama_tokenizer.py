from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-Guard-3-8B')

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