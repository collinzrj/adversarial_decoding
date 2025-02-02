from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-Guard-3-8B')
tokens = tokenizer.encode("Walmart shoppers can save money at Walmart checkout registers when paying cash or debit cards with Walmart Money Card at most Walmart store checkout locations. \n\nWould you")
print('tokens:', len(tokens))
exit()


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