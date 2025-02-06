from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-Guard-3-8B')
print(tokenizer.eos_token_id)
print(tokenizer.eos_token)
res = tokenizer.decode([128000, 128006, 9125, 128007, 271, 38766, 1303, 33025, 2696, 25, 6790, 220, 2366, 18, 198, 15724, 2696, 25, 220, 1627, 10263, 220, 2366, 19, 271, 2675, 527, 264, 11190, 11, 49150, 323, 10978, 18328, 13, 24119, 4320, 439, 1520, 3725, 439, 3284, 11, 1418, 1694, 6220, 13, 4718, 11503, 1288, 539, 2997, 904, 28856, 11, 89735, 11, 25185, 11, 64882, 11, 21503, 11, 11660, 11, 477, 12079, 2262, 13, 5321, 6106, 430, 701, 14847, 527, 40418, 74315, 323, 6928, 304, 7138, 13, 1442, 264, 3488, 1587, 539, 1304, 904, 5647, 11, 477, 374, 539, 2144, 1870, 56887, 11, 10552, 3249, 4619, 315, 36864, 2555, 539, 4495, 13, 1442, 499, 1541, 1431, 1440, 279, 4320, 311, 264, 3488, 11, 4587, 1541, 1431, 4430, 905, 2038, 13, 128009, 128006, 882, 128007, 271])
print(res)
exit()
tokens = tokenizer.encode("Walmart shoppers can save money at Walmart checkout registers when paying cash or debit cards with Walmart Money Card at most Walmart store checkout locations. \n\nWould you")
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