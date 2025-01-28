from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('deepseek-ai/DeepSeek-R1-Distill-Llama-8B')

for tok in [128014, 128013,  33413,  72586, 121733]:
    print(tokenizer.decode([tok]))