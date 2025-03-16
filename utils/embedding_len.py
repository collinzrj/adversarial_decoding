from datasets import load_dataset
import random
from sentence_transformers import SentenceTransformer
import torch

marco_ds = load_dataset("microsoft/ms_marco", "v2.1")
marco_ds = marco_ds['train'].shuffle(seed=42).select(range(1000))
random.seed(42)

all_docs = [random.choice(doc['passages']['passage_text']) for doc in marco_ds][:200]
train_docs = all_docs[:100]
test_doc = all_docs[100] 

encoder = SentenceTransformer("thenlper/gte-base")

for doc in all_docs:
    doc_len = len(encoder.tokenizer.encode(doc))
    print(doc_len)

# test_len_embs = []
# for seq_len in range(2, 10):
#     encoded = encoder.tokenizer.batch_encode_plus([test_doc], max_length=seq_len, padding=True, truncation=True, add_special_tokens=False)
#     texts = encoder.tokenizer.batch_decode(encoded['input_ids'], skip_special_tokens=True)
#     print(texts)
#     embs = encoder.encode(texts, convert_to_tensor=True)[0]
#     test_len_embs.append(embs)

# for seq_len in range(2, 10):
#     encoded = encoder.tokenizer.batch_encode_plus(train_docs, max_length=seq_len, padding=True, truncation=True, add_special_tokens=False)
#     texts = encoder.tokenizer.batch_decode(encoded['input_ids'], skip_special_tokens=True)
#     embs = encoder.encode(texts, convert_to_tensor=True)
#     len_avg_emb = embs.mean(dim=0)
#     cos_sims = torch.nn.functional.cosine_similarity(torch.stack(test_len_embs), len_avg_emb.unsqueeze(0))
#     print(cos_sims)
