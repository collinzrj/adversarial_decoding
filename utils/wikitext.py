from datasets import load_dataset
from sentence_transformers import SentenceTransformer

ds = load_dataset("wikimedia/wikipedia", "20231101.en")
encoder = SentenceTransformer("thenlper/gte-base")

## randomly select 10 from this
docs = ds['train'].shuffle(seed=42).select(range(100))
filtered_docs = []

for doc in docs:
    doc_len = len(encoder.tokenizer.encode(doc['text']))
    print(doc_len)
    if doc_len >= 512:
        filtered_docs.append(doc)

print("filtered docs")
print(len(filtered_docs))
