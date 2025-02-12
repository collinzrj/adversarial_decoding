from sentence_transformers import SentenceTransformer
import faiss, datasets
from tqdm import tqdm
import torch

# 1. Set up Contriever
ds = datasets.load_dataset("microsoft/ms_marco", "v2.1")
device = 'cuda'
name = 'gte'
if name == 'qwen':
    model_name = "Alibaba-NLP/gte-Qwen2-1.5B-instruct"
    out_path = "./data/qwen_ms_marco_test_float32.faiss"
    dimension = 1536
elif name == 'gte':
    model_name = "thenlper/gte-base"
    out_path = "./data/qwen_ms_marco_test_float32.faiss"
    dimension = 768
elif name == 'gtr':
    model_name = "sentence-transformers/gtr-t5-base"
    out_path = "./data/gtr_ms_marco_test_float32.faiss"
    dimension = 768
elif name == 'st5':
    model_name = "sentence-transformers/sentence-t5-base"
    out_path = "./data/st5_ms_marco_test_float32.faiss"
    dimension = 768
model = SentenceTransformer(model_name, trust_remote_code=True, device=device)

# 3. Create a FAISS index
# Assuming the embedding dimension is 768 for Contriever
index = faiss.IndexFlatIP(dimension)

# 4. Process the database in batches
batch_size = 1000  # Adjust based on your available memory
texts = []
for passages in ds['test']['passages']: # type: ignore
    for text in passages['passage_text']:
        texts.append(text)

# texts = texts[:5000]

for i in tqdm(range(0, len(texts), batch_size)):
    batch = texts[i:i+batch_size]
    encoded_batch = model.encode(batch, normalize_embeddings=True)
    index.add(encoded_batch) # type: ignore

# 5. Save the FAISS index
faiss.write_index(index, out_path)