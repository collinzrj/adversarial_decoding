from sentence_transformers import SentenceTransformer
import faiss, datasets
from tqdm import tqdm

# 1. Set up Contriever
ds = datasets.load_dataset("microsoft/ms_marco", "v2.1")
device = 'cuda'
model_name = "facebook/contriever"
model = SentenceTransformer(model_name, device=device)

# 3. Create a FAISS index
# Assuming the embedding dimension is 768 for Contriever
dimension = 768
index = faiss.IndexFlatIP(dimension)

# 4. Process the database in batches
batch_size = 5000  # Adjust based on your available memory
texts = []
for passages in ds['train']['passages']: # type: ignore
    for text in passages['passage_text']:
        texts.append(text)

for i in tqdm(range(0, len(texts), batch_size)):
    batch = texts[i:i+batch_size]
    encoded_batch = model.encode(batch, normalize_embeddings=True)
    index.add(encoded_batch) # type: ignore

# 5. Save the FAISS index
faiss.write_index(index, "./data/contriever_ms_marco.faiss")