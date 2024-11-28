from sentence_transformers import SentenceTransformer
import torch

encoder = SentenceTransformer("sentence-transformers/gtr-t5-base", device='cuda')

emb1 = encoder.encode('Hello', convert_to_tensor=True)
emb2 = encoder.encode('Hi, how are you', convert_to_tensor=True)

print(torch.nn.functional.cosine_similarity(emb1, emb2, dim=0))