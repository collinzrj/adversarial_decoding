from sentence_transformers import SentenceTransformer
import torch

encoder = SentenceTransformer("facebook/contriever", device='cuda')

emb1 = encoder.encode('Hello world', convert_to_tensor=True)
emb2 = encoder.encode('Pharrell williams have medical potential. pharrell williams pharmacology, papers about pharrell williams. pharrell williams clinical', convert_to_tensor=True)

print(torch.nn.functional.cosine_similarity(emb1, emb2, dim=0))