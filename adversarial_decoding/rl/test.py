# import torch
# import torch.nn.functional as F

# # Generate a random vector and 100 other random vectors
# torch.manual_seed(0)
# v = torch.rand(10)  # Random vector of dimension 10
# vectors = torch.rand(100, 10)  # 100 random vectors of dimension 10

# # Compute the cosine similarity of v with each of the 100 vectors
# cos_sim_individual = [F.cosine_similarity(v, u, dim=0) for u in vectors]
# mean_cos_sim = torch.mean(torch.tensor(cos_sim_individual))

# # Compute the mean vector
# mean_vector = torch.mean(vectors, dim=0)

# # Compute the cosine similarity of v with the mean vector
# cos_sim_with_mean_vector = F.cosine_similarity(v, mean_vector, dim=0)

# print("Mean cosine similarity with each vector:", mean_cos_sim.item())
# print("Cosine similarity with the mean vector:", cos_sim_with_mean_vector.item())

import torch

# Generate a random vector and 100 other random vectors
torch.manual_seed(0)
v = torch.rand(10)  # Random vector of dimension 10
vectors = torch.rand(100, 10)  # 100 random vectors of dimension 10

# Normalize v and each vector in vectors
v_normalized = v / torch.norm(v)
vectors_normalized = vectors / torch.norm(vectors, dim=1, keepdim=True)

# Compute cosine similarities as dot products (since vectors are normalized)
cos_sim_individual = torch.mm(v_normalized.view(1, -1), vectors_normalized.t()).squeeze()

# Calculate the mean cosine similarity
mean_cos_sim = cos_sim_individual.mean()

# Compute the mean vector and normalize it
mean_vector = vectors_normalized.mean(dim=0)
mean_vector_normalized = mean_vector / torch.norm(mean_vector)

# Compute cosine similarity of v with the mean vector
cos_sim_with_mean_vector = torch.dot(v_normalized, mean_vector_normalized)

print(mean_cos_sim.item(), cos_sim_with_mean_vector.item(), torch.dot(v_normalized, mean_vector))
