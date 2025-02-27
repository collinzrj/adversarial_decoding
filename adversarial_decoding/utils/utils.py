import torch
import time
import os

# File device configuration
file_device = os.environ.get('FILE_DEVICE', 'cuda:0')
if file_device == 'cuda:0':
    second_device = 'cuda:1'
else:
    second_device = 'cuda:0'

def top_k_top_p_filtering(logits: torch.Tensor, top_k: int = 50, top_p: float = 0.9):
    """
    Given a 1D tensor of `logits`, return the indices of tokens that
    satisfy both the Top-K and Top-P filtering constraints.
    """
    probs = torch.softmax(logits, dim=-1)
    topk_probs, topk_indices = torch.topk(probs, top_k)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    top_p_cutoff = torch.sum(cumulative_probs <= top_p).item()
    if top_p_cutoff < 1:
        top_p_cutoff = 1
    top_p_indices = sorted_indices[:top_p_cutoff]
    top_k_set = set(topk_indices.tolist())
    top_p_set = set(top_p_indices.tolist())
    intersection_set = top_k_set.intersection(top_p_set)

    if len(intersection_set) == 0:
        raise ValueError("No tokens satisfy both Top-K and Top-P constraints.")

    intersection_indices = torch.tensor(list(intersection_set), dtype=torch.long)
    intersection_probs = probs[intersection_indices]
    sorted_intersection_probs, sorted_order = torch.sort(intersection_probs, descending=True)
    intersection_indices = intersection_indices[sorted_order.cpu()]
    return intersection_indices


class MyTimer:
    """Simple timer to measure code sections."""
    def __init__(self):
        self.timer_dict = {}

    def start(self, name):
        if name in self.timer_dict:
            self.timer_dict[name][1] = time.time()
        else:
            self.timer_dict[name] = [0, time.time()]
    
    def stop(self, name):
        self.timer_dict[name][0] += time.time() - self.timer_dict[name][1]
        self.timer_dict[name][1] = None

    def display(self):
        for name, (total_time, _) in self.timer_dict.items():
            print(f"[{name}] {total_time:.4f}s")


class ModelSwitcher:
    def __init__(self, models):
        self.models = models

    def switch_to(self, idx):
        return
        torch.cuda.synchronize()
        memory_allocated = torch.cuda.memory_allocated()
        print(f"Allocated memory before switch: {memory_allocated / 1024**3:.2f} GB")

        for i in range(len(self.models)):
            self.models[i].to('cpu')
        torch.cuda.synchronize()
        memory_allocated = torch.cuda.memory_allocated()
        print(f"Allocated memory after switch: {memory_allocated / 1024**3:.2f} GB")

        self.models[idx].to(file_device)
        torch.cuda.synchronize()
        memory_allocated = torch.cuda.memory_allocated()
        print(f"Allocated memory after switch: {memory_allocated / 1024**3:.2f} GB")


def highest_avg_cos_sim(embs):
    avg_emb = torch.mean(embs, dim=0)
    return torch.nn.functional.cosine_similarity(avg_emb, embs).mean().item()


def compute_doc_embs(encoder, documents):
    return encoder.encode(documents, convert_to_tensor=True, normalize_embeddings=True)


import os, json
def append_to_target_dir(target_dir, dict):
    if os.path.exists(target_dir):
        with open(target_dir, 'r') as f:
            res = json.load(f)
    else:
        res = []
    res.append(dict)
    with open(target_dir, 'w') as f:
        json.dump(res, f, indent=4) 