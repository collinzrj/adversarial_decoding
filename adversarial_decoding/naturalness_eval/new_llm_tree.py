from transformers import AutoTokenizer, AutoModelForCausalLM
import torch, time

class TreeNode:
    def __init__(self):
        self.children = {}
        self.key_value = None  # Stores key_value for this token (n_layers, 2, num_heads, embed_size_per_head)
        self.logits = None  # Stores logits for this token (vocab_size)

class LLMKeyValueTree:
    def __init__(self):
        self.root = TreeNode()

    def load_key_values(self, tokens):
        # print("tokens", tokens)
        node = self.root
        key_values_list = []
        for token in tokens:
            # print("token is", token)
            # print("child is", node.children)
            if token in node.children:
                node = node.children[token]
                # print(node.key_value)
                if node.key_value is not None:
                    key_values_list.append(node.key_value)
                else:
                    # If key_value is missing, cannot build further
                    break
            else:
                break

        if key_values_list:
            # Stack collected key_values along the sequence_length dimension
            key_values = torch.stack(key_values_list)  # Shape: (sequence_length, n_layers, 2, num_heads, embed_size_per_head)
            return key_values
        else:
            return None

    def store_key_values(self, tokens, key_values, tokens_logits=None):
        t0 = time.time()
        # print("tokens len", len(tokens))
        # if tokens_logits is not None:
        #     print("tokens_logits len", len(tokens_logits))
        # print("tokens", tokens)
        # key_values shape: (sequence_length, n_layers, 2, num_heads, embed_size_per_head)
        # tokens_logits shape: (sequence_length, vocab_size)
        node = self.root
        sequence_length = key_values.shape[0]
        assert sequence_length == len(tokens), f"Tokens length and key_values sequence length must match. But sequence_length is {sequence_length}, tokens len is {len(tokens)}"

        a, b, c, d = 0, 0, 0, 0
        for i, token in enumerate(tokens):
            t1 = time.time()
            if token not in node.children:
                node.children[token] = TreeNode()
            t2 = time.time()
            node = node.children[token]
            t3 = time.time()
            node.key_value = key_values[i]  # Shape: (n_layers, 2, num_heads, embed_size_per_head)
            t4 = time.time()
            if tokens_logits is not None:
                pad_len = len(tokens) - len(tokens_logits)
                if i >= pad_len:
                    node.logits = tokens_logits[i - pad_len]
            t5 = time.time()
            a += t2 - t1
            b += t3 - t2
            c += t4 - t3
            d += t5 - t4
        ttt = time.time()
        # print("store_key_values time", a, b, c, d)
        # print("total time", ttt - t0)

    def load_logits(self, tokens):
        node = self.root
        logits_list = []
        for token in tokens:
            # print(node.children, token in node.children)
            if token in node.children:
                node = node.children[token]
                if node.logits is not None:
                    logits_list.append(node.logits)
                else:
                    # print("Logits missing for token", token)
                    break
            else:
                # print("Token missing in tree", token)
                break
        logits = torch.stack(logits_list)
        return logits


def test_llm_key_value():
    tree = LLMKeyValueTree()
    causal_llm_tokenizer = AutoTokenizer.from_pretrained('gpt2')
    causal_llm = AutoModelForCausalLM.from_pretrained('gpt2').to('cuda')
    first_tokens = [1, 2, 3, 4, 5]
    second_tokens = [1, 2, 3, 8, 9]
    outputs = causal_llm(input_ids=torch.tensor([first_tokens, second_tokens]), use_cache=True)
    past_key_values = outputs.past_key_values  # (n_layers, 2, batch_size, num_heads, sequence_length, embed_size_per_head)
    batch_size, num_heads, sequence_length, embed_size_per_head = past_key_values[0][0].shape
    results = []
    for layer_i in range(len(past_key_values)):
        for idx in range(len(past_key_values[layer_i])):
            results.append(past_key_values[layer_i][idx])
    results = torch.cat(results).reshape(len(past_key_values), 2, batch_size, num_heads, sequence_length, embed_size_per_head)
    results = results.permute(2, 4, 0, 1, 3, 5)
    # (n_layers, 2, batch_size, num_heads, sequence_length, embed_size_per_head) => (batch_size, sequence_length, n_layers, 2, num_heads, embed_size_per_head) 
    tree.store_key_values(first_tokens, results[0])
    tree.store_key_values(second_tokens, results[1])
    first_tokens2 = [1, 2, 3, 4, 5, 20]
    second_tokens2 = [1, 2, 3, 8, 9, 30]
    first_tokens2_kv = tree.load_key_values(first_tokens2)
    second_tokens2_kv = tree.load_key_values(second_tokens2)
    past_key_values = torch.stack([first_tokens2_kv, second_tokens2_kv]).permute((2, 3, 0, 4, 1, 5)) # type: ignore
    n_layers = past_key_values.shape[0]
    results = []
    for i in range(n_layers):
        pair = (past_key_values[i][0], past_key_values[i][1])
        results.append(pair)
    print(causal_llm(input_ids=torch.tensor([first_tokens2[-1:], second_tokens2[-1:]]), past_key_values=results, use_cache=True).logits[:, -1, :])
    print(causal_llm(input_ids=torch.tensor([first_tokens2, second_tokens2])).logits[:, -1, :])
    print(causal_llm(input_ids=torch.tensor([first_tokens2[-1:], second_tokens2[-1:]]), past_key_values=outputs.past_key_values, use_cache=True).logits[:, -1, :])

# # past_key_values: (n_layers, 2, batch_size, num_heads, sequence_length, embed_size_per_head) => batch_kv: (batch_size, sequence_length, n_layers, 2, num_heads, embed_size_per_head) 
# def permute_to_batch_format(past_key_values):
#     batch_size, num_heads, sequence_length, embed_size_per_head = past_key_values[0][0].shape
#     results = []
#     for layer_i in range(len(past_key_values)):
#         for idx in range(len(past_key_values[layer_i])):
#             results.append(past_key_values[layer_i][idx])
#     results = torch.cat(results).reshape(len(past_key_values), 2, batch_size, num_heads, sequence_length, embed_size_per_head)
#     results = results.permute(2, 4, 0, 1, 3, 5)
#     return results

def permute_to_batch_format(past_key_values):
    """
    past_key_values: list of length n_layers,
      each element is a tuple/list of length 2 => (key, value),
      each key/value is shape: (batch_size, num_heads, seq_len, embed_per_head).

    We want to produce a tensor of shape:
      (batch_size, sequence_length, n_layers, 2, num_heads, embed_size_per_head)
    """
    # Stack key/value at each layer => shape (n_layers, 2, batch_size, num_heads, seq_len, embed_per_head)
    t0 = time.time()
    x = torch.stack([torch.stack(layer, dim=0) for layer in past_key_values], dim=0).to('cpu')
    t1 = time.time()

    # Permute to (batch_size, seq_len, n_layers, 2, num_heads, embed_per_head)
    x = x.permute(2, 4, 0, 1, 3, 5)
    t2 = time.time()
    print("permute_to_batch_format time", t1 - t0, t2 - t1)
    return x

# batch_kv: (batch_size, sequence_length, n_layers, 2, num_heads, embed_size_per_head) => past_key_values: (n_layers, 2, batch_size, num_heads, sequence_length, embed_size_per_head) 
def permute_to_nlayer_format(batch_kv, device='cuda'):
    past_key_values = batch_kv.permute((2, 3, 0, 4, 1, 5)) # type: ignore
    n_layers = past_key_values.shape[0]
    results = []
    for i in range(n_layers):
        pair = (past_key_values[i][0].to(device), past_key_values[i][1].to(device))
        results.append(pair)
    return results

def llm_tree_accelerate_last_logit(batch_tokens, tree: LLMKeyValueTree, causal_llm):
    print(batch_tokens)
    batch_tokens = [torch.tensor(tokens) for tokens in batch_tokens]
    assert all(len(lst) == len(batch_tokens[0]) for lst in batch_tokens), "All tokens should have the same length"
    # print(f"Tokens length {len(batch_tokens[0])}")
    if True:
        min_cache_len = 100000000
        prefix_batch_kv = []
        for tokens in batch_tokens:
            kv = tree.load_key_values(tokens.tolist())
            if kv is None:
                min_cache_len = 0
                break
            min_cache_len = min(kv.shape[0], min_cache_len)
            prefix_batch_kv.append(kv)
        # print("min_cache_len", min_cache_len)
        if min_cache_len > 0:
            prefix_batch_kv = torch.stack([kv[:min_cache_len] for kv in prefix_batch_kv])
            suffix_tokens = torch.stack([tokens[min_cache_len:] for tokens in batch_tokens])
            past_kv = permute_to_nlayer_format(prefix_batch_kv)
            outputs = causal_llm(input_ids=suffix_tokens, past_key_values=past_kv, use_cache=True)
            full_batch_kv = permute_to_batch_format(outputs.past_key_values)
            for tokens, kv in zip(batch_tokens, full_batch_kv):
                tree.store_key_values(tokens.tolist(), kv)
        else:
            outputs = causal_llm(input_ids=torch.stack(batch_tokens), use_cache=True)
            batch_kv = permute_to_batch_format(outputs.past_key_values)
            for tokens, kv in zip(batch_tokens, batch_kv):
                tree.store_key_values(tokens.tolist(), kv)
                kv = tree.load_key_values(tokens.tolist())
        return outputs.logits[:, -1, :]
    else:
        outputs = causal_llm(input_ids=torch.stack(batch_tokens), use_cache=True)
        return outputs.logits[:, -1, :]

def llm_tree_accelerate_logits(batch_tokens, tree: LLMKeyValueTree, causal_llm):
    f_start = time.time()
    # print(batch_tokens)
    batch_tokens = [torch.tensor(tokens) for tokens in batch_tokens]
    assert all(len(lst) == len(batch_tokens[0]) for lst in batch_tokens), "All tokens should have the same length"
    # print(f"Tokens length {len(batch_tokens[0])}")
    if True:
        min_cache_len = 100000000
        prefix_batch_kv = []
        for tokens in batch_tokens:
            kv = tree.load_key_values(tokens.tolist())
            if kv is None:
                min_cache_len = 0
                break
            min_cache_len = min(kv.shape[0], min_cache_len)
            prefix_batch_kv.append(kv)
        # print("min_cache_len", min_cache_len)
        if min_cache_len > 0:
            prefix_batch_kv = torch.stack([kv[:min_cache_len] for kv in prefix_batch_kv])
            suffix_tokens = torch.stack([tokens[min_cache_len:] for tokens in batch_tokens])
            past_kv = permute_to_nlayer_format(prefix_batch_kv, device=causal_llm.device)
            causal_start = time.time()
            suffix_tokens = suffix_tokens.to(causal_llm.device)
            outputs = causal_llm(input_ids=suffix_tokens, past_key_values=past_kv, use_cache=True)
            causal_end = time.time()
            permute_start = time.time()
            # this is slow, but executed lazily, how to speed this up?
            full_batch_kv = permute_to_batch_format(outputs.past_key_values)
            # print(full_batch_kv[0])
            permute_end = time.time()
            zip_start = time.time()
            lll = zip(batch_tokens, full_batch_kv, outputs.logits.to('cpu'))
            zip_end = time.time()
            store_start = time.time()
            for tokens, kv, tokens_logits in lll:
                sub_store_start = time.time()
                tree.store_key_values(tokens.tolist(), kv, tokens_logits)
                sub_store_end = time.time()
                # print("sub_store time", sub_store_end - sub_store_start)
            store_end = time.time()
        else:
            causal_start = time.time()
            outputs = causal_llm(input_ids=torch.stack(batch_tokens).to(causal_llm.device), use_cache=True)
            causal_end = time.time()
            permute_start = time.time()
            batch_kv = permute_to_batch_format(outputs.past_key_values)
            permute_end = time.time()
            store_start = time.time()
            for tokens, kv, tokens_logits in zip(batch_tokens, batch_kv, outputs.logits.to('cpu')):
                sub_store_start = time.time()
                tree.store_key_values(tokens.tolist(), kv, tokens_logits)
                sub_store_end = time.time()
                # print("sub_store time", sub_store_end - sub_store_start)
                # kv = tree.load_key_values(tokens.tolist())
            store_end = time.time()
        batch_cache_logits = [tree.load_logits(tokens.tolist()) for tokens in batch_tokens]
        for tokens, cache_logits in zip(batch_tokens, batch_cache_logits):
            assert(len(tokens) == len(cache_logits)), f"Tokens length {len(tokens)} and cache_logits length {len(cache_logits)} must match"
        f_end = time.time()
        print("causal llm time", causal_end - causal_start)
        print("permute time", permute_end - permute_start)
        print("store time", store_end - store_start)
        print("full time", f_end - f_start)
        return torch.stack(batch_cache_logits)
    else:
        outputs = causal_llm(input_ids=torch.stack(batch_tokens), use_cache=True)
        return outputs.logits[:, -1, :]

# def llm_tree_accelerate_logits(batch_tokens, tree, causal_llm):
#     with torch.profiler.profile(
#         activities=[
#             torch.profiler.ProfilerActivity.CPU,
#             torch.profiler.ProfilerActivity.CUDA,
#         ],
#         # on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'),  # Save logs for TensorBoard visualization
#         record_shapes=True,  # Records the input shapes
#         with_stack=True,  # Records the stack trace
#         profile_memory=True,  # Profiles memory usage
#     ) as prof:
#         res = _llm_tree_accelerate_logits(batch_tokens, tree, causal_llm)
    
#     # Print profiler results
#     print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
#     prof.export_chrome_trace("trace.json")  # Save trace for Chrome tracing
#     return res


if __name__ == '__main__':
    causal_model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    causal_llm_tokenizer = AutoTokenizer.from_pretrained(causal_model_name)
    causal_llm = AutoModelForCausalLM.from_pretrained(causal_model_name).to('cuda')
    test_tokens = causal_llm_tokenizer.encode("Hello, my name is Collin Zhang, how can I help you?")
    tree = LLMKeyValueTree()
    output = llm_tree_accelerate_logits([test_tokens[:-1]], tree, causal_llm)
    output = llm_tree_accelerate_logits([test_tokens], tree, causal_llm)