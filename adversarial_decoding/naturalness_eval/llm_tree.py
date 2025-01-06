from transformers import AutoTokenizer, AutoModelForCausalLM
import torch, time

class TreeNode:
    def __init__(self):
        self.children = {}
        self.key_value = None  # Stores key_value for this token (n_layers, 2, num_heads, embed_size_per_head)
        self.logits = None  # Stores logits for this token (vocab_size)


class LLMDict:
    def __init__(self):
        self.d = {}

    def load_key_values(self, batch_tokens):
        pass

    def store_key_values(self, batch_tokens, key_values):
        pass


class MyKeyValuesCache:
    def __init__(self, kv):
        # (n_layers, 2, batch_size, num_heads, sequence_length, embed_size_per_head)
        self.kv = kv

    def get(self, batch_num, token_num):
        return [[layer[idx][batch_num, :, token_num] for idx in range(2)] for layer in self.kv]

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

        return key_values_list

    def store_key_values(self, tokens, kv: MyKeyValuesCache, batch_num, tokens_logits=None):
        # print("tokens len", len(tokens))
        # if tokens_logits is not None:
        #     print("tokens_logits len", len(tokens_logits))
        # print("tokens", tokens)
        # key_values shape: (sequence_length, n_layers, 2, num_heads, embed_size_per_head)
        # tokens_logits shape: (sequence_length, vocab_size)
        node = self.root
        # sequence_length = key_values.shape[0]
        # assert sequence_length == len(tokens), f"Tokens length and key_values sequence length must match. But sequence_length is {sequence_length}, tokens len is {len(tokens)}"

        for i, token in enumerate(tokens):
            if token not in node.children:
                node.children[token] = TreeNode()
            node = node.children[token]
            node.key_value = kv.get(batch_num, i)
            if tokens_logits is not None:
                pad_len = len(tokens) - len(tokens_logits)
                if i >= pad_len:
                    node.logits = tokens_logits[i - pad_len]

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

# past_key_values: (n_layers, 2, batch_size, num_heads, sequence_length, embed_size_per_head) => batch_kv: (batch_size, sequence_length, n_layers, 2, num_heads, embed_size_per_head) 
def permute_to_batch_format(past_key_values):
    batch_size, num_heads, sequence_length, embed_size_per_head = past_key_values[0][0].shape
    results = []
    for layer_i in range(len(past_key_values)):
        for idx in range(len(past_key_values[layer_i])):
            results.append(past_key_values[layer_i][idx].to('cpu'))
    results = torch.cat(results).reshape(len(past_key_values), 2, batch_size, num_heads, sequence_length, embed_size_per_head)
    results = results.permute(2, 4, 0, 1, 3, 5).to('cuda')
    return results

# batch_kv: (batch_size, sequence_length, n_layers, 2, num_heads, embed_size_per_head) => past_key_values: (n_layers, 2, batch_size, num_heads, sequence_length, embed_size_per_head) 
def permute_to_nlayer_format(batch_kv):
    res = []
    nlayer = len(batch_kv[0][0])
    ntoken = len(batch_kv[0])
    nbatch = len(batch_kv)
    for layer_num in range(nlayer):
        res.append([])
        for idx in [0, 1]:
            res[layer_num].append([])
            for batch_num in range(nbatch):
                res[layer_num][idx].append([])
                for token_num in range(ntoken):
                    # print(len(batch_kv[batch_num][token_num][layer_num]), idx)
                    res[layer_num][idx][batch_num].append(batch_kv[batch_num][token_num][layer_num][idx])
                res[layer_num][idx][batch_num] = torch.stack(res[layer_num][idx][batch_num], dim=1)
            res[layer_num][idx] = torch.stack(res[layer_num][idx])
    
    for layer_num in range(nlayer):
        print(res[layer_num][0].shape, res[layer_num][1].shape)
    return res

def llm_tree_accelerate_last_logit(batch_tokens, tree: LLMKeyValueTree, causal_llm):
    print(batch_tokens)
    batch_tokens = [torch.tensor(tokens) for tokens in batch_tokens]
    assert all(len(lst) == len(batch_tokens[0]) for lst in batch_tokens), "All tokens should have the same length"
    print(f"Tokens length {len(batch_tokens[0])}")
    if True:
        min_cache_len = 100000000
        prefix_batch_kv = []
        for tokens in batch_tokens:
            kv = tree.load_key_values(tokens.tolist())
            if kv is None:
                min_cache_len = 0
                break
            min_cache_len = min(len(kv), min_cache_len)
            prefix_batch_kv.append(kv)
        print("min_cache_len", min_cache_len)
        if min_cache_len > 0:
            prefix_batch_kv = [kv[:min_cache_len] for kv in prefix_batch_kv] # (batch_size, sequence_length, n_layers, 2, (num_heads, embed_size_per_head))
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
    batch_tokens = [torch.tensor(tokens).to(causal_llm.device) for tokens in batch_tokens]
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
            min_cache_len = min(len(kv), min_cache_len)
            prefix_batch_kv.append(kv)
        # print("min_cache_len", min_cache_len)
        causal_start = time.time()
        if min_cache_len > 0:
            prefix_batch_kv = [kv[:min_cache_len] for kv in prefix_batch_kv] # (batch_size, sequence_length, n_layers, 2, (num_heads, embed_size_per_head))
            suffix_tokens = torch.stack([tokens[min_cache_len:] for tokens in batch_tokens])
            past_kv = permute_to_nlayer_format(prefix_batch_kv)
            outputs = causal_llm(input_ids=suffix_tokens, past_key_values=past_kv, use_cache=True)
            causal_end = time.time()
        else:
            outputs = causal_llm(input_ids=torch.stack(batch_tokens), use_cache=True)
        causal_end = time.time()
        permute_start = time.time()
        my_kv = MyKeyValuesCache(outputs.past_key_values)
        permute_end = time.time()
        store_start = time.time()
        for batch_num in range(len(batch_tokens)):
            tree.store_key_values(batch_tokens[batch_num].tolist(), my_kv, batch_num, outputs.logits[batch_num])
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

if __name__ == '__main__':
    import random
    causal_model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    # causal_model_name = 'gpt2'
    causal_llm_tokenizer = AutoTokenizer.from_pretrained(causal_model_name)
    causal_llm = AutoModelForCausalLM.from_pretrained(causal_model_name).to('cuda')
    # test_tokens = causal_llm_tokenizer.encode("Hello, my name is Collin Zhang, how can I help you?")
    test_tokens = [[random.randint(0, 50257) for _ in range(100)] for _ in range(5)]
    # t1 = time.time()
    # causal_llm(input_ids=torch.tensor(test_tokens[:1]).to('cuda'))
    # t2 = time.time()
    # causal_llm(input_ids=torch.tensor(test_tokens[:1]).to('cuda'))
    # t3 = time.time()
    # causal_llm(input_ids=torch.tensor(test_tokens).to('cuda'))
    # t4 = time.time()
    # print("Time taken for 5 tokens", t2 - t1)
    # print("Time taken for 1 tokens", t3 - t2)
    # print(t4 - t3)
    # exit()

    # print("test_tokens", test_tokens)
    tree = LLMKeyValueTree()
    output = llm_tree_accelerate_logits([tokens[:-1] for tokens in test_tokens], tree, causal_llm)
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        # on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'),  # Save logs for TensorBoard visualization
        record_shapes=True,  # Records the input shapes
        with_stack=True,  # Records the stack trace
        profile_memory=True,  # Profiles memory usage
    ) as prof:
        output = llm_tree_accelerate_logits(test_tokens, tree, causal_llm)
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    prof.export_chrome_trace("trace.json")  # Save trace for Chrome tracing