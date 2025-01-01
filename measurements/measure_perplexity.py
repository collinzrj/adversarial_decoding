from transformers import AutoTokenizer, AutoModelForCausalLM
import torch, random, json
from datasets import load_dataset
from tqdm import tqdm

torch.set_default_device('cuda')

class PerplexityMeasurer:

    def __init__(self):
        device = torch.get_default_device()
        # model_name = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
        model_name = 'gpt2'
        self.causal_llm_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.causal_llm_tokenizer.pad_token = self.causal_llm_tokenizer.eos_token
        self.causal_llm = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    def compute_perplexity(self, texts):
        inputs = self.causal_llm_tokenizer.batch_encode_plus(texts, pad_to_max_length=True)
        attention_mask = torch.tensor(inputs['attention_mask'])
        inputs = torch.tensor(inputs['input_ids'])
        labels = inputs
        # input_ids = torch.tensor([seq]).to(device)
        lm_logits = self.causal_llm(input_ids=inputs, attention_mask=attention_mask).logits
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_masks = attention_mask[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        loss = loss.view(shift_labels.shape[0], -1) * shift_masks
        loss = torch.sum(loss, -1) / torch.sum(shift_masks, -1)
        return torch.exp(loss).tolist()


def measure_doc_perplexities():
    ds = load_dataset("microsoft/ms_marco", "v2.1")
    texts = []
    for passages in tqdm(ds['train']['passages']):
        for text in passages['passage_text']:
            texts.append(text)
    random_queries = random.sample(texts, 1000)
    measurer = PerplexityMeasurer()
    # print(random_queries)
    perplexities = []
    for i in tqdm(range(0, len(random_queries), 100)):
        perplexities += measurer.compute_perplexity(random_queries[i:i+100])
    results = list(zip(random_queries, perplexities))
    with open('../data/doc_perplexities.json', 'w') as f:
        json.dump(results, f, indent=2)


def measure_adv_perplexities():
    measurer = PerplexityMeasurer()
    opt_trig_perp = {}
    with open('../data/rl_trigger_results.json') as f:
        res = json.load(f)
        for opt in res:
            opt_trig_perp[opt] = {}
            for trig in res[opt]:
                print(opt, trig)
                adv = res[opt][trig]
                perp = measurer.compute_perplexity([adv])[0]
                opt_trig_perp[opt][trig] = perp
    # with open('./data/optimizer_trigger_res_perplexities.json', 'w') as f:
    #     json.dump(opt_trig_perp, f, indent=2)
    print(json.dumps(opt_trig_perp, indent=2))
    return opt_trig_perp


if __name__ == '__main__':
    measure_doc_perplexities()
    from seaborn import violinplot
    import matplotlib.pyplot as plt
    import numpy as np
    opt_trig_perp = measure_adv_perplexities()
    data = []
    with open('../data/doc_perplexities.json', 'r') as f:
        results = json.load(f)
        data.append(np.log([p[1] for p in results]))
    for k in ['BasicAdversarialDecoding', 'AdversarialDecoding', 'RL_better']:
        v = opt_trig_perp[k]
        data.append(np.log(list(v.values())))
    ax = violinplot(data=data)
    ax.set_xticklabels(['real', 'BasicAdv', 'AdvDec', 'RL'])
    ax.set_ylabel('Perplexity (log scale)')
    plt.savefig('violin.pdf')
    plt.savefig('violin.png')
        