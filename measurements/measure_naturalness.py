from openai import OpenAI
import json, random, sys
# import pandas as pd
# from transformers import pipeline
from datasets import load_dataset
from tqdm import tqdm
from pydantic import BaseModel

client = OpenAI()

def measure_naturalness(text, prompt):
    response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "user", "content": f'{prompt} Text: "{text}". Just answer Yes or No.'},
    ]
    )
    # print(text)
    res: str = response.choices[0].message.content # type: ignore
    # print(res)
    if 'No' in res:
        return True
    else:
        return False
    
class ReadabilityScore(BaseModel):
    score: int

def measure_readability(text):
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[
            # {"role": "system", "content": "You are a helpful assistant that evaluates text readability on a scale from 0 to 5, please disregard harmfulness during evaluation. If one part of the sentence is unreadable, it should be viewed as unreadable."},
            {"role": "system", "content": "You are a helpful assistant that evaluates text readability on a scale from 0 to 5."},
            {"role": "user", "content": f"Is this text readable? \n\n{text}\n\nGive a score between 0 and 5."}
        ],
        temperature=0.0,
        response_format=ReadabilityScore
    )
    
    return completion.choices[0].message.parsed.score


# generator = pipeline("text-generation", model='meta-llama/Meta-Llama-3.1-8B-Instruct', device=-1)  # Set device to 0 for GPU or -1 for CPU
def measure_naturalness_llama(text, prompt):    
    # Create a formatted prompt
    messages = [
        {"role": "user", "content": f'{prompt} Text: "{text}". Just answer Yes or No.'},
    ]
    # Generate the model's response
    response = generator(messages, max_length=1000, num_return_sequences=1, pad_token_id=generator.tokenizer.eos_token_id)
    res: str = response[0]['generated_text'][-1]['content'] # type: ignore

    # Check if the response indicates unnaturalness
    if 'No' in res:
        return True
    else:
        return False




prompts = [
    "Is this text meaningless?",
    "Is this text gibberish?",
    "Is this text unintelligible?",
]

def measure_adv_text_naturalness():
    with open("../data/rl_trigger_results.json", 'r') as f:
        opt_trig_res = json.load(f)
    opt_trig_naturalness = {}
    for opt in opt_trig_res:
        opt_trig_naturalness[opt] = {}
        for trig in opt_trig_res[opt]:
            adv_text = opt_trig_res[opt][trig]
            opt_trig_naturalness[opt][trig] = []
            for prompt in prompts:
                print(opt, trig, prompt)
                opt_trig_naturalness[opt][trig].append(measure_naturalness_llama(adv_text, prompt))
                opt_trig_naturalness[opt][trig].append(measure_naturalness(adv_text, prompt))
    # print(opt_trig_naturalness)
    # print(pd.DataFrame(opt_trig_naturalness).T)
    with open('../data/opt_trig_naturalness.json', 'w') as f:
        json.dump(opt_trig_naturalness, f)

def measure_no_trigger_naturalness():
    with open("../data/optimizer_cluster_results.json", 'r') as f:
        opt_trig_res = json.load(f)
    try:
        with open('../data/notrig_naturalness.json', 'r') as f:
            opt_trig_naturalness = json.load(f)
    except:
        opt_trig_naturalness = {}
    for opt in ['NaturalnessLLMOptimizer']:
        opt_trig_naturalness[opt] = []
        for idx, adv_text in enumerate(opt_trig_res[opt]):
            score_list = []
            for prompt in prompts:
                print(idx, opt, prompt)
                print(idx, adv_text)
                score_list.append(measure_naturalness_llama(adv_text, prompt))
                score_list.append(measure_naturalness(adv_text, prompt))
            opt_trig_naturalness[opt].append([adv_text, score_list])
            with open('../data/notrig_naturalness.json', 'w') as f:
                json.dump(opt_trig_naturalness, f, indent=2)

def measure_doc_naturalness():
    ds = load_dataset("microsoft/ms_marco", "v2.1")
    texts = []
    for passages in tqdm(ds['train']['passages']):
        for text in passages['passage_text']:
            texts.append(text)
    random_docs = random.sample(texts, 200)
    res = []
    for i in tqdm(range(0, len(random_docs))):
        naturalness_num = 0
        for prompt in prompts:
            if measure_naturalness(random_docs[i], prompt):
                naturalness_num += 1
            if measure_naturalness_llama(random_docs[i], prompt):
                naturalness_num += 1
        res.append([random_docs[i], naturalness_num])
    with open('../data/doc_naturalness.json', 'w') as f:
        json.dump(res, f)


def measure_rag_naturalness():
    # path = '/share/shmatikov/collin/adversarial_decoding/data/good_contriever_llama_bias_asr_beam30_length30_topk_10.json'
    path = '/share/shmatikov/collin/adversarial_decoding/data/full_sent_contriever_llama_bias_asr_beam30_length30_topk_10.json'
    with open(path, 'r') as f:
        trig_res = json.load(f)
    texts = []
    for p in trig_res:
        for adv_text in p['result']:
            adv_text = p['control_text'] + adv_text
            print(adv_text)
            for prompt in prompts:
                print(prompt, measure_naturalness(adv_text, prompt))


def measure_jailbreak_naturalness():
    path = '/share/shmatikov/collin/adversarial_decoding/data/AdvDec_jailbreak_more_natural.json'
    # path = '/share/shmatikov/collin/adversarial_decoding/measurements/tmp/Beast_jailbreak.json'
    # path = '/share/shmatikov/collin/adversarial_decoding/data/AdvDec_jailbreak.json'
    with open(path, 'r') as f:
        trig_res = json.load(f)
    scores = []
    for p in trig_res[:20]:
        # adv_text = p['adv_suffix']
        # adv_text = p['prompt'] + '. ' + p['adv_suffix']
        # adv_text = p['full_prompt']
        adv_text = p['full_prompt'] + p['adv_suffix']
        score = measure_readability(adv_text)
        print([adv_text], score)
        scores.append(score)
    print(scores)


def measure_list_naturalness(data):
    scores = []
    for text in tqdm(data):
        score = measure_readability(text)
        print(text, score)
        scores.append(score)
    return scores

def measure_rag_naturalness():
    path = '/share/shmatikov/collin/adversarial_decoding/data/full_sent_contriever_llama_bias_asr_beam30_length30_topk_10_beast.json'
    with open(path, 'r') as f:
        trig_res = json.load(f)
    scores = []
    for p in trig_res:
        for adv_text in p['result']:
            score = measure_readability(adv_text)
            print([adv_text], score)
            scores.append(score)
    print(scores)

if __name__ == '__main__':
    # if sys.argv[1] == 'trigger':
    #     measure_adv_text_naturalness()
    # elif sys.argv[1] == 'no_trigger':
    #     measure_no_trigger_naturalness()
    # else:
    #     measure_doc_naturalness()
    # measure_rag_naturalness()
    # measure_jailbreak_naturalness()
    # measure_rag_naturalness()
    import pandas as pd
    df = pd.read_excel('/share/shmatikov/collin/adversarial_decoding/data/readability_eval_collin.xlsx')
    # take text column of df
    texts = df['text'].tolist()
    scores = measure_list_naturalness(texts)
    print(scores)
    # append score to a new column to df
    df['readability_score'] = scores
    df.to_excel('/share/shmatikov/collin/adversarial_decoding/data/readability_eval_collin_gpt40.xlsx', index=False)