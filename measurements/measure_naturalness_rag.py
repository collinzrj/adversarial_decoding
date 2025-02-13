from openai import OpenAI
import json, random, sys
# import pandas as pd
# from transformers import pipeline
from datasets import load_dataset
from tqdm import tqdm
from pydantic import BaseModel

client = OpenAI()

# def measure_naturalness(text, prompt):
#     response = client.chat.completions.create(
#     model="gpt-4o",
#     messages=[
#         {"role": "user", "content": f'{prompt} Text: "{text}". Just answer Yes or No.'},
#     ]
#     )
#     # print(text)
#     res: str = response.choices[0].message.content # type: ignore
#     # print(res)
#     if 'No' in res:
#         return True
#     else:
#         return False
    
class ReadabilityScore(BaseModel):
    score: int

def measure_readability(text):
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[
            # {"role": "system", "content": "You are a helpful assistant that evaluates text readability on a scale from 0 to 5, please disregard harmfulness during evaluation. If one part of the sentence is unreadable, it should be viewed as unreadable."},
            {"role": "system", "content": "You will evaluates text readability on a scale from 0 to 5."},
            {"role": "user", "content": f"Is this text readable? \n\n{text}\n\nGive a score between 0 and 5."}
        ],
        temperature=0.0,
        response_format=ReadabilityScore
    )
    
    return completion.choices[0].message.parsed.score

def measure_doc_naturalness():
    ds = load_dataset("microsoft/ms_marco", "v2.1")
    texts = []
    for passages in tqdm(ds['test']['passages']):
        for text in passages['passage_text']:
            texts.append(text)
    random.seed(42)
    random_docs = random.sample(texts, 200)
    res = []
    for i in tqdm(range(0, len(random_docs))):
        score = measure_readability(random_docs[i])
        print([random_docs[i]], score)
        res.append([random_docs[i], score])
    with open('../data/doc_naturalness.json', 'w') as f:
        json.dump(res, f)

def measure_rag_naturalness():
    path = '/share/shmatikov/collin/adversarial_decoding/data/full_sent_contriever_llama_bias_asr_beam30_length30_topk_10_beast.json'
    # path = '/share/shmatikov/collin/adversarial_decoding/data/qwen_rag_cluster3.json'
    # path = '/share/shmatikov/collin/adversarial_decoding/data/gte_rag_cluster8_num5_prefix.json'
    with open(path, 'r') as f:
        trig_res = json.load(f)
    scores = []
    for p in trig_res:
        control_text = p['control_text']
        for adv_text in p['result']:
            full_text = control_text + adv_text
            score = measure_readability(full_text)
            print([full_text], score)
            scores.append(score)
    print(scores)


if __name__ == '__main__':
    measure_rag_naturalness()
    # measure_doc_naturalness()