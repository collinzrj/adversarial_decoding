import json
from openai import OpenAI
from nltk.translate.bleu_score import sentence_bleu
from tqdm import tqdm

client = OpenAI()

path = '/share/shmatikov/collin/adversarial_decoding/data/emb_inv_attack_unnatural_gte-Qwen_20250312_192926.json'

with open(path, 'r') as f:
    data = json.load(f)

train_data = data[:10]
test_data = data[10:]

start_prompt = 'Try reconstruct good text from bad text, here are some examples:'
for item in train_data:
    start_prompt += f'\nBad text: {item["generation"].strip()}\nGood text: {item["target"].strip()}\n'

results = []
for item in tqdm(test_data):
    prompt = start_prompt
    prompt += '\nNow try to reconstruct the following text:'
    prompt += f'\nBad text: {item["generation"].strip()}\nGood text: '
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    new_item = item.copy()
    new_item['reconstruction'] = completion.choices[0].message.content
    bleu_score = sentence_bleu([new_item['target'].split(' ')], new_item['reconstruction'].split(' '))
    new_item['reconstruction_bleu_score'] = bleu_score
    results.append(new_item)

with open('results.json', 'w') as f:
    json.dump(results, f, indent=4)
