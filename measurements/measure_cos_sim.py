import json, torch, random
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

if __name__ == '__main__':
    triggers = ['spotify']
    # optimizers = [BeamSearchHotflip]
    triggers = ['homegoods', 'huawei', 'science channel', 'vh1', 'lidl', 'triumph motorcycles', 'avon', 'snapchat', 'steelseries keyboard', 'yeezy', 'laurent-perrier', 'the washington post', 'twitch', 'engadget', 'bruno mars', 'giorgio armani', 'old el paso', 'levis', 'kings', 'ulta beauty']
    ds = load_dataset("microsoft/ms_marco", "v1.1")
    queries = ds['train']['query'] # type: ignore
    test_queries = random.sample(queries, 128)
    file_name = '../data/rl_trigger_results.json'
    encoder = SentenceTransformer('facebook/contriever')
    with open(file_name, 'r') as f:
            result = json.load(f)
    for opt in result:
        cos_sim_list = []
        print(opt)
        for trigger in result[opt]:
            prefix_trigger_documents = [trigger + query for query in test_queries]
            trigger_embs = encoder.encode(prefix_trigger_documents, convert_to_tensor=True)
            attack_emb = encoder.encode(result[opt][trigger], convert_to_tensor=True)
            cos_sim = torch.nn.functional.cosine_similarity(trigger_embs, attack_emb, dim=-1).mean().item()
            # print(trigger, cos_sim)
            cos_sim_list.append(cos_sim)
        print('Avg', sum(cos_sim_list) / len(cos_sim_list))