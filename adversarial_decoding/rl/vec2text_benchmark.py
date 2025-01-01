import vec2text, torch, random
from transformers import AutoTokenizer, AutoModel, T5ForConditionalGeneration
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

inversion_model = vec2text.models.InversionModel.from_pretrained("jxm/gtr__nq__32")
corrector_model = vec2text.models.CorrectorEncoderModel.from_pretrained("jxm/gtr__nq__32__correct")
corrector = vec2text.load_corrector(inversion_model, corrector_model)

tokenizer = AutoTokenizer.from_pretrained("t5-base")
# inputs = tokenizer("hello world", return_tensors="pt")
# embs: torch.Tensor = inversion_model.call_embedding_model(**inputs)

encoder = SentenceTransformer('sentence-transformers/gtr-t5-base')

ds = load_dataset("microsoft/ms_marco", "v1.1")
queries = ds['train']['query']
random_queries = random.sample(queries, 128)
trigger = 'homegoods'
target_queries = [trigger + query for query in random_queries]

# embeddings = encoder.encode(target_queries, convert_to_tensor=True, normalize_embeddings=True).to('cpu')
# print(embeddings)

def get_embeddings(texts):
    inputs = tokenizer.batch_encode_plus(texts, return_tensors="pt", padding=True, truncation=True).to('cuda')
    embeddings = inversion_model.call_embedding_model(**inputs)
    return embeddings

embeddings = get_embeddings(target_queries)
emb = embeddings.mean(dim=0)
embs = emb.unsqueeze(0)

# logits = inversion_model.forward(None, None, frozen_embeddings=embs, decoder_input_ids=torch.tensor([[0]])).logits
# output = logits[0].argmax(dim=-1)
# print(output)


tokens = inversion_model.generate(
    inputs={
    'frozen_embeddings': embs
    },
    generation_kwargs={})
print(tokens)
adv = tokenizer.batch_decode(tokens)[0]
print(adv)

adv2 = vec2text.invert_embeddings(embs, corrector, num_steps=40, sequence_beam_width=20)[0]
print(adv2)



# doc_embeds = encoder.encode([trigger + query for query in random_queries], convert_to_tensor=True)
doc_embeds = get_embeddings([trigger + query for query in random_queries])

for adv_text in [adv, adv2,
                 "what is homegood, what are these things that they are called when they are"]:
    print("")
    print([adv_text])
    # adv_emb = encoder.encode(adv_text, convert_to_tensor=True)
    adv_emb = get_embeddings([adv_text])
    print(torch.nn.functional.cosine_similarity(doc_embeds, adv_emb, dim=-1).mean().item())

# adv_text = "I was listening to Spotify on my phone when a friend recommended a new playlist on Spotify, so I opened the Spotify app, searched for it on Spotify, and saved it to my Spotify library."
# adv_emb = inversion_model.call_embedding_model(**tokenizer(adv_text, return_tensors="pt"))
# # print(adv_emb)
# print(torch.nn.functional.cosine_similarity(embeddings, adv_emb, dim=-1).mean().item())

# adv_text = " HomeGoods is a popular chain of stores specializing in home decor, accessories, textiles"
# adv_emb = inversion_model.call_embedding_model(**tokenizer(adv_text, return_tensors="pt"))
# # print(adv_emb)
# print(torch.nn.functional.cosine_similarity(embeddings, adv_emb, dim=-1).mean().item())
