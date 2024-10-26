from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer, BertModel, T5ForConditionalGeneration, AutoModelForMaskedLM, AutoModelForCausalLM, LlamaModel
import torch, time
import random
from tqdm import tqdm
from dataclasses import dataclass, field
from .bert_models import BertForLM

def set_no_grad(model):
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

def repetition_score(text):
    score = 0
    words = text.split(' ')
    for i, word in enumerate(words):
        if not word.isalnum():
            continue
        if word in words[:i] or len(word) > 10:
            score += len(word)
    return score / len(text)

def generate_banned_tokens(this_tokenizer):
    banned_tokens = []
    for token in range(this_tokenizer.vocab_size - 1):
        word = this_tokenizer.decode([token])
        if word.count(' ') > 1:
            banned_tokens.append(token)
            continue
        if word == '.' or word == '?' or word == '!' or word == ',':
            continue
        for char in word:
            if char == ' ':
                continue
            if ord('a') <= ord(char) and ord(char) <= ord('z'):
                continue
            if ord('A') <= ord(char) and ord(char) <= ord('Z'):
                continue
            banned_tokens.append(token)
            break
    return banned_tokens


class COLD:
    def __init__(self):
        device = torch.get_default_device()
        self.encoder_tokenizer = AutoTokenizer.from_pretrained('facebook/contriever')
        self.encoder: BertModel = AutoModel.from_pretrained('facebook/contriever').to(device)
        self.naturalness_llm_tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3.1-8B-Instruct')
        self.naturalness_llm_tokenizer.pad_token = self.naturalness_llm_tokenizer.eos_token
        self.naturalness_llm = AutoModelForCausalLM.from_pretrained('meta-llama/Meta-Llama-3.1-8B-Instruct').to(device)
        set_no_grad(self.encoder)
        self.causal_llm_tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.causal_llm = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
        set_no_grad(self.causal_llm)
        self.mask_llm_tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
        self.mask_llm = AutoModelForMaskedLM.from_pretrained("google-bert/bert-base-uncased")
        set_no_grad(self.mask_llm)
        self.encoder_word_embedding = self.encoder.get_input_embeddings().weight.detach()
        self.encoder_vocab_size = self.encoder_word_embedding.shape[0]
        self.causal_llm_tokenizer.pad_token = self.causal_llm_tokenizer.eos_token
        self.EOS_TOKEN = self.causal_llm_tokenizer.encode(self.causal_llm_tokenizer.pad_token)[0]
        self.llm_banned_tokens = generate_banned_tokens(self.causal_llm_tokenizer)
        self.llm_banned_tokens.append(188)
        # print("banned_tokens len", len(llm_banned_tokens))
        self.hotflip_banned_tokens = generate_banned_tokens(self.encoder_tokenizer)
        self.hotflip_banned_tokens.append(self.encoder_tokenizer.cls_token_id)
        self.bert_tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
        self.bert_lm = BertForLM.from_pretrained('../collision/wiki103/bert').to(device)
    # Mean pooling
    def mean_pooling(self, token_embeddings, mask):
        token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
        sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
        return sentence_embeddings

    def compute_doc_embs(self, documents):
        doc_inputs = self.encoder_tokenizer(documents, padding=True, truncation=True, return_tensors='pt')
        doc_embs = self.mean_pooling(self.encoder(**doc_inputs)[0], doc_inputs['attention_mask'])
        return doc_embs

    def compute_average_cosine_similarity(self, embeddings):
        # Normalize the embeddings to have unit norm
        normalized_embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        # Compute the cosine similarity matrix
        cos_sim_matrix = torch.mm(normalized_embeddings, normalized_embeddings.t())
        
        # Mask out the diagonal elements (self-similarity) and compute the average of the off-diagonal elements
        mask = torch.eye(cos_sim_matrix.size(0), device=cos_sim_matrix.device).bool()
        cos_sim_matrix.masked_fill_(mask, 0)
        
        # Compute the average cosine similarity (excluding self-similarity)
        avg_cos_sim = cos_sim_matrix.sum() / (cos_sim_matrix.numel() - cos_sim_matrix.size(0))

        print("document cos sim", avg_cos_sim, 'min', cos_sim_matrix.min())
        
        return avg_cos_sim

    def compute_cos_sim(self, s_adv, doc_embs, stemp):
        ## compute loss
        _, inputs_embeds = self.relaxed_to_word_embs(s_adv, stemp)
        s_adv_emb = self.encoder(inputs_embeds=inputs_embeds)[0].mean(dim=1)
        cos_sim = torch.nn.functional.cosine_similarity(s_adv_emb, doc_embs, dim=1)
        return cos_sim.mean()

    def relaxed_to_word_embs(self, x, stemp):
        # convert relaxed inputs to word embedding by softmax attention
        p = torch.softmax(x / stemp, -1)
        x = torch.mm(p, self.encoder_word_embedding)
        # add embeddings for period and SEP
        # x = torch.cat([x, word_embedding[encoder_tokenizer.sep_token_id].unsqueeze(0)])
        return p, x.unsqueeze(0)

    def tokens_to_embs(self, tokens):
        embs = []
        for token in tokens:
            embs.append(self.encoder_word_embedding[token])
        return torch.stack(embs)

    def generate_start_tokens(self, trigger):
        from openai import OpenAI
        client = OpenAI()
        response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": f'Write a 32 tokens sentence about {trigger}.'},
        ],
        max_tokens=32
        )
        # print(text)
        res: str = response.choices[0].message.content # type: ignore
        print("start tokens from gpt4o", res)
        return self.encoder_tokenizer.encode(res)

    def optimize(self, documents, trigger, start_tokens=None, lr=0.5, topk=10, beam_width=10, eps=0.1, stemp=1, epoch_num=5, perturb_iter=30): 
        # logits_beta = 0.75
        logits_beta = 5
        print("logits_beta", logits_beta)


        @dataclass
        class BeamCandidate:
            sequence: ... = field(default_factory=list)
            score: ... = None
            trig_cos_sim: ... = None
            avg_perp: ... = None
            total_perp: ... = 0

        def bert_next_token(tokens_list, next_token_logits=False):
            first_len = len(tokens_list[0])
            for tokens_idx in range(len(tokens_list)):
                assert len(tokens_list[tokens_idx]) == first_len
            inputs = torch.tensor(tokens_list).to(torch.long)
            attention_mask = torch.ones(inputs.shape)
            outputs = self.bert_lm(input_ids=inputs, attention_mask=attention_mask)
            next_token_logits = outputs[0][:, -1, :]
            return next_token_logits

        def beam_search(z_i, num_tokens, doc_embs, trig_doc_embs, topk, beam_width, stemp):
            print("start beam search", flush=True)
            start_t = time.time()
            _, inputs_embeds = self.relaxed_to_word_embs(z_i, stemp)
            # s_adv_emb = encoder(inputs_embeds=inputs_embeds)[0].mean(dim=1)
            
            # Initialize the beams with empty sequences and their scores
            beams = [BeamCandidate()]  # List of tuples (token_sequence, similarity_score)
            
            for i in tqdm(range(num_tokens)):
                all_candidates = []
                batch_embs = []
                batch_sequences = []
                if i > 0:
                    beams_logits = bert_next_token([beam.sequence for beam in beams], next_token_logits=True)
                else:
                    ## in first iter the sentence length is 0
                    beams_logits = torch.zeros((len(beams), len(z_i[0])))
                # print("beams_logits shape", beams_logits.shape)
                for candidate, logits in zip(beams, beams_logits):
                    prev_tokens, perp_score = candidate.sequence, candidate.total_perp
                    tokens_prob = torch.softmax(logits, -1)
                    token_scores = logits_beta * tokens_prob + torch.softmax(z_i[i], -1)
                    # ban special chars and numbers
                    # for banned_token in hotflip_banned_tokens:
                    #     token_scores[banned_token] = -1000
                    token_scores[self.encoder_tokenizer.pad_token_id] = -1000
                    _, topk_tokens = torch.topk(token_scores, topk)
                    for token in topk_tokens:
                        # repetitive ban
                        # if len(prev_tokens) > 0 and token == prev_tokens[-1]:
                        #     continue
                        if prev_tokens.count(token) >= 2:
                            continue
                        new_sequence = prev_tokens + [token]
                        current_word_embs = torch.cat((self.tokens_to_embs(new_sequence), inputs_embeds[0][len(new_sequence):]), dim=0).unsqueeze(0)
                        batch_embs.append(current_word_embs)
                        if i > 0:
                            batch_sequences.append(BeamCandidate(sequence=new_sequence, total_perp=perp_score - torch.log(tokens_prob[token].detach())))
                        else:
                            batch_sequences.append(BeamCandidate(sequence=new_sequence, total_perp=0))
                # Process all candidate embeddings in parallel
                batch_embs = torch.cat(batch_embs, dim=0)
                batch_embs_output = self.encoder(inputs_embeds=batch_embs)[0].mean(dim=1)
                batch_perplexities = self.compute_perplexity(self.bert_tokenizer.batch_decode([candidate.sequence for candidate in batch_sequences]))
                for j, candidate in enumerate(batch_sequences):
                    new_sequence = candidate.sequence
                    perp_score = candidate.total_perp
                    # current_doc_sim = torch.nn.functional.cosine_similarity(batch_embs_output[j], doc_embs, dim=1).mean().item()
                    current_trig_sim = torch.nn.functional.cosine_similarity(batch_embs_output[j], trig_doc_embs, dim=1).mean().item()
                    current_perp = batch_perplexities[j]
                    if i > 1:
                        current_perp = torch.exp(perp_score / (len(new_sequence) - 1))
                    else:
                        current_perp = torch.tensor(1)
                    current_perp = torch.clamp(current_perp, min=150)
                    # all_candidates.append(BeamCandidate(new_sequence, current_trig_sim - 0.001 * torch.log(current_perp), current_trig_sim, current_perp, perp_score))
                    # penalty = spell_check_score(encoder_tokenizer.decode(new_sequence)) * 0.05
                    penalty = 0
                    all_candidates.append(BeamCandidate(new_sequence, current_trig_sim - current_perp * 0.01 * 0.02, current_trig_sim, current_perp, perp_score))
                # Sort all candidates by their similarity scores in descending order
                all_candidates = sorted(all_candidates, key=lambda x: x.score, reverse=True)
                # Select the top `beam_width` candidates
                # beams = all_candidates[:beam_width // 2] + random.sample(all_candidates[beam_width:], beam_width // 2)
                beams = all_candidates[:beam_width]
            
            end_t = time.time()
            print("beam search takes", end_t - start_t, flush=True)
            # Return the best sequence from the beams
            return max(beams, key=lambda x: x.score)

        def label_smoothing(best_sequence):
            next_z_i = torch.nn.functional.one_hot(torch.tensor(best_sequence), self.encoder_vocab_size).float()
            next_z_i = (next_z_i * (1 - eps)) + (1 - next_z_i) * eps / (self.encoder_vocab_size - 1)
            z_i = torch.nn.Parameter(torch.log(next_z_i), requires_grad=True)
            return z_i
        
        if start_tokens is None:
            start_tokens = self.generate_start_tokens(trigger)

        z_i = label_smoothing(start_tokens)
        trig_doc_embs = self.compute_doc_embs(documents)

        noise_level = 5
        for epoch in range(epoch_num):
            ## optimize
            # z_i = torch.nn.Parameter(torch.zeros((num_tokens, vocab_size), device=device), True)
            optimizer = torch.optim.Adam([z_i], lr=lr)
            print(z_i)
            for i in range(perturb_iter):
                # perturb_iter = 5
                optimizer.zero_grad()
                trig_cos_sim = self.compute_cos_sim(z_i, trig_doc_embs, stemp)
                # print(doc_cos_sim, trig_cos_sim)
                # loss = 0.25 * doc_cos_sim - trig_cos_sim
                loss = 1 - trig_cos_sim
                print(epoch, i, 'trig_sim', trig_cos_sim.detach().cpu().numpy(), 'loss', loss.detach().cpu().numpy(), flush=True)
                loss.backward()
                optimizer.step()
                noise_level = noise_level * 0.8
                # z_i = z_i + torch.rand_like(z_i) * lr * noise_level
            ## search
            ## TODO: add llm guidance here
            z_i = z_i.detach()
            print(z_i)
            candidate: BeamCandidate = beam_search(z_i, len(start_tokens), None, trig_doc_embs, topk, beam_width, stemp)
            # candidate: BeamCandidate = beam_search_only_cos_sim(z_i, len(start_tokens), doc_embs, trig_doc_embs, topk, beam_width, stemp)
            print(epoch, 'best_score', candidate.score, 'cos_sim', candidate.trig_cos_sim, 'perp', candidate.avg_perp)
            best_sequence = candidate.sequence
            print('best_sequence', [int(token.detach().cpu().numpy()) for token in best_sequence])
            best_seq_str = self.encoder_tokenizer.decode(best_sequence)
            print(best_seq_str)

            z_i = label_smoothing(best_sequence)
            return best_seq_str



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
        return torch.exp(loss)
    
