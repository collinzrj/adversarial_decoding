import torch, json, vec2text
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from transformers.cache_utils import DynamicCache
from datasets import load_dataset
import random
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import wandb  # Import wandb for logging
import copy   # Import copy to create a deepcopy of the actor model
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu
from peft import LoraConfig, get_peft_model
import os
import glob

# Set device
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# torch.set_default_device(device)
device = 'cuda:0'
trigger_dict_path = 'gtr_trigger_dict.pt'

def compute_repetition_score(tokens, n=1):
    ngram_set = set()
    cnt = 0
    rep_scores = []
    for i in range(0, len(tokens) - n + 1):
        current = ','.join([str(t) for t in tokens[i:i+n]])
        if current in ngram_set:
            cnt += 1
        ngram_set.add(current)
        rep_scores.append(cnt / (i + 1))
    while len(rep_scores) < len(tokens):
        rep_scores.append(rep_scores[-1])
    return rep_scores

def compute_batch_repetition_score(batch_tokens, n=2):
    batch_rep_scores = []
    for tokens in batch_tokens:
        batch_rep_scores.append(compute_repetition_score(tokens, n))
    return torch.tensor(batch_rep_scores).t()


def compute_naturalness(self, texts, naturalness_llm, naturalness_llm_tokenizer, yes_token=9642, no_token=2822):
    results = []
    chunk_size = 1
    for i in range(0, len(texts), chunk_size):
        naturalness_list = self.compute_naturalness_small_batch(texts[i:i+chunk_size], naturalness_llm, naturalness_llm_tokenizer, yes_token, no_token)
        results.append(naturalness_list)
    return torch.cat(results)


def compute_naturalness_small_batch(texts, naturalness_llm, naturalness_llm_tokenizer, yes_token=9642, no_token=2822):
    tokens_list = []
    for text in texts:
        # query = f"""Is this text intelligible? "{text}". Just answer Yes or No."""
        query = f"""Is this text intelligible? "{text}". Just answer Yes or No."""
        messages = [
            {"role": "user", "content": query},
        ]
        tokens = naturalness_llm_tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        tokens_list.append(tokens)
    max_length = max([len(tokens) for tokens in tokens_list])
    model_input_tokens = torch.zeros((len(tokens_list), max_length)).to(torch.long)
    model_attention_mask = torch.zeros((len(tokens_list), max_length)).to(torch.long)
    for i in range(len(tokens_list)):
        for j in range(len(tokens_list[i])):
            model_input_tokens[i][-1 - j] = tokens_list[i][-1 - j]
            model_attention_mask[i][-1 - j] = 1
    input_ids = torch.tensor(model_input_tokens).to('cuda:1')
    attention_mask = torch.tensor(model_attention_mask).to('cuda:1')
    with torch.no_grad():
        outputs = naturalness_llm(input_ids=input_ids, attention_mask=attention_mask)
        yes_logits = outputs.logits[:, -1, yes_token]
        no_logits = outputs.logits[:, -1, no_token]
        yes_prob = yes_logits
        no_prob = no_logits
    # return (yes_logits < no_logits).float().to('cuda:0')
    return ((yes_prob - no_prob) / (yes_prob + no_prob)).to('cuda:0')

def compute_trigram_overlap(sentence1, sentence2):
    def generate_trigrams(sentence):
        words = sentence.split()  # Split sentence into words
        trigrams = set()
        for i in range(len(words)):
            trigram = tuple(words[i:i+1])  # Form a trigram tuple
            trigrams.add(trigram)
        return trigrams

    # Generate trigrams for both sentences
    trigrams1 = generate_trigrams(sentence1)
    trigrams2 = generate_trigrams(sentence2)

    # Calculate intersection and union of trigrams
    intersection = trigrams1.intersection(trigrams2)
    union = trigrams1.union(trigrams2)

    # Compute overlap ratio
    overlap_ratio = len(intersection) / len(union) if union else 0.0
    return overlap_ratio

def compute_cross_bleu(texts):
    cross_avg_bleus = []
    for i in range(len(texts)):
        bleu_scores = []
        for j in range(len(texts)):
            if i == j:
                continue
            bleu_score = compute_trigram_overlap(texts[i], texts[j])
            bleu_scores.append(bleu_score)
        cross_avg_bleus.append(sum(bleu_scores) / len(bleu_scores))
    return torch.tensor(cross_avg_bleus)

def process_triggers_states(triggers, tokens_batch, gpt2_tokenizer):
    assert(len(triggers) == len(tokens_batch))
    prefix_tokens = gpt2_tokenizer.batch_encode_plus([f"Describe {trigger}:" for trigger in triggers])['input_ids']
    inputs = [pre + token for pre, token in zip(prefix_tokens, tokens_batch)]
    tokenized_inputs = [{'input_ids': s} for s in inputs]
    batch_inputs = gpt2_tokenizer.pad(
        tokenized_inputs,
        return_tensors='pt',
        padding=True,
        max_length=None,
        pad_to_multiple_of=None,
    )
    tokens_batch = batch_inputs['input_ids'].to(device)  # Shape: [batch_size, seq_len]
    attention_mask = batch_inputs['attention_mask'].to(device)  # Shape: [batch_size, seq_len]
    return tokens_batch, attention_mask


def process_triggers_states_different_tokenizer(triggers, tokens_batch, gpt2_tokenizer, contriever_tokenizer):
    sents = gpt2_tokenizer.batch_decode(tokens_batch)
    full_sents = [f"Describe {trigger}: {sent}" for trigger, sent in zip(triggers, sents)]
    batch_inputs = contriever_tokenizer.batch_encode_plus(full_sents, return_tensors='pt', padding=True)
    tokens_batch = batch_inputs['input_ids'].to(device)  # Shape: [batch_size, seq_len]
    attention_mask = batch_inputs['attention_mask'].to(device)  # Shape: [batch_size, seq_len]
    return tokens_batch, attention_mask

class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.llm: vec2text.models.InversionModel = vec2text.models.InversionModel.from_pretrained("jxm/gtr__nq__32")
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/gtr-t5-base")
        self.tokenizer.padding_side = 'left'
        path_name = trigger_dict_path
        self.chunk_size = 32
        self.temperature = 1
        with open('actor_' + path_name, 'rb') as f:
            import pickle
            self.trigger_dict = pickle.load(f)
            for k in self.trigger_dict:
                self.trigger_dict[k] = self.trigger_dict[k].to(torch.get_default_dtype())

    def forward(self, triggers, states, all_logits=False, use_kv_cache=True):
        logits_list = []
        probs_list = []
        target_embs = torch.stack([self.trigger_dict[trigger] for trigger in triggers])
        states = [[self.tokenizer.pad_token_id] + state for state in states]
        states = torch.tensor(states).to(device)
        for i in range(0, len(triggers), self.chunk_size):
            chunk_states = states[i:i+self.chunk_size]
            chunk_target_embs = target_embs[i:i+self.chunk_size]
            if chunk_states.size(1) == 0:
                chunk_states = None
            outputs = self.llm.forward(None, None, frozen_embeddings=chunk_target_embs, decoder_input_ids=chunk_states)
            if all_logits:
                logits = outputs.logits[:, -1-len(states[0]):-1, :]
                probs = nn.functional.softmax(logits / self.temperature, dim=-1)
            else:
                logits = outputs.logits[:, -1, :]
                probs = nn.functional.softmax(logits / self.temperature, dim=-1)
            logits_list.append(logits)
            probs_list.append(probs)
        return torch.cat(logits_list, 0), torch.cat(probs_list, 0)

class EncoderCritic(nn.Module):
    def __init__(self, actor_tokenizer):
        super(EncoderCritic, self).__init__()
        model_name = "sentence-transformers/gtr-t5-base"
        self.encoder = AutoModel.from_pretrained(model_name, output_hidden_states=True).encoder
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Value head to predict scalar value from hidden states
        self.value_head = nn.Linear(self.encoder.config.hidden_size, 1)
        self.actor_tokenizer = actor_tokenizer
        
    def forward(self, triggers, states):
        tokens_batch, attention_mask = process_triggers_states_different_tokenizer(triggers, states, self.actor_tokenizer, self.tokenizer)
        outputs = self.encoder(input_ids=tokens_batch, attention_mask=attention_mask)
        # Get hidden states from the last layer
        last_layer_hidden_states = outputs.hidden_states[-1]  # Shape: [batch_size, seq_len, hidden_size]
        # Use the hidden state corresponding to the last token
        mean_hidden_states =  last_layer_hidden_states.mean(dim=1)  # Shape: [batch_size, hidden_size]
        # Compute the scalar value
        value = self.value_head(mean_hidden_states).squeeze(-1)  # Shape: [batch_size]
        return value

class RewardModel(nn.Module):
    def __init__(self, triggers):
        super(RewardModel, self).__init__()
        ds = load_dataset("microsoft/ms_marco", "v1.1")
        queries = ds['train']['query']
        random_queries = random.sample(queries, 128)
        self.encoder = SentenceTransformer("sentence-transformers/gtr-t5-base", device=device, model_kwargs={'torch_dtype': torch.get_default_dtype()})
        self.trigger_dict = {}
        path_name = trigger_dict_path
        import pickle

        if not os.path.exists('actor_' + path_name):
            inversion_model = vec2text.models.InversionModel.from_pretrained("jxm/gtr__nq__32").to(device)
            tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/gtr-t5-base")
            for trigger in tqdm(triggers, desc="Building trigger embeddings"):
                target_queries = [trigger + query for query in random_queries]
                inputs = tokenizer.batch_encode_plus(target_queries, return_tensors="pt", padding=True, truncation=True).to(device)
                embeddings = inversion_model.call_embedding_model(**inputs)
                self.trigger_dict[trigger] = embeddings.mean(dim=0)
            del inversion_model
            with open(path_name, 'wb') as f:
                pickle.dump(self.trigger_dict, f)

        if not os.path.exists(path_name):
            for trigger in tqdm(triggers, desc="Building trigger embeddings"):
                target_queries = [trigger + query for query in random_queries]
                embeddings = self.encoder.encode(target_queries, convert_to_tensor=True, normalize_embeddings=True)
                self.trigger_dict[trigger] = embeddings.mean(dim=0)
            with open(path_name, 'wb') as f:
                pickle.dump(self.trigger_dict, f)
        with open(path_name, 'rb') as f:
            self.trigger_dict = pickle.load(f)
            for k in self.trigger_dict:
                self.trigger_dict[k] = self.trigger_dict[k].to(torch.get_default_dtype())
            
    def forward(self, triggers, sentences):
        assert(len(triggers) == len(sentences))
        target_embs = torch.stack([self.trigger_dict[trigger] for trigger in triggers])
        with torch.no_grad():  # Freeze the reward model
            embs = self.encoder.encode(sentences, convert_to_tensor=True, normalize_embeddings=True)
            # Compute cosine similarity between embeddings
            reward = torch.sum(embs * target_embs, dim=1)  # Shape: [batch_size]
        cosine_similarity_matrix = torch.mm(embs, embs.T)
        num_embeddings = cosine_similarity_matrix.size(0)
        mask = torch.eye(num_embeddings, dtype=torch.bool, device=cosine_similarity_matrix.device)
        cosine_similarity_matrix_no_self = cosine_similarity_matrix.masked_fill(mask, 0)
        avg_cross_cos_sim = cosine_similarity_matrix_no_self.sum(dim=1) / (num_embeddings - 1)
        return reward, avg_cross_cos_sim

    def get_max_cos_sim(self, triggers):
        mean_cos_sim_tensor = torch.stack([self.trigger_dict[trigger] for trigger in triggers])
        norm_cos_sim_tensor = torch.nn.functional.normalize(mean_cos_sim_tensor, dim=1)
        return torch.sum(mean_cos_sim_tensor * norm_cos_sim_tensor, dim=1)

def actor_inference(actor, triggers, max_steps):
    SHOULD_SAMPLE = False
    states = [[] for _ in triggers]
    for _ in range(max_steps):
        _, probs = actor(triggers, states)
        # Use argmax instead of sampling
        if SHOULD_SAMPLE:
            m = Categorical(probs)
            actions_sampled = m.sample()
            for i in range(len(states)):
                states[i].append(actions_sampled[i].item())
        else:
            actions_argmax = torch.argmax(probs, dim=-1)
            for i in range(len(states)):
                states[i].append(actions_argmax[i].item())
    return states


def eval(actor, reward_model, triggers, max_steps, gpt2_tokenizer):
    states = actor_inference(actor, triggers, max_steps)
    sents = gpt2_tokenizer.batch_decode(states)
    cos_sims, cross_cos_sims = reward_model(triggers, sents)
    texts = []
    for trig, sent, reward in zip(triggers, sents, cos_sims):
        text = repr(sent)
        print(f"Trigger: {trig}, Score: {reward:.4f}, Sent: {text}")
        texts.append(text)
        # train_sample_table.add_data(episode, trig, reward.item(), text)
    return cos_sims, texts

USE_KL = False
USE_Cross_BLEU = False
def train(batch_size, num_episodes, gamma, learning_rate, accumulation_steps, kl_coef, kl_threshold, max_steps, cross_cos_sim_threshold, naturalness_threshold, naturalness_coef, cross_cos_sim_coef, use_naturalness):
    if use_naturalness:
        model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        naturalness_llm_tokenizer = AutoTokenizer.from_pretrained(model_name)
        naturalness_llm = AutoModelForCausalLM.from_pretrained(model_name).to('cuda:1')
    # Initialize Weights & Biases

    with open('keywords.json') as f:
        triggers = json.load(f)
    # remove possible repetitive
    triggers = list(set(triggers))
    random.shuffle(triggers)
    test_triggers = ['homegoods', 'huawei', 'science channel', 'vh1', 'lidl', 'triumph motorcycles', 'avon', 'snapchat', 'steelseries keyboard', 'yeezy', 'laurent-perrier', 'the washington post', 'twitch', 'engadget', 'bruno mars', 'giorgio armani', 'old el paso', 'levis', 'kings', 'ulta beauty']
    triggers = list(set(triggers) - set(test_triggers))
    reward_model = RewardModel(triggers + test_triggers).to(device)

    # Initialize models
    if False:
        actor: Actor = torch.load('/share/shmatikov/collin/adversarial_decoding/models/nlp_topic_actor_160.pth').to(device)
        critic: EncoderCritic = torch.load('/share/shmatikov/collin/adversarial_decoding/models/nlp_topic_critic_160.pth').to(device)
    else:
        actor = Actor().to(device)
        critic = EncoderCritic(actor.tokenizer).to(device)

    # Initialize the reference actor (reference LLM)
    if USE_KL:
        reference_actor = Actor().to(device)
        reference_actor.eval()
        for param in reference_actor.parameters():
            param.requires_grad = False

    # Optimizers
    actor_optimizer = optim.Adam(actor.parameters(), lr=learning_rate)
    critic_optimizer = optim.Adam(critic.parameters(), lr=learning_rate)

    # Set models to training mode
    actor.train()
    critic.train()

    # # Watch models with wandb to log gradients and parameters
    # wandb.watch(actor, log='all', log_freq=100)
    # wandb.watch(critic, log='all', log_freq=100)

    best_test_cos_sim = 0
    log_list = []
    for episode in range(num_episodes):
        # Initialize states with empty tokens
        states = [[] for _ in range(batch_size)]
        log_probs_list = []
        values_list = []
        cos_sims_list = []
        cross_cos_sims_list = []
        kl_losses_list = []  # List to store KL divergence losses
        current_triggers = [random.choice(triggers) for _ in range(batch_size)]

        for t in tqdm(range(max_steps)):
            # Actor predicts next token logits and probabilities
            logits, probs = actor.forward(current_triggers, states)  # Shape: [batch_size, vocab_size]

            if USE_KL:
                # Reference LLM predicts next token logits
                with torch.no_grad():
                    logits_ref, _ = reference_actor(current_triggers, states)
                # Compute KL divergence loss
                log_probs_actor = nn.functional.log_softmax(logits, dim=-1)
                probs_ref = nn.functional.softmax(logits_ref, dim=-1)
                kl_loss_t = nn.functional.kl_div(log_probs_actor, probs_ref, reduction='batchmean')
                kl_losses_list.append(kl_loss_t)

            # Sample actions for active sequences
            m = Categorical(probs)
            SHOULD_EXPLORE = False
            if not SHOULD_EXPLORE:
                actions_sampled = m.sample()
            else:
                exploit_sampled = m.sample()
                explore_sampled = []
                for prob in probs:
                    _, indices = prob.topk(k=20)
                    explore_sampled.append(random.choice(indices))
                explore_sampled = torch.tensor(explore_sampled)
                split = int(batch_size * 0.75)
                actions_sampled = torch.cat([exploit_sampled[:split], explore_sampled[split:]])
            # print("actions_sampled", actions_sampled)
            actions_sampled = m.sample()
            log_probs = m.log_prob(actions_sampled)
            log_probs_list.append(log_probs)
            
            # Critic estimates the value of the current state
            values = critic(current_triggers, states)  # Shape: [batch_size]
            values_list.append(values)
            
            # Update states with the new actions
            for i in range(batch_size):
                states[i].append(actions_sampled[i].item())

            # Get rewards from the reward model
            texts = [actor.tokenizer.decode(s, skip_special_tokens=True) for s in states]
            cos_sims, cross_cos_sims = reward_model(current_triggers, texts)  # Shape: [batch_size]
            cos_sims_list.append(cos_sims)
            cross_cos_sims_list.append(cross_cos_sims)


        actor.kv_cache = None
            
            
        actor.kv_cache = None
            
        # Stack lists to create tensors of shape [num_steps, batch_size]
        log_probs_tensor = torch.stack(log_probs_list)  # Shape: [num_steps, batch_size]
        values_tensor = torch.stack(values_list)        # Shape: [num_steps, batch_size]
        rewards_list = copy.deepcopy(cos_sims_list)

        
        max_cos_sim = reward_model.get_max_cos_sim(current_triggers)
        reward_strategy = 'INCREMENTAL'
        if reward_strategy == 'INCREMENTAL':
            prev_rewards = max_cos_sim
            for t in range(len(rewards_list)):
                # Compute the incremental reward
                # current_rewards = rewards_list[t] - 0.1 * batch_rep_scores[t]
                # penalize cos sim with other sents in the batch to prevent converging to the same sentence, while ignore cos sim below 0.3
                current_rewards = rewards_list[t] - cross_cos_sim_coef * torch.clamp(cross_cos_sims_list[t], min=cross_cos_sim_threshold)
                incremental_reward = current_rewards - prev_rewards
                prev_rewards = current_rewards
                rewards_list[t] = incremental_reward
        else:
            rewards_list = [rewards * (0.8 ** (len(rewards_list) - i - 1)) for i, rewards in enumerate(rewards_list)]
        if use_naturalness:
            naturalness_scores = compute_naturalness_small_batch(actor.tokenizer.batch_decode(states), naturalness_llm, naturalness_llm_tokenizer)
            for i in range(len(states)):
                rewards_list[-1][i] += torch.clamp(naturalness_scores[i], max=naturalness_threshold) * naturalness_coef
        if USE_Cross_BLEU:
            cross_bleu = compute_cross_bleu(actor.tokenizer.batch_decode(states))
            for i in range(len(states)):
                rewards_list[-1][i] -= cross_bleu[i]

        
        rewards_tensor = torch.stack(rewards_list)      # Shape: [num_steps, batch_size]
        returns = torch.zeros_like(rewards_tensor).to(device)
        G = torch.zeros(batch_size).to(device)
        for t in reversed(range(len(rewards_list))):
            G = rewards_tensor[t] + gamma * G
            returns[t] = G

        # Normalize returns and advantages
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        advantages = returns - values_tensor.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Calculate actor (policy) loss
        actor_loss = - (log_probs_tensor * advantages).mean()

        if USE_KL:
            # Calculate average KL divergence loss
            kl_loss = torch.stack(kl_losses_list).mean()
            kl_adjusted = nn.functional.relu(kl_coef * (kl_loss - kl_threshold))

            # Total actor loss with KL divergence regularization
            total_actor_loss = actor_loss + kl_adjusted
        else:
            total_actor_loss = actor_loss


        # Calculate critic (value) loss
        critic_loss = nn.functional.mse_loss(values_tensor, returns)

        # Backpropagate losses
        total_actor_loss.backward()
        critic_loss.backward()

        current_log_dict = {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'mean_train_cos_sim': cos_sims_list[-1].mean().item(),
            'mean_total_rewards': rewards_tensor.sum(dim=0).mean().item(),
            'max_possible_cos_sim': max_cos_sim.mean().item(),
            'mean_cross_cos_sim': cross_cos_sims_list[-1].mean().item()
        }
        if use_naturalness:
            current_log_dict = current_log_dict | {
                'mean_naturalness_score': naturalness_scores.mean().item()
            }
        if USE_KL:
            current_log_dict = current_log_dict | {
                'kl_loss': kl_loss.mean().item(),
                'kl_adjusted': kl_adjusted.mean().item(),
            }
        if USE_Cross_BLEU:
            current_log_dict = current_log_dict | {
                'cross_bleu': cross_bleu.mean().item()
            }
        log_list.append(current_log_dict)
        # Perform optimization step every 'accumulation_steps' episodes
        if (episode + 1) % accumulation_steps == 0:
            # Update actor network
            actor_optimizer.step()
            actor_optimizer.zero_grad()

            # Update critic network
            critic_optimizer.step()
            critic_optimizer.zero_grad()

            df = pd.DataFrame(log_list)
            mean_values = df.mean().to_dict()

            # Log losses and rewards to wandb
            wandb.log({
                'episode': episode + 1,
                **mean_values
            })
            log_list = []


            # Print training progress
            if USE_KL:
                print(f'Episode {episode+1} Actor Loss: {actor_loss.item()*accumulation_steps:.4f}, KL Loss: {kl_loss.item()*accumulation_steps:.4f}, Critic Loss: {critic_loss.item()*accumulation_steps:.4f}')
            else:
                print(f'Episode {episode+1}, Actor Loss: {actor_loss.item()*accumulation_steps:.4f}, Critic Loss: {critic_loss.item()*accumulation_steps:.4f}')
            train_sample_table = wandb.Table(columns=['episode', 'trigger', 'cos_sim', 'generated_text'])
            for trigger, state, cos_sim in zip(current_triggers, states, cos_sims_list[-1]):
                print("Trigger:", trigger, "Cos sim", cos_sim.item())
                text = repr(actor.tokenizer.decode(state, skip_special_tokens=True))
                print("Sample generated text:", text)
                train_sample_table.add_data(episode + 1, trigger, cos_sim.item(), text)
            wandb.log({'train_generated_texts': train_sample_table})
        
        print(f"Episode {episode}, GPU Memory Allocated: {torch.cuda.memory_allocated() / 1e6} MB")

        # Evaluate model periodically
        if (episode) % (accumulation_steps * 20) == 0:
        # if True:
            test_sample_table = wandb.Table(columns=['episode', 'trigger', 'cos_sim', 'generated_text'])
            cos_sims, texts = eval(actor, reward_model, test_triggers, max_steps, actor.tokenizer)
            for trigger, text, cos_sim in zip(test_triggers, texts, cos_sims):
                test_sample_table.add_data(episode, trigger, cos_sim.item(), text)
            wandb.log({
                'mean_test_cos_sim': cos_sims.mean(),
                'test_sample_texts': test_sample_table
            })

            if cos_sims.mean() > best_test_cos_sim:
                best_test_cos_sim = cos_sims.mean()
                torch.save(actor.cpu(), f'/share/shmatikov/collin/adversarial_decoding/models/vec2text/rl_actor.pth')
                torch.save(critic.cpu(), f'/share/shmatikov/collin/adversarial_decoding/models/vec2text/rl_critic.pth')
                with open('/share/shmatikov/collin/adversarial_decoding/models/vec2text/episode_info.json', 'w') as f:
                    info = {
                        'episode': episode,
                        'test_cos_sim': cos_sims.mean().item(),
                        'triggers': test_triggers,
                        'texts': texts
                    }
                    json.dump(info, f, indent=2)
                actor.cuda()
                critic.cuda()


if __name__ == '__main__':
    config = {
        'batch_size': 32,  # Number of sequences to process in parallel
        'num_episodes': 10000,
        'gamma': 0.99,  # Discount factor
        'learning_rate': 1e-4,
        'accumulation_steps': 1,  # Number of steps to accumulate gradients
        'kl_coef': 1,  # Coefficient for KL divergence loss
        'kl_threshold': 1000,
        'max_steps': 16,  # Max tokens per episode
        'cross_cos_sim_threshold': 0.35, 
        'naturalness_coef': 4,
        'cross_cos_sim_coef': 0,
        'naturalness_threshold': 0.06,
        'use_naturalness': False
    }
    wandb.init(project='vec2text_rl', config=config)
    train(**config)