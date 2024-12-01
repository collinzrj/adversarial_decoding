import torch, os, pickle
import json
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from datasets import load_dataset
import random
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import wandb
import copy
import pandas as pd
import numpy as np
from peft import LoraConfig, get_peft_model

# Set device
# torch.set_default_dtype(torch.bfloat16)
device = 'cuda:0'

def process_triggers_states(triggers, tokens_batch, tokenizer):
    assert len(triggers) == len(tokens_batch)
    prefix_texts = [f"Describe {trigger}:" for trigger in triggers]
    prefix_tokens = tokenizer(prefix_texts, add_special_tokens=False).input_ids
    inputs = [pre + tok for pre, tok in zip(prefix_tokens, tokens_batch)]
    tokenized_inputs = [{'input_ids': s} for s in inputs]
    batch_inputs = tokenizer.pad(
        tokenized_inputs,
        return_tensors='pt',
        padding=True,
    )
    tokens_batch = batch_inputs['input_ids'].to(device)
    attention_mask = batch_inputs['attention_mask'].to(device)
    return tokens_batch, attention_mask

def process_triggers_states_different_tokenizer(triggers, tokens_batch, actor_tokenizer, critic_tokenizer):
    sents = actor_tokenizer.batch_decode(tokens_batch, skip_special_tokens=True)
    full_sents = [f"Describe {trigger}: {sent}" for trigger, sent in zip(triggers, sents)]
    batch_inputs = critic_tokenizer(full_sents, return_tensors='pt', padding=True)
    tokens_batch = batch_inputs['input_ids'].to(device)
    attention_mask = batch_inputs['attention_mask'].to(device)
    return tokens_batch, attention_mask

class Actor(nn.Module):
    def __init__(self, tokenizer, model_name, use_lora=True):
        super(Actor, self).__init__()
        self.llm = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.get_default_dtype())
        self.temperature = 1.0
        self.chunk_size = 32
        self.tokenizer = tokenizer

        if use_lora:
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                bias="none",
                task_type="CAUSAL_LM"
            )
            self.llm = get_peft_model(self.llm, lora_config, autocast_adapter_dtype=False)

    def forward_with_past(self, input_ids, attention_mask, past_key_values=None):
        outputs = self.llm(input_ids=input_ids, attention_mask=attention_mask, use_cache=True, past_key_values=past_key_values)
        logits = outputs.logits[:, -1, :]  # Get logits for the last token
        probs = nn.functional.softmax(logits / self.temperature, dim=-1)
        past_key_values = outputs.past_key_values  # Update past_key_values
        return logits, probs, past_key_values

    def forward(self, triggers, states, all_logits=False):
        logits_list = []
        probs_list = []
        for i in range(0, len(triggers), self.chunk_size):
            chunk_triggers = triggers[i:i+self.chunk_size]
            chunk_states = states[i:i+self.chunk_size]
            # print("actor tokens batch start", chunk_states)
            tokens_batch, attention_mask = process_triggers_states(chunk_triggers, chunk_states, self.tokenizer)
            # print("actor tokens batch", tokens_batch)
            # print("actor tokens batch", self.tokenizer.batch_decode(tokens_batch))
            outputs = self.llm(input_ids=tokens_batch, attention_mask=attention_mask)
            if all_logits:
                logits = outputs.logits[:, -1-len(states[0]):-1, :]
                probs = nn.functional.softmax(logits / self.temperature, dim=-1)
            else:
                logits = outputs.logits[:, -1, :]
                probs = nn.functional.softmax(logits / self.temperature, dim=-1)
            logits_list.append(logits)
            probs_list.append(probs)
        return torch.cat(logits_list, 0), torch.cat(probs_list, 0)
    

class Critic(nn.Module):
    def __init__(self, tokenizer, model_name, use_lora=False):
        super(Critic, self).__init__()
        self.llm = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.get_default_dtype(), output_hidden_states=True)
        # Value head to predict scalar value from hidden states
        self.value_head = nn.Linear(self.llm.config.hidden_size, 1)
        self.tokenizer = tokenizer
        if use_lora:
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                bias="none",
                task_type="CAUSAL_LM"
            )
            self.llm = get_peft_model(self.llm, lora_config, autocast_adapter_dtype=False)
    
    def forward_with_past(self, input_ids, attention_mask, past_key_values=None):
        outputs = self.llm(input_ids=input_ids, attention_mask=attention_mask, use_cache=True, past_key_values=past_key_values)
        hidden_states = outputs.hidden_states[-1]
        last_hidden_state = hidden_states[:, -1, :]
        value = self.value_head(last_hidden_state).squeeze(-1)
        return value
        
    def forward(self, triggers, states, all_logits=False):
        tokens_batch, attention_mask = process_triggers_states(triggers, states, self.tokenizer)
        outputs = self.llm(input_ids=tokens_batch, attention_mask=attention_mask)
        # Get hidden states from the last layer
        hidden_states = outputs.hidden_states[-1]  # Shape: [batch_size, seq_len, hidden_size]
        # Use the hidden state corresponding to the last token
        if all_logits:
            last_hidden_state = hidden_states[:, -1-len(states[0]):-1, :]  # Shape: [batch_size, len_states, hidden_size]
            value = self.value_head(last_hidden_state).squeeze(-1)  # Shape: [batch_size, len_states]
        else:
            last_hidden_state = hidden_states[:, -1, :]  # Shape: [batch_size, hidden_size]
            # Compute the scalar value
            value = self.value_head(last_hidden_state).squeeze(-1)  # Shape: [batch_size]
        return value

class EncoderCritic(nn.Module):
    def __init__(self, actor_tokenizer, critic_tokenizer):
        super(EncoderCritic, self).__init__()
        self.encoder = AutoModel.from_pretrained("facebook/contriever", torch_dtype=torch.get_default_dtype(), output_hidden_states=True)
        self.value_head = nn.Linear(self.encoder.config.hidden_size, 1)
        self.actor_tokenizer = actor_tokenizer
        self.critic_tokenizer = critic_tokenizer

    def forward(self, triggers, states):
        tokens_batch, attention_mask = process_triggers_states_different_tokenizer(
            triggers, states, self.actor_tokenizer, self.critic_tokenizer)
        outputs = self.encoder(input_ids=tokens_batch, attention_mask=attention_mask)
        last_hidden_states = outputs.last_hidden_state.mean(dim=1)
        values = self.value_head(last_hidden_states).squeeze(-1)
        return values

class RewardModel(nn.Module):
    def __init__(self, triggers, use_naturalness):
        super(RewardModel, self).__init__()
        ds = load_dataset("microsoft/ms_marco", "v1.1")
        queries = ds['train']['query']
        random_queries = random.sample(queries, 128)
        self.encoder = SentenceTransformer("facebook/contriever", device=device, model_kwargs={'torch_dtype': torch.get_default_dtype()})
        self.trigger_dict = {}
        self.use_naturalness = use_naturalness
        if use_naturalness:
            self.naturalness_llm = AutoModelForCausalLM.from_pretrained('meta-llama/Meta-Llama-3.1-8B-Instruct', torch_dtype=torch.get_default_dtype()).to('cuda:1')
            self.naturalness_llm_tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3.1-8B-Instruct')
        path_name = 'trigger_dict.pt'
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
        assert len(triggers) == len(sentences)
        target_embs = torch.stack([self.trigger_dict[trigger] for trigger in triggers])
        with torch.no_grad():
            embs = self.encoder.encode(sentences, convert_to_tensor=True, normalize_embeddings=True)
            reward = torch.sum(embs * target_embs, dim=1)
        cosine_similarity_matrix = torch.mm(embs, embs.T)
        num_embeddings = cosine_similarity_matrix.size(0)
        mask = torch.eye(num_embeddings, dtype=torch.bool, device=cosine_similarity_matrix.device)
        cosine_similarity_matrix.masked_fill_(mask, 0)
        avg_cross_cos_sim = cosine_similarity_matrix.sum(dim=1) / (num_embeddings - 1)
        return reward, avg_cross_cos_sim

    def get_max_cos_sim(self, triggers):
        mean_cos_sim_tensor = torch.stack([self.trigger_dict[trigger] for trigger in triggers])
        norm_cos_sim_tensor = torch.nn.functional.normalize(mean_cos_sim_tensor, dim=1)
        return torch.sum(mean_cos_sim_tensor * norm_cos_sim_tensor, dim=1)
    
    def compute_naturalness_small_batch(self, texts, yes_token=9642, no_token=2822):
        tokens_list = []
        for text in texts:
            query = f"""Is this text intelligible? "{text}". Just answer Yes or No."""
            messages = [{"role": "user", "content": query}]
            tokens = self.naturalness_llm_tokenizer.apply_chat_template(messages, add_generation_prompt=True)
            tokens_list.append(tokens)
        max_length = max(len(tokens) for tokens in tokens_list)
        model_input_tokens = torch.zeros((len(tokens_list), max_length), dtype=torch.long)
        model_attention_mask = torch.zeros((len(tokens_list), max_length), dtype=torch.long)
        for i, tokens in enumerate(tokens_list):
            seq_len = len(tokens)
            model_input_tokens[i, -seq_len:] = torch.tensor(tokens, dtype=torch.long)
            model_attention_mask[i, -seq_len:] = 1
        input_ids = model_input_tokens.to('cuda:1')
        attention_mask = model_attention_mask.to('cuda:1')
        with torch.no_grad():
            outputs = self.naturalness_llm(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits[:, -1, :]
            yes_logits = logits[:, yes_token]
            no_logits = logits[:, no_token]
            scores = ((yes_logits - no_logits) / (yes_logits + no_logits)).to(device)
        return scores

def actor_inference(actor, triggers, max_steps, tokenizer):
    states = [[] for _ in triggers]
    for _ in range(max_steps):
        _, probs = actor(triggers, states)
        actions_argmax = torch.argmax(probs, dim=-1)
        for i in range(len(states)):
            states[i].append(actions_argmax[i].item())
    return states

def evaluate(actor, reward_model, triggers, max_steps, tokenizer):
    states = actor_inference(actor, triggers, max_steps, tokenizer)
    sents = tokenizer.batch_decode(states, skip_special_tokens=True)
    cos_sims, cross_cos_sims = reward_model(triggers, sents)
    texts = []
    for trig, sent, reward in zip(triggers, sents, cos_sims):
        text = repr(sent)
        print(f"Trigger: {trig}, Score: {reward:.4f}, Sent: {text}")
        texts.append(text)
    return cos_sims, texts

def collect_trajectories(triggers, actor, critic, reward_model, actor_tokenizer, critic_tokenizer, config):
    batch_size = config['batch_size']
    max_steps = config['max_steps']
    gamma = config['gamma']
    gae_lambda = config['gae_lambda']
    naturalness_threshold = config['naturalness_threshold']
    naturalness_coef = config['naturalness_coef']
    cross_cos_sim_threshold = config['cross_cos_sim_threshold']
    cross_cos_sim_coef = config['cross_cos_sim_coef']

    # Sample triggers
    current_triggers = [random.choice(triggers) for _ in range(batch_size)]
    
    # Initialize states and past_key_values
    states = [[] for _ in range(batch_size)]
    log_probs_list = []
    values_list = []
    cos_sims_list = []
    cross_cos_sims_list = []
    actions_list = []
    texts_list = []

    # Process triggers to get initial tokens and attention masks
    prefix_texts = [f"Describe {trigger}:" for trigger in current_triggers]
    prefix_tokens = actor_tokenizer(prefix_texts, add_special_tokens=False, return_tensors='pt', padding=True)
    tokens_batch = prefix_tokens['input_ids'].to(device)  # Shape: (batch_size, seq_len)
    attention_mask = prefix_tokens['attention_mask'].to(device)
    past_key_values = None  # Initialize past_key_values

    for t in range(max_steps):
        # Get logits, probs, and update past_key_values using kv cache        
        values = critic.forward_with_past(tokens_batch, attention_mask, past_key_values)
        values_list.append(values)

        logits, probs, past_key_values = actor.forward_with_past(tokens_batch, attention_mask, past_key_values)
        test_logits, test_probs = actor(current_triggers, states)
        print('logits diff', torch.sum(logits-test_logits))
        print('probs diff', torch.sum(probs-test_probs))
        m = Categorical(probs)
        actions = m.sample()
        log_probs = m.log_prob(actions)
        log_probs_list.append(log_probs)
        actions_list.append(actions)

        # Update tokens_batch with the new actions (tokens)
        tokens_batch = actions.unsqueeze(-1)  # Shape: (batch_size, 1)
        attention_mask = torch.ones_like(tokens_batch).to(device)  # Since we're adding one token at a time


        # Update states
        for i in range(batch_size):
            states[i].append(actions[i].item())

        # Since critic evaluates the whole sequence, we can process it separately if needed
        # For demonstration, we'll skip kv caching for the critic in this example

        # Get texts and compute rewards
        texts = actor_tokenizer.batch_decode([s for s in states], skip_special_tokens=True)
        texts_list.append(texts)
        cos_sims, cross_cos_sims = reward_model(current_triggers, texts)
        cos_sims_list.append(cos_sims)
        cross_cos_sims_list.append(cross_cos_sims)

    exit()

    # Compute rewards
    rewards_list = []
    for t in range(max_steps):
        if t == 0:
            prev_cos_sim = torch.zeros_like(cos_sims_list[0])
        else:
            prev_cos_sim = cos_sims_list[t-1]
        incremental_reward = cos_sims_list[t] - prev_cos_sim
        # Subtract cross cosine similarity as a penalty
        # incremental_reward -= cross_cos_sim_coef * cross_cos_sims_list[t]
        rewards_list.append(incremental_reward)


    # If using naturalness scores, we can add them to the final reward
    actual_naturalness_list = None
    if reward_model.use_naturalness:
        final_texts = actor_tokenizer.batch_decode(states, skip_special_tokens=True)
        naturalness_scores = reward_model.compute_naturalness_small_batch(final_texts)
        actual_naturalness_list = naturalness_scores
        naturalness_scores = torch.clamp(naturalness_scores, max=naturalness_threshold) * naturalness_coef
        rewards_list[-1] += naturalness_scores

    # Collect terminal values
    final_values = critic.forward_with_past(tokens_batch, attention_mask, past_key_values)
    values_list.append(final_values)  # Now values_list has length max_steps + 1

    # Convert lists to tensors
    rewards_tensor = torch.stack(rewards_list)  # Shape: (max_steps, batch_size)
    values_tensor = torch.stack(values_list)    # Shape: (max_steps + 1, batch_size)

    # Compute GAE
    advantages = torch.zeros_like(rewards_tensor).to(device)
    gae = torch.zeros(batch_size).to(device)
    deltas = rewards_tensor + gamma * values_tensor[1:] - values_tensor[:-1]
    for t in reversed(range(max_steps)):
        gae = deltas[t] + gamma * gae_lambda * gae
        advantages[t] = gae
    returns = advantages + values_tensor[:-1]

    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return {
        'states': states,
        'actions': actions_list, # Shape: (max_steps, batch_size)
        'log_probs': log_probs_list, # Shape: (max_steps, batch_size)
        'values': values_list[:-1],
        'returns': returns,
        'advantages': advantages, # Shape: (max_steps, batch_size)
        'triggers': current_triggers,
        'cos_sims': cos_sims_list[-1],
        'cross_cos_sims': cross_cos_sims[-1],
        'texts': texts_list[-1],
        'max_cos_sim': reward_model.get_max_cos_sim(current_triggers),
        'naturalness_list': actual_naturalness_list
    }

def ppo_update(actor, critic, actor_optimizer, critic_optimizer, data, config, critic_only=False):
    num_epochs = config['num_epochs']
    epsilon = config['ppo_clip']
    entropy_coef = config['entropy_coef']  # Entropy coefficient from config

    actor_loss_list = []
    critic_loss_list = []
    entropy_list = []
    for epoch in range(num_epochs):

        flatten_advantages = data['advantages'].view((-1))
        old_log_probs = torch.stack(data['log_probs']) # Shape: (max_steps, batch_size)
        flatten_old_log_probs = old_log_probs.view((-1))
        flatten_actions = torch.stack(data['actions']).view(-1) # Shape: (max_steps, batch_size)

        if not critic_only:
            logits, probs = actor(data['triggers'], data['states'], all_logits=True) # Shape: (batch_size, max_steps, vocab_size)
            probs = torch.permute(probs, (1, 0, 2)) # Shape: (max_steps, batch_size, vocab_size)
            flatten_probs = probs.reshape((-1, probs.size(-1)))
            m = Categorical(flatten_probs)
            new_log_probs = m.log_prob(flatten_actions) 
            entropy = m.entropy()

            ratio = torch.exp(new_log_probs - flatten_old_log_probs)
            print(f'{epoch}, difference {torch.sum(new_log_probs - flatten_old_log_probs)}', flush=True)
            surr1 = ratio * flatten_advantages
            surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * flatten_advantages
            actor_loss = (-torch.min(surr1, surr2) - entropy_coef * torch.clamp(entropy, max=config['entropy_threshold'])).mean()

            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            actor_loss_list.append(actor_loss)
            entropy_list.append(entropy)

        values = critic(data['triggers'], data['states'], all_logits=True) # Shape: (batch_size, max_steps)
        values = torch.permute(values, (1, 0))
        critic_loss = nn.functional.mse_loss(values, data['returns'])

        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()
        critic_loss_list.append(critic_loss)

    # Logging average metrics for this epoch (optional)
    data['critic_loss'] = torch.tensor(critic_loss_list).mean().detach()
    if not critic_only:
        data['actor_loss'] = torch.tensor(actor_loss_list).mean().detach()
        data['entropy'] = torch.stack(entropy_list).mean().detach()

def train(config):
    # Initialize tokenizers and models
    actor_tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    critic_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    actor_tokenizer.padding_side = 'left'
    actor_tokenizer.pad_token = actor_tokenizer.eos_token
    critic_tokenizer.pad_token = critic_tokenizer.cls_token

    # Load triggers
    with open('keywords.json') as f:
        triggers_list = json.load(f)
    triggers_set = set(triggers_list)
    test_triggers = ['homegoods', 'huawei', 'science channel', 'vh1', 'lidl', 'triumph motorcycles',
                     'avon', 'snapchat', 'steelseries keyboard', 'yeezy', 'laurent-perrier', 'the washington post',
                     'twitch', 'engadget', 'bruno mars', 'giorgio armani', 'old el paso', 'levis', 'kings', 'ulta beauty']
    train_triggers = list(triggers_set - set(test_triggers))
    random.shuffle(train_triggers)

    # Initialize models
    reward_model = RewardModel(train_triggers + test_triggers, config['use_naturalness'])
    if True:
        actor = Actor(actor_tokenizer, config['model_name']).to(device)
        critic = Critic(actor_tokenizer, config['model_name']).to(device)
    else:
        actor = torch.load('actor.pth').to(device)
        critic = torch.load('critic.pth').to(device)

    # Optimizer
    actor_optimizer = optim.Adam(actor.parameters(), lr=config['learning_rate'] * 10)
    critic_optimizer = optim.Adam(critic.parameters(), lr=config['learning_rate'] * 10)
    scheduler = optim.lr_scheduler.StepLR(critic_optimizer, 1, gamma=0.1)

    # Training loop
    for episode in range(config['num_episodes']):
        with torch.no_grad():
            data = collect_trajectories(train_triggers, actor, critic, reward_model, actor_tokenizer, critic_tokenizer, config)
        critic_only = False
        critic_only_episodes = 0
        if episode < critic_only_episodes:
            critic_only = True
        if episode == critic_only_episodes:
            scheduler.step()
        ppo_update(actor, critic, actor_optimizer, critic_optimizer, data, config, critic_only)

        # Logging
        log_data = {
            'episode': episode + 1,
            'critic_loss': data['critic_loss'].item(),
            'mean_train_cos_sim': data['cos_sims'].mean().item(),
            'mean_cross_cos_sims': data['cross_cos_sims'].mean().item(),
            'max_possible_cos_sim': data['max_cos_sim'].mean().item(),
            'train_generated_texts': wandb.Table(columns=['Episode', 'Trigger', 'Cosine Similarity', 'Generated Text'],
                                                data=list(zip([episode + 1 for _ in range(config['batch_size'])], data['triggers'], data['cos_sims'], actor_tokenizer.batch_decode(data['states']))))
        }
        if reward_model.use_naturalness:
            log_data = log_data | {
                'naturalness_list': data['naturalness_list'].mean().item()
            }
        if not critic_only:
            log_data = log_data | {
                'actor_loss': data['actor_loss'].item(),
                'entropy': data['entropy'].item(),
            }
        commit = True
        if (episode + 1) % config['eval_interval'] == 0:
            commit = False
        wandb.log(log_data, commit=commit)
        # Print progress
        # print(f"Episode {episode+1}, Actor Loss: {log_data['actor_loss']:.4f}, Critic Loss: {log_data['critic_loss']:.4f}")

        # Evaluate periodically
        if (episode + 1) % config['eval_interval'] == 0:
            num_test_trig = len(test_triggers)
            cos_sims, texts = evaluate(actor, reward_model, test_triggers, config['max_steps'], actor_tokenizer)
            wandb.log({
                'mean_test_cos_sim': cos_sims.mean().item(),
                'test_generated_texts': wandb.Table(columns=['Episode', 'Trigger', 'Cosine Similarity', 'Generated Text'],
                                                    data=list(zip([episode + 1 for _ in range(num_test_trig)], test_triggers, cos_sims, texts)))
            })
            torch.save(actor, f"actor.pth")
            torch.save(critic, f"critic.pth")

if __name__ == '__main__':
    config = {
        'batch_size': 32,
        'num_episodes': 10000,
        'gamma': 0.99,
        'gae_lambda': 0.95,  # Added GAE lambda parameter
        'learning_rate': 1e-5,
        'max_steps': 16,
        'cross_cos_sim_threshold': 0.35,
        'naturalness_coef': 4,
        'cross_cos_sim_coef': 1.2,
        'naturalness_threshold': 0.05,
        'use_naturalness': False,
        'num_epochs': 4,
        'minibatch_size': 32,
        'ppo_clip': 0.2,  # Adjusted epsilon value to 0.2 (common in PPO)
        'entropy_threshold': 2,
        'entropy_coef': 0.025,  # NEW: Entropy coefficient
        'eval_interval': 40,
        'model_name': 'Qwen/Qwen2.5-0.5B'
        # 'model_name': 'Qwen/Qwen2.5-3B-Instruct'
        # 'model_name': 'meta-llama/Llama-3.2-3B-Instruct'
    }
    wandb.init(project='ppo_attack', config=config)
    train(config)
