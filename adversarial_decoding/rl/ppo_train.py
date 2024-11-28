import torch
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
    def __init__(self, tokenizer, model_name, use_lora=False):
        super(Actor, self).__init__()
        self.llm = AutoModelForCausalLM.from_pretrained(model_name)
        self.temperature = 1.0
        self.chunk_size = 64
        self.tokenizer = tokenizer

        if use_lora:
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                bias="none",
                task_type="CAUSAL_LM"
            )
            self.llm = get_peft_model(self.llm, lora_config)

    def forward(self, triggers, states):
        logits_list = []
        probs_list = []
        for i in range(0, len(triggers), self.chunk_size):
            chunk_triggers = triggers[i:i+self.chunk_size]
            chunk_states = states[i:i+self.chunk_size]
            tokens_batch, attention_mask = process_triggers_states(chunk_triggers, chunk_states, self.tokenizer)
            outputs = self.llm(input_ids=tokens_batch, attention_mask=attention_mask)
            logits = outputs.logits[:, -1, :]
            probs = nn.functional.softmax(logits / self.temperature, dim=-1)
            logits_list.append(logits)
            probs_list.append(probs)
        return torch.cat(logits_list, 0), torch.cat(probs_list, 0)

class EncoderCritic(nn.Module):
    def __init__(self, actor_tokenizer, critic_tokenizer):
        super(EncoderCritic, self).__init__()
        self.encoder = AutoModel.from_pretrained("facebook/contriever", output_hidden_states=True)
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
        self.encoder = SentenceTransformer("facebook/contriever", device=device)
        self.trigger_dict = {}
        self.use_naturalness = use_naturalness
        if use_naturalness:
            self.naturalness_llm = AutoModelForCausalLM.from_pretrained('meta-llama/Meta-Llama-3.1-8B-Instruct').to('cuda:1')
            self.naturalness_llm_tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3.1-8B-Instruct')
        for trigger in tqdm(triggers, desc="Building trigger embeddings"):
            target_queries = [trigger + query for query in random_queries]
            embeddings = self.encoder.encode(target_queries, convert_to_tensor=True, normalize_embeddings=True)
            self.trigger_dict[trigger] = embeddings.mean(dim=0)

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
            print("naturalness llm device", self.naturalness_llm.device)
            print("input ids device", input_ids.device)
            print('attention_mask device', attention_mask.device)
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
    naturalness_threshold = config['naturalness_threshold']
    naturalness_coef = config['naturalness_coef']
    cross_cos_sim_threshold = config['cross_cos_sim_threshold']
    cross_cos_sim_coef = config['cross_cos_sim_coef']

    # Sample triggers
    current_triggers = [random.choice(triggers) for _ in range(batch_size)]
    states = [[] for _ in range(batch_size)]
    log_probs_list = []
    values_list = []
    cos_sims_list = []
    cross_cos_sims_list = []
    actions_list = []

    for t in range(max_steps):
        logits, probs = actor(current_triggers, states)
        m = Categorical(probs)
        actions = m.sample()
        log_probs = m.log_prob(actions)
        log_probs_list.append(log_probs)
        actions_list.append(actions)
        for i in range(batch_size):
            states[i].append(actions[i].item())
        values = critic(current_triggers, states)
        values_list.append(values)
        texts = actor_tokenizer.batch_decode(states, skip_special_tokens=True)
        cos_sims, cross_cos_sims = reward_model(current_triggers, texts)
        cos_sims_list.append(cos_sims)
        cross_cos_sims_list.append(cross_cos_sims)

    # Compute rewards
    rewards_list = copy.deepcopy(cos_sims_list)
    max_cos_sim = reward_model.get_max_cos_sim(current_triggers)
    prev_rewards = max_cos_sim
    if False:
        for t in range(len(rewards_list)):
            # current_rewards = rewards_list[t] - cross_cos_sim_coef * torch.clamp(cross_cos_sims_list[t], min=cross_cos_sim_threshold)
            current_rewards = rewards_list[t]
            incremental_reward = current_rewards - prev_rewards
            prev_rewards = current_rewards
            rewards_list[t] = incremental_reward
    else:
        # Compute rewards
        rewards_list = [torch.zeros_like(cos_sims_list[0]) for _ in range(len(cos_sims_list))]
        rewards_list[-1] = cos_sims_list[-1]  # Reward only at the final timestep

    if reward_model.use_naturalness:
        final_texts = actor_tokenizer.batch_decode(states, skip_special_tokens=True)
        naturalness_scores = reward_model.compute_naturalness_small_batch(final_texts)
        naturalness_scores = torch.clamp(naturalness_scores, max=naturalness_threshold) * naturalness_coef
        rewards_list[-1] += naturalness_scores

    # Compute returns and advantages
    rewards_tensor = torch.stack(rewards_list)
    returns = torch.zeros_like(rewards_tensor).to(device)
    G = torch.zeros(batch_size).to(device)
    for t in reversed(range(len(rewards_list))):
        G = rewards_tensor[t] + gamma * G
        returns[t] = G
    # o1 says this is wrong
    # returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    values_tensor = torch.stack(values_list)
    advantages = returns - values_tensor.detach()
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return {
        'states': states,
        'actions': actions_list,
        'log_probs': log_probs_list,
        'values': values_list,
        'returns': returns,
        'advantages': advantages,
        'triggers': current_triggers,
        'cos_sims': cos_sims_list[-1],
        'texts': texts,
        'max_cos_sim': max_cos_sim,
    }

def ppo_update(actor, critic, actor_optimizer, critic_optimizer, data, config):
    num_epochs = config['num_epochs']
    minibatch_size = config['minibatch_size']
    epsilon = config['ppo_clip']

    # Flatten data
    states_list_flat = []
    actions_list_flat = []
    old_log_probs_list_flat = []
    returns_list_flat = []
    advantages_list_flat = []
    triggers_list_flat = []

    batch_size = len(data['states'])
    max_steps = len(data['actions'])
    for i in range(batch_size):
        for t in range(max_steps):
            states_list_flat.append(data['states'][i][:t+1])
            actions_list_flat.append(data['actions'][t][i])
            old_log_probs_list_flat.append(data['log_probs'][t][i])
            returns_list_flat.append(data['returns'][t][i])
            advantages_list_flat.append(data['advantages'][t][i])
            triggers_list_flat.append(data['triggers'][i])

    total_samples = len(states_list_flat)
    indices = np.arange(total_samples)
    for epoch in range(num_epochs):
        actor_optimizer.zero_grad()
        critic_optimizer.zero_grad()
        np.random.shuffle(indices)

        # Initialize lists to collect data
        collected_new_log_probs = []
        collected_mb_advantages = []
        collected_mb_old_log_probs = []
        collected_values = []
        collected_mb_returns = []

        for start in tqdm(range(0, total_samples, minibatch_size)):
            end = start + minibatch_size
            mb_indices = indices[start:end]
            mb_states = [states_list_flat[idx] for idx in mb_indices]
            mb_actions = torch.tensor([actions_list_flat[idx] for idx in mb_indices], device=device)
            mb_old_log_probs = torch.tensor([old_log_probs_list_flat[idx] for idx in mb_indices], device=device)
            mb_returns = torch.tensor([returns_list_flat[idx] for idx in mb_indices], device=device)
            mb_advantages = torch.tensor([advantages_list_flat[idx] for idx in mb_indices], device=device)
            mb_triggers = [triggers_list_flat[idx] for idx in mb_indices]

            # Prepare inputs for actor
            tokens_batch, attention_mask = process_triggers_states(mb_triggers, mb_states, actor.tokenizer)
            tokens_batch = tokens_batch.to(device)
            attention_mask = attention_mask.to(device)

            # Forward pass through actor
            outputs = actor.llm(input_ids=tokens_batch, attention_mask=attention_mask)
            logits = outputs.logits[:, -1, :]  # Only the last token's logits
            log_probs = nn.functional.log_softmax(logits, dim=-1)
            new_log_probs = log_probs.gather(1, mb_actions.unsqueeze(-1)).squeeze(-1)

            # Collect data for later loss computation
            collected_new_log_probs.append(new_log_probs)
            collected_mb_advantages.append(mb_advantages)
            collected_mb_old_log_probs.append(mb_old_log_probs)

            # Forward pass through critic
            values = critic(mb_triggers, mb_states)
            collected_values.append(values)
            collected_mb_returns.append(mb_returns)

        # Concatenate all collected data
        all_new_log_probs = torch.cat(collected_new_log_probs, dim=0)
        all_mb_advantages = torch.cat(collected_mb_advantages, dim=0)
        all_mb_old_log_probs = torch.cat(collected_mb_old_log_probs, dim=0)
        all_values = torch.cat(collected_values, dim=0)
        all_mb_returns = torch.cat(collected_mb_returns, dim=0)

        # Compute PPO loss for the actor
        ratio = torch.exp(all_new_log_probs - all_mb_old_log_probs)
        surr1 = ratio * all_mb_advantages
        surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * all_mb_advantages
        # problem seems to come from here... the reward is negative, why?
        actor_loss = -torch.min(surr1, surr2).mean()

        # Compute value loss for the critic
        critic_loss = nn.functional.mse_loss(all_values, all_mb_returns)

        actor_loss.backward()
        actor_optimizer.step()
        critic_loss.backward()
        critic_optimizer.step()

        # Logging average losses
        data['actor_loss'] = actor_loss
        data['critic_loss'] = critic_loss


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
    actor = Actor(actor_tokenizer, config['model_name']).to(device)
    critic = EncoderCritic(actor_tokenizer, critic_tokenizer).to(device)

    # Optimizer
    actor_optimizer = optim.Adam(actor.parameters(), lr=config['learning_rate'])
    critic_optimizer = optim.Adam(critic.parameters(), lr=config['learning_rate'])

    # Training loop
    for episode in range(config['num_episodes']):
        with torch.no_grad():
            data = collect_trajectories(train_triggers, actor, critic, reward_model, actor_tokenizer, critic_tokenizer, config)
        ppo_update(actor, critic, actor_optimizer, critic_optimizer, data, config)

        # Logging
        log_data = {
            'episode': episode + 1,
            'actor_loss': data['actor_loss'].item(),
            'critic_loss': data['critic_loss'].item(),
            'mean_train_cos_sim': data['cos_sims'].mean().item(),
            'max_possible_cos_sim': data['max_cos_sim'].mean().item(),
        }
        wandb.log(log_data)

        # Print progress
        print(f"Episode {episode+1}, Actor Loss: {log_data['actor_loss']:.4f}, Critic Loss: {log_data['critic_loss']:.4f}")

        # Evaluate periodically
        if (episode + 1) % config['eval_interval'] == 0:
            cos_sims, texts = evaluate(actor, reward_model, test_triggers, config['max_steps'], actor_tokenizer)
            wandb.log({
                'mean_test_cos_sim': cos_sims.mean().item(),
                'test_generated_texts': wandb.Table(columns=['Trigger', 'Cosine Similarity', 'Generated Text'],
                                                    data=list(zip(test_triggers, cos_sims.cpu().numpy(), texts)))
            })
            torch.save(actor.state_dict(), f"actor.pth")
            torch.save(critic.state_dict(), f"critic.pth")

if __name__ == '__main__':
    config = {
        'batch_size': 16,
        'num_episodes': 1000,
        'gamma': 0.99,
        'learning_rate': 2e-5,
        'max_steps': 16,
        'cross_cos_sim_threshold': 0.35,
        'naturalness_coef': 4,
        'cross_cos_sim_coef': 1.2,
        'naturalness_threshold': 0.06,
        'use_naturalness': False,
        'num_epochs': 1,
        'minibatch_size': 64,
        'ppo_clip': 0.5,
        'eval_interval': 10,
        'model_name': 'Qwen/Qwen2-0.5B-Instruct'
    }
    wandb.init(project='ppo_attack', config=config)
    train(config)
