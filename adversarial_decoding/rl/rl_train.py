import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import random
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# Initialize the tokenizer and set the pad token
gpt2_tokenizer = AutoTokenizer.from_pretrained('gpt2')
gpt2_tokenizer.padding_side = 'left'
gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token  # Set pad token


def process_triggers_states(triggers, tokens_batch):
    assert(len(triggers) == len(tokens_batch))
    prefix_tokens = gpt2_tokenizer.batch_encode_plus([trigger + gpt2_tokenizer.eos_token for trigger in triggers])['input_ids']
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

class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.llm = AutoModelForCausalLM.from_pretrained('gpt2')
        self.temperature = 1
    def forward(self, triggers, states):
        tokens_batch, attention_mask = process_triggers_states(triggers, states)
        outputs = self.llm(input_ids=tokens_batch, attention_mask=attention_mask)
        # Get the logits for the last token in each sequence
        logits = outputs.logits[:, -1, :]  # Shape: [batch_size, vocab_size]
        # Convert logits to probabilities
        probs = nn.functional.softmax(logits / self.temperature, dim=-1)  # Shape: [batch_size, vocab_size]
        return probs

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.llm = AutoModelForCausalLM.from_pretrained('gpt2', output_hidden_states=True)
        # Value head to predict scalar value from hidden states
        self.value_head = nn.Linear(self.llm.config.hidden_size, 1)
        
    def forward(self, triggers, tokens_batch):
        tokens_batch, attention_mask = process_triggers_states(triggers, states)
        outputs = self.llm(input_ids=tokens_batch, attention_mask=attention_mask)
        # Get hidden states from the last layer
        hidden_states = outputs.hidden_states[-1]  # Shape: [batch_size, seq_len, hidden_size]
        # Use the hidden state corresponding to the last token
        last_hidden_state = hidden_states[:, -1, :]  # Shape: [batch_size, hidden_size]
        # Compute the scalar value
        value = self.value_head(last_hidden_state).squeeze(-1)  # Shape: [batch_size]
        return value

class RewardModel(nn.Module):
    def __init__(self, triggers):
        super(RewardModel, self).__init__()
        ds = load_dataset("microsoft/ms_marco", "v1.1")
        queries = ds['train']['query']
        random_queries = random.sample(queries, 128)
        self.encoder = SentenceTransformer("facebook/contriever", device='cuda')
        self.trigger_dict = {}
        for trigger in tqdm(triggers):
            target_queries = [trigger + query for query in random_queries]
            self.trigger_dict[trigger] = self.encoder.encode(target_queries, convert_to_tensor=True, normalize_embeddings=True).mean(dim=0)
            
    def forward(self, triggers, sentences):
        assert(len(triggers) == len(sentences))
        target_embs = torch.stack([self.trigger_dict[trigger] for trigger in triggers])
        with torch.no_grad():  # Freeze the reward model
            emb = self.encoder.encode(sentences, convert_to_tensor=True, normalize_embeddings=True)
            # Compute cosine similarity between embeddings
            reward = torch.sum(emb * target_embs, dim=1)  # Shape: [batch_size]
        return reward

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_device(device)

# Hyperparameters
batch_size = 32  # Number of sequences to process in parallel
num_episodes = 1000
max_steps = 16  # Max tokens per episode
gamma = 0.9  # Discount factor
learning_rate = 1e-4

with open('keywords.json') as f:
    import json
    triggers = json.load(f)

# Initialize models
actor = Actor().to(device)
critic = Critic().to(device)
reward_model = RewardModel(triggers).to(device)

# Optimizers
actor_optimizer = optim.Adam(actor.parameters(), lr=learning_rate)
critic_optimizer = optim.Adam(critic.parameters(), lr=learning_rate)

for episode in range(num_episodes):
    # Initialize states with a start token
    # start_text = 'Once upon a time,'
    # start_tokens = gpt2_tokenizer.encode(start_text, add_special_tokens=False)
    # states = [start_tokens.copy() for _ in range(batch_size)]
    states = [[220] for _ in range(batch_size)]
    
    log_probs_list = []
    values_list = []
    rewards_list = []
    current_triggers = random.sample(triggers, batch_size)
    # print(current_triggers)

    for t in tqdm(range(max_steps)): 
        # Actor predicts next token probabilities
        probs = actor(current_triggers, states)  # Shape: [batch_size, vocab_size]
        
        # Sample actions for active sequences
        m = Categorical(probs)
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
        texts = [gpt2_tokenizer.decode(s, skip_special_tokens=True) for s in states]
        rewards = reward_model(current_triggers, texts)  # Shape: [batch_size]
        rewards_list.append(rewards)
        
    # Stack lists to create tensors of shape [num_steps, batch_size]
    log_probs_tensor = torch.stack(log_probs_list)  # Shape: [num_steps, batch_size]
    values_tensor = torch.stack(values_list)        # Shape: [num_steps, batch_size]
    if True:
        rewards_list = [rewards * (gamma ** (len(rewards_list) - i - 1)) for i, rewards in enumerate(rewards_list)]
    else:
        new_rewards_list = []
        for i, rewards in enumerate(rewards_list):
            if i == len(rewards_list) - 1:
                new_rewards_list.append(rewards)
            else:
                new_rewards_list.append(rewards * 0.01)
        rewards_list = new_rewards_list
    rewards_tensor = torch.stack(rewards_list)      # Shape: [num_steps, batch_size]
    # Compute returns (discounted rewards)
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
    
    # Calculate critic (value) loss
    critic_loss = nn.functional.mse_loss(values_tensor, returns)
    
    # Update actor network
    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()
    
    # Update critic network
    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()
    
    # Print training progress
    if episode % 5 == 0:
        print(f'Episode {episode}, Actor Loss: {actor_loss.item():.4f}, Critic Loss: {critic_loss.item():.4f}')
        sample_state = random.choice(states)
        print("Sample tokens", sample_state)
        print("Sample generated text:", repr(gpt2_tokenizer.decode(sample_state, skip_special_tokens=True)))
        print("Rewards List mean", [rewards.mean().item() for rewards in rewards_list])
