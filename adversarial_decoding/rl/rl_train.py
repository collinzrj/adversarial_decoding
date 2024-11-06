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
gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token  # Set pad token

class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.llm = AutoModelForCausalLM.from_pretrained('gpt2')
        self.temperature = 1
    def forward(self, tokens_batch, attention_mask):
        # Forward pass through GPT-2
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
        
    def forward(self, tokens_batch, attention_mask):
        # Forward pass through GPT-2 with hidden states
        outputs = self.llm(input_ids=tokens_batch, attention_mask=attention_mask)
        # Get hidden states from the last layer
        hidden_states = outputs.hidden_states[-1]  # Shape: [batch_size, seq_len, hidden_size]
        # Use the hidden state corresponding to the last token
        last_hidden_state = hidden_states[:, -1, :]  # Shape: [batch_size, hidden_size]
        # Compute the scalar value
        value = self.value_head(last_hidden_state).squeeze(-1)  # Shape: [batch_size]
        return value

class RewardModel(nn.Module):
    def __init__(self):
        super(RewardModel, self).__init__()
        ds = load_dataset("microsoft/ms_marco", "v1.1")
        queries = ds['train']['query']
        random_queries = random.sample(queries, 128)
        target_queries = ['spotify ' + query for query in random_queries]
        self.encoder = SentenceTransformer("facebook/contriever", device='cuda')
        self.target_embs = self.encoder.encode(target_queries, convert_to_tensor=True, normalize_embeddings=True)
            
    def forward(self, sentences):
        with torch.no_grad():  # Freeze the reward model
            emb = self.encoder.encode(sentences, convert_to_tensor=True, normalize_embeddings=True)
            # Compute cosine similarity between embeddings
            similarity_matrix = torch.mm(emb, self.target_embs.t())  # Shape: [batch_size, num_targets]
            reward = similarity_matrix.mean(dim=1)  # Shape: [batch_size]
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

# Initialize models
actor = Actor().to(device)
critic = Critic().to(device)
reward_model = RewardModel().to(device)

# Optimizers
actor_optimizer = optim.Adam(actor.parameters(), lr=learning_rate)
critic_optimizer = optim.Adam(critic.parameters(), lr=learning_rate)

for episode in range(num_episodes):
    # Initialize states with a start token
    start_text = 'Once upon a time,'
    start_tokens = gpt2_tokenizer.encode(start_text, add_special_tokens=False)
    states = [start_tokens.copy() for _ in range(batch_size)]
    dones = [False for _ in range(batch_size)]
    
    log_probs_list = []
    values_list = []
    rewards_list = []

    for t in tqdm(range(max_steps)):
        # Pad sequences and create attention masks
        tokenized_inputs = [{'input_ids': s} for s in states]
        batch_inputs = gpt2_tokenizer.pad(
            tokenized_inputs,
            return_tensors='pt',
            padding=True,
            max_length=None,
            pad_to_multiple_of=None
        )
        tokens_batch = batch_inputs['input_ids'].to(device)  # Shape: [batch_size, seq_len]
        attention_mask = batch_inputs['attention_mask'].to(device)  # Shape: [batch_size, seq_len]
        
        # Actor predicts next token probabilities
        probs = actor(tokens_batch, attention_mask)  # Shape: [batch_size, vocab_size]
        
        # Sample actions for active sequences
        actions = torch.zeros(batch_size, dtype=torch.long).to(device)
        log_probs = torch.zeros(batch_size).to(device)
        mask = torch.tensor([not done for done in dones], dtype=torch.bool).to(device)
        if mask.any():
            m = Categorical(probs[mask])
            actions_sampled = m.sample()
            actions[mask] = actions_sampled
            log_probs[mask] = m.log_prob(actions_sampled)
        else:
            # All sequences are done
            break
        log_probs_list.append(log_probs)
        
        # Critic estimates the value of the current state
        values = critic(tokens_batch, attention_mask)  # Shape: [batch_size]
        values_list.append(values)
        
        # Update states with the new actions
        for i in range(batch_size):
            if not dones[i]:
                states[i].append(actions[i].item())
                if actions[i].item() == gpt2_tokenizer.eos_token_id:
                    dones[i] = True  # Mark sequence as done

        # Get rewards from the reward model
        texts = [gpt2_tokenizer.decode(s, skip_special_tokens=True) for s in states]
        rewards = reward_model(texts)  # Shape: [batch_size]
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
        G = rewards_tensor[t] + gamma * G * (~torch.tensor(dones, dtype=torch.bool).to(device))
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
