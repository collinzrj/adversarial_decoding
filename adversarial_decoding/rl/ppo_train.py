import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


class PPOWithLLM:
    def __init__(self, actor_model_name, critic_model_name, learning_rate=1e-5):
        # Load Actor (LLM) and Critic (LLM)
        self.actor = AutoModelForCausalLM.from_pretrained(actor_model_name)
        self.critic = AutoModelForCausalLM.from_pretrained(critic_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(actor_model_name)
        
        # Optimizer for the actor
        self.optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)
        
    def generate_action(self, input_ids):
        # Generate token probabilities with the actor model
        outputs = self.actor(input_ids=input_ids)
        logits = outputs.logits[:, -1, :]  # Take the last token logits
        probs = F.softmax(logits, dim=-1)
        
        # Sample a token based on probabilities
        action = torch.multinomial(probs, num_samples=1)
        return action, probs.gather(-1, action)
    
    def evaluate_action(self, input_ids, action):
        # Get critic score for the token chosen by the actor
        inputs = torch.cat((input_ids, action), dim=-1)  # Concatenate action to context
        outputs = self.critic(input_ids=inputs)
        score = outputs.logits[:, -1, :].mean()  # Use the mean logit as the score
        
        return score
    
    def compute_ppo_loss(self, old_log_probs, new_log_probs, advantages, epsilon=0.2):
        # Calculate PPO loss
        ratio = torch.exp(new_log_probs - old_log_probs)
        clip_adv = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages
        loss = -torch.min(ratio * advantages, clip_adv).mean()
        
        return loss
    
    def train(self, input_text, epochs=10):
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids

        for epoch in range(epochs):
            old_log_probs = []
            actions = []
            rewards = []
            
            # Generate tokens and log probabilities
            action, log_prob = self.generate_action(input_ids)
            old_log_probs.append(log_prob)
            actions.append(action)
            
            # Critic evaluates the action
            reward = self.evaluate_action(input_ids, action)
            rewards.append(reward)
            
            # Convert to tensors
            old_log_probs = torch.cat(old_log_probs)
            rewards = torch.tensor(rewards)
            advantages = rewards - rewards.mean()  # Advantage calculation

            # Update actor with PPO loss
            new_log_probs = []
            for action in actions:
                _, log_prob = self.generate_action(input_ids)
                new_log_probs.append(log_prob)
            
            new_log_probs = torch.cat(new_log_probs)
            loss = self.compute_ppo_loss(old_log_probs, new_log_probs, advantages)
            
            # Backpropagate the loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

# Usage
ppo = PPOWithLLM(actor_model_name="gpt2", critic_model_name="gpt2")
ppo.train("Once upon a time")
