import torch
import torch.nn as nn
import torch.nn.functional as F

class DESModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=64, hidden_dim=128, num_layers=1):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Embedding layer (padding index is typically 0)
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Recurrent sequential encoder (LSTM)
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers=num_layers, 
            batch_first=True
        )
        
        # Policy head: outputs logits for 3 actions
        # Action 0: Wait & Observe
        # Action 1: Classify Benign
        # Action 2: Classify Ransomware
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)
        )
        
        # Value head: outputs scalar state value V(s)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, sequences):
        """
        Forward pass to process entire sequences in a batch.
        Args:
            sequences: PyTorch tensor of shape [batch_size, seq_len]
        Returns:
            policy_logits: PyTorch tensor of shape [batch_size, seq_len, 3]
            values: PyTorch tensor of shape [batch_size, seq_len, 1]
        """
        # Embed sequences: [batch_size, seq_len, embedding_dim]
        embedded = self.embedding(sequences)
        
        # LSTM output: [batch_size, seq_len, hidden_dim]
        lstm_out, _ = self.lstm(embedded)
        
        # Compute policy logits and values for all steps
        policy_logits = self.policy_head(lstm_out)
        values = self.value_head(lstm_out)
        
        return policy_logits, values
        
    def get_action_dist(self, sequences, step):
        """
        Helper method to get policy action distribution at a specific time step.
        Useful during interactive execution or simulation.
        Args:
            sequences: PyTorch tensor of shape [batch_size, seq_len]
            step: current time step index (int)
        Returns:
            probs: action probabilities at the given step [batch_size, 3]
            value: value function estimate at the given step [batch_size, 1]
        """
        with torch.no_grad():
            policy_logits, values = self.forward(sequences)
            step_logits = policy_logits[:, step, :]
            step_values = values[:, step, :]
            probs = F.softmax(step_logits, dim=-1)
            return probs, step_values

if __name__ == "__main__":
    # Smoke test of model forward pass
    vocab_size = 16
    model = DESModel(vocab_size=vocab_size)
    print(model)
    
    # Dummy sequences batch of size 2, length 10
    dummy_seq = torch.randint(0, vocab_size, (2, 10))
    policy_logits, values = model(dummy_seq)
    
    print("\nPolicy logits shape:", policy_logits.shape) # Expected: [2, 10, 3]
    print("Values shape:", values.shape)                 # Expected: [2, 10, 1]
    
    probs, val = model.get_action_dist(dummy_seq, 5)
    print("\nAction probabilities at step 5:", probs)
    print("Value estimate at step 5:", val)
