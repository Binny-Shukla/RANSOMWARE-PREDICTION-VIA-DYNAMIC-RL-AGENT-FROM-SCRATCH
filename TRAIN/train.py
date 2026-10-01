import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from data_loader import get_dataloaders, API_TO_ID
from models import DESModel

# Hyperparameters
EMBEDDING_DIM = 64
HIDDEN_DIM = 128
NUM_LAYERS = 1
BATCH_SIZE = 64
NUM_EPOCHS = 50
LEARNING_RATE = 1e-3

# RL Environment / Reward parameters
ALPHA = 0.02         # Step delay penalty
BETA = 0.5           # Damage penalty per write/delete call
GAMMA = 10.0         # Reward for correct classification
PENALTY_FN = 15.0    # Penalty for False Negative (Ransomware classified as Benign)
PENALTY_FP = 5.0     # Penalty for False Positive (Benign classified as Ransomware)
GAMMA_DISCOUNT = 0.99 # Reward discount factor
ENTROPY_COEF = 0.02  # Entropy coefficient for exploration
VALUE_COEF = 0.5     # Value loss weight

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_episode_batch(model, batch, device, train_mode=True):
    """
    Runs the early stopping agent over a batch of sequences in a fully vectorized manner.
    """
    sequences = batch["sequence"].to(device)
    damages = batch["damage"].to(device)
    lengths = batch["length"].to(device)
    labels = batch["label"].to(device) # 0 for Benign, 1 for Ransomware
    
    batch_size, seq_len = sequences.shape
    
    # Forward pass through the sequence encoder once to get all time-step outputs
    policy_logits, values = model(sequences)
    
    active = torch.ones(batch_size, dtype=torch.bool, device=device)
    stop_steps = torch.zeros(batch_size, dtype=torch.long, device=device)
    rewards = torch.zeros(batch_size, seq_len, device=device)
    actions_taken = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    log_probs = torch.zeros(batch_size, seq_len, device=device)
    entropies = torch.zeros(batch_size, seq_len, device=device)
    
    for t in range(seq_len):
        if not active.any():
            break
            
        logits_t = policy_logits[:, t, :].clone()
        
        # Action masking: If we reach the end of the sequence, the agent MUST classify (1 or 2).
        mask_wait = (t >= (lengths - 1))
        logits_t[mask_wait, 0] = -1e9
        
        probs_t = F.softmax(logits_t, dim=-1)
        dist = torch.distributions.Categorical(probs_t)
        
        if train_mode:
            action = dist.sample()
        else:
            action = torch.argmax(probs_t, dim=-1)
            
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        # Vectorized step updates
        # 1. Wait reward
        wait_rewards = -ALPHA - BETA * damages[:, t]
        
        # 2. Benign terminal reward (Action 1)
        benign_rewards = torch.where(labels == 0.0, torch.tensor(GAMMA, device=device), torch.tensor(-PENALTY_FN, device=device))
        
        # 3. Ransomware terminal reward (Action 2)
        # Compute cumulative damage up to t
        cum_dmg_t = damages[:, :t+1].sum(dim=1)
        ransom_rewards = torch.where(labels == 1.0, GAMMA - BETA * cum_dmg_t, torch.tensor(-PENALTY_FP, device=device))
        
        # Select rewards based on actions
        step_rewards = torch.zeros(batch_size, device=device)
        step_rewards = torch.where(action == 0, wait_rewards, step_rewards)
        step_rewards = torch.where(action == 1, benign_rewards, step_rewards)
        step_rewards = torch.where(action == 2, ransom_rewards, step_rewards)
        
        # Record only for active processes
        actions_taken[active, t] = action[active]
        log_probs[active, t] = log_prob[active]
        entropies[active, t] = entropy[active]
        rewards[active, t] = step_rewards[active]
        stop_steps[active] = t
        
        # Deactivate finished processes
        active = active & (action == 0)
        
    # Vectorized Return and Advantage computation
    returns = torch.zeros(batch_size, seq_len, device=device)
    G = torch.zeros(batch_size, device=device)
    for t in reversed(range(seq_len)):
        mask_t = (t <= stop_steps)
        G = torch.where(mask_t, rewards[:, t] + GAMMA_DISCOUNT * G, torch.zeros(batch_size, device=device))
        returns[:, t] = G
        
    advantages = returns - values.squeeze(-1)
    
    # Vectorized classification metrics
    final_actions = actions_taken[torch.arange(batch_size, device=device), stop_steps]
    
    corrects = torch.where(
        ((labels == 0.0) & (final_actions == 1)) | ((labels == 1.0) & (final_actions == 2)),
        torch.tensor(1.0, device=device),
        torch.tensor(0.0, device=device)
    )
    
    false_positives = torch.where(
        (labels == 0.0) & (final_actions == 2),
        torch.tensor(1.0, device=device),
        torch.tensor(0.0, device=device)
    )
    
    false_negatives = torch.where(
        (labels == 1.0) & (final_actions == 1),
        torch.tensor(1.0, device=device),
        torch.tensor(0.0, device=device)
    )
    
    ransomware_mask = (labels == 1.0)
    benign_mask = (labels == 0.0)
    
    avg_stop_ransomware = stop_steps[ransomware_mask].float().mean().item() if ransomware_mask.any() else 0.0
    avg_stop_benign = stop_steps[benign_mask].float().mean().item() if benign_mask.any() else 0.0
    
    # Vectorized ransomware damage
    arange = torch.arange(seq_len, device=device).unsqueeze(0)
    mask = arange <= stop_steps.unsqueeze(1)
    sample_damages = (damages * mask.float()).sum(dim=1)
    
    total_ransomware_damage = sample_damages[ransomware_mask].mean().item() if ransomware_mask.any() else 0.0
    
    return {
        "log_probs": log_probs,
        "values": values,
        "returns": returns,
        "advantages": advantages,
        "entropies": entropies,
        "stop_steps": stop_steps,
        "rewards": rewards,
        "corrects": corrects,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "avg_stop_ransomware": avg_stop_ransomware,
        "avg_stop_benign": avg_stop_benign,
        "avg_ransomware_damage": total_ransomware_damage
    }

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    
    epoch_loss = 0.0
    epoch_policy_loss = 0.0
    epoch_value_loss = 0.0
    epoch_entropy_loss = 0.0
    
    epoch_reward = 0.0
    epoch_acc = 0.0
    
    all_stop_r = []
    all_stop_b = []
    all_dmg = []
    batch_rewards = []
    batch_policy_losses = []
    batch_value_losses = []
    
    for batch in dataloader:
        optimizer.zero_grad()
        
        results = run_episode_batch(model, batch, device, train_mode=True)
        
        log_probs = results["log_probs"]      # [batch_size, seq_len]
        values = results["values"]            # [batch_size, seq_len, 1]
        returns = results["returns"]          # [batch_size, seq_len]
        advantages = results["advantages"]    # [batch_size, seq_len]
        entropies = results["entropies"]      # [batch_size, seq_len]
        stop_steps = results["stop_steps"]    # [batch_size]
        
        batch_size, seq_len = log_probs.shape
        
        # Create mask of active steps (steps <= stop_steps)
        arange = torch.arange(seq_len, device=device).unsqueeze(0) # [1, seq_len]
        mask = (arange <= stop_steps.unsqueeze(1)).float()          # [batch_size, seq_len]
        
        total_steps = mask.sum().item()
        
        # Vectorized losses
        policy_loss = (-log_probs * advantages * mask).sum() / total_steps
        value_loss = (F.mse_loss(values.squeeze(-1), returns, reduction="none") * mask).sum() / total_steps
        entropy_loss = (-entropies * mask).sum() / total_steps
        
        loss = policy_loss + VALUE_COEF * value_loss + ENTROPY_COEF * entropy_loss
        
        loss.backward()
        # Gradient clipping for stability
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        batch_reward_val = results["rewards"].sum(dim=1).mean().item()
        epoch_loss += loss.item()
        epoch_policy_loss += policy_loss.item()
        epoch_value_loss += value_loss.item()
        epoch_entropy_loss += entropy_loss.item()
        epoch_reward += batch_reward_val
        epoch_acc += results["corrects"].mean().item()
        batch_rewards.append(batch_reward_val)
        batch_policy_losses.append(policy_loss.item())
        batch_value_losses.append(value_loss.item())
        all_stop_r.append(results["avg_stop_ransomware"])
        all_stop_b.append(results["avg_stop_benign"])
        all_dmg.append(results["avg_ransomware_damage"])
        
    num_batches = len(dataloader)
    return {
        "loss": epoch_loss / num_batches,
        "policy_loss": epoch_policy_loss / num_batches,
        "value_loss": epoch_value_loss / num_batches,
        "entropy_loss": epoch_entropy_loss / num_batches,
        "reward": epoch_reward / num_batches,
        "reward_variance": float(np.var(batch_rewards)),
        "actor_loss_variance": float(np.var(batch_policy_losses)),
        "critic_loss_variance": float(np.var(batch_value_losses)),
        "accuracy": epoch_acc / num_batches,
        "avg_stop_ransomware": np.mean(all_stop_r),
        "avg_stop_benign": np.mean(all_stop_b),
        "avg_ransomware_damage": np.mean(all_dmg)
    }

def evaluate_model(model, dataloader, device):
    model.eval()
    
    epoch_reward = 0.0
    epoch_acc = 0.0
    total_fp = 0
    total_fn = 0
    total_samples = 0
    
    all_stop_r = []
    all_stop_b = []
    all_dmg = []
    
    with torch.no_grad():
        for batch in dataloader:
            results = run_episode_batch(model, batch, device, train_mode=False)
            
            epoch_reward += results["rewards"].sum(dim=1).mean().item()
            epoch_acc += results["corrects"].mean().item()
            
            total_fp += results["false_positives"].sum().item()
            total_fn += results["false_negatives"].sum().item()
            total_samples += batch["sequence"].shape[0]
            
            all_stop_r.append(results["avg_stop_ransomware"])
            all_stop_b.append(results["avg_stop_benign"])
            all_dmg.append(results["avg_ransomware_damage"])
            
    num_batches = len(dataloader)
    return {
        "reward": epoch_reward / num_batches,
        "accuracy": epoch_acc / num_batches,
        "false_positive_rate": total_fp / (total_samples / 2), # 50% benign
        "false_negative_rate": total_fn / (total_samples / 2), # 50% ransomware
        "avg_stop_ransomware": np.mean(all_stop_r),
        "avg_stop_benign": np.mean(all_stop_b),
        "avg_ransomware_damage": np.mean(all_dmg)
    }

def main():
    print("Initializing Data Loaders...")
    # Generate 2500 sample dataset (80% train / 20% test)
    train_loader, test_loader = get_dataloaders(
        num_samples=2500, 
        max_len=100, 
        batch_size=BATCH_SIZE, 
        train_split=0.8, 
        seed=42
    )
    
    # 16 is the vocabulary size defined in data_loader
    vocab_size = 16 
    
    print(f"Instantiating DESModel on device: {DEVICE}")
    model = DESModel(vocab_size=vocab_size, embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS)
    model.to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_acc = 0.0

    headers = ["Epoch", "Avg Reward", "Var(Reward)", "Avg Actor Loss",
               "Var(Actor)", "Avg Critic Loss", "Var(Critic)", "Train Acc", "Val Acc"]
    col_w = [6, 12, 12, 15, 12, 15, 12, 10, 10]

    def fmt_row(cells):
        return " | ".join(str(c).ljust(w) for c, w in zip(cells, col_w))

    sep = "-" * (sum(col_w) + 3 * (len(col_w) - 1))

    print("\nStarting DES-RL Training Loop...")
    print(sep)
    print(fmt_row(headers))
    print(sep)

    for epoch in range(1, NUM_EPOCHS + 1):
        train_metrics = train_epoch(model, train_loader, optimizer, DEVICE)
        val_metrics   = evaluate_model(model, test_loader, DEVICE)

        row = [
            epoch,
            f"{train_metrics['reward']:.4f}",
            f"{train_metrics['reward_variance']:.5f}",
            f"{train_metrics['policy_loss']:.4f}",
            f"{train_metrics['actor_loss_variance']:.5f}",
            f"{train_metrics['value_loss']:.4f}",
            f"{train_metrics['critic_loss_variance']:.5f}",
            f"{train_metrics['accuracy']:.2%}",
            f"{val_metrics['accuracy']:.2%}",
        ]
        print(fmt_row(row))

        if val_metrics["accuracy"] >= best_acc:
            best_acc = val_metrics["accuracy"]
            torch.save(model.state_dict(), "des_model.pt")

    print(sep)
    print(f"Training completed. Best Validation Accuracy: {best_acc:.2%}")
    print("Best model checkpoint saved to 'des_model.pt'.")

if __name__ == "__main__":
    main()
