import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from data_loader import get_dataloaders, API_TO_ID
from models import DESModel
from train import run_episode_batch, DEVICE, EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS

# Output paths
ARTIFACT_DIR = r"C:\Users\kiosh\.gemini\antigravity\brain\60a7b161-dc13-430b-8798-b0311f493e33"
WORKSPACE_DIR = r"c:\Users\kiosh\Downloads\aakash_2"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

def evaluate_static_baseline(model, dataloader, stop_step, device):
    """
    Evaluates a static baseline agent that is forced to stop and classify at a fixed step.
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_damages = []
    
    with torch.no_grad():
        for batch in dataloader:
            sequences = batch["sequence"].to(device)
            damages = batch["damage"].to(device)
            lengths = batch["length"].to(device)
            labels = batch["label"].to(device)
            
            batch_size, seq_len = sequences.shape
            policy_logits, _ = model(sequences)
            
            # For each sequence, determine prediction and damage at stop_step
            for i in range(batch_size):
                actual_len = lengths[i].item()
                # Determine the step we stop at (cannot exceed sequence length - 1)
                t_stop = min(stop_step, actual_len - 1)
                
                # Get policy logits at that step
                logits = policy_logits[i, t_stop, :]
                
                # Compare action 1 (Benign) vs action 2 (Ransomware)
                pred = 1 if logits[1] > logits[2] else 2
                pred_class = 0 if pred == 1 else 1 # 0: Benign, 1: Ransomware
                
                all_preds.append(pred_class)
                all_labels.append(labels[i].item())
                
                # Calculate damage up to t_stop if ransomware
                if labels[i].item() == 1.0:
                    dmg_acc = damages[i, :t_stop+1].sum().item()
                    all_damages.append(dmg_acc)
                    
    acc = accuracy_score(all_labels, all_preds)
    p, r, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="binary", zero_division=0)
    avg_dmg = np.mean(all_damages) if all_damages else 0.0
    
    return {
        "accuracy": acc,
        "precision": p,
        "recall": r,
        "f1": f1,
        "avg_damage": avg_dmg
    }

def main():
    print("Loading Test Dataset...")
    _, test_loader = get_dataloaders(
        num_samples=2500, 
        max_len=100, 
        batch_size=64, 
        train_split=0.8, 
        seed=42
    )
    
    vocab_size = 16
    model = DESModel(vocab_size=vocab_size, embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS)
    
    model_path = os.path.join(WORKSPACE_DIR, "des_model.pt")
    if not os.path.exists(model_path):
        print(f"Error: Trained model checkpoint not found at '{model_path}'. Please run train.py first.")
        return
        
    print(f"Loading trained weights from {model_path}...")
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    
    # 1. Evaluate DES-RL Agent
    print("\nEvaluating DES-RL Agent...")
    rl_preds = []
    rl_labels = []
    rl_stop_steps_r = []
    rl_stop_steps_b = []
    rl_damages = []
    
    with torch.no_grad():
        for batch in test_loader:
            results = run_episode_batch(model, batch, DEVICE, train_mode=False)
            
            stop_steps = results["stop_steps"]
            labels = batch["label"]
            damages = batch["damage"]
            
            for i in range(len(labels)):
                T = stop_steps[i].item()
                lbl = labels[i].item()
                
                # Get the action taken at the stop step
                # results["rewards"] logic: we can deduce prediction from which category reward is positive/negative
                # or evaluate logits at the stop step
                policy_logits, _ = model(batch["sequence"][i:i+1].to(DEVICE))
                logits = policy_logits[0, T, :]
                pred = 1 if logits[1] > logits[2] else 2
                pred_class = 0 if pred == 1 else 1
                
                rl_preds.append(pred_class)
                rl_labels.append(lbl)
                
                if lbl == 1.0:
                    rl_stop_steps_r.append(T)
                    rl_damages.append(damages[i, :T+1].sum().item())
                else:
                    rl_stop_steps_b.append(T)
                    
    rl_acc = accuracy_score(rl_labels, rl_preds)
    rl_p, rl_r, rl_f1, _ = precision_recall_fscore_support(rl_labels, rl_preds, average="binary", zero_division=0)
    rl_avg_stop_r = np.mean(rl_stop_steps_r)
    rl_avg_stop_b = np.mean(rl_stop_steps_b)
    rl_avg_dmg = np.mean(rl_damages)
    
    # 2. Evaluate Static Baselines
    print("\nEvaluating Static Baselines...")
    static_steps = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    static_results = []
    
    for step in static_steps:
        res = evaluate_static_baseline(model, test_loader, step, DEVICE)
        static_results.append(res)
        print(f"Static Stop at Step {step:2d} | Accuracy: {res['accuracy']:6.2%} | Ransomware Damage: {res['avg_damage']:5.2f}")
        
    # Print Markdown Table
    print("\n" + "="*80)
    print("EVALUATION COMPARISON TABLE")
    print("="*80)
    print(f"| Model / Strategy | Accuracy | Precision | Recall | F1-Score | Avg Steps (Ransom) | Avg Steps (Benign) | Avg Damage |")
    print(f"|---|---|---|---|---|---|---|---|")
    print(f"| **DES-RL (Ours)** | **{rl_acc:.2%}** | **{rl_p:.2%}** | **{rl_r:.2%}** | **{rl_f1:.2%}** | **{rl_avg_stop_r:.1f}** | **{rl_avg_stop_b:.1f}** | **{rl_avg_dmg:.2f}** |")
    for step, res in zip(static_steps, static_results):
        print(f"| Static Stop @ {step} | {res['accuracy']:.2%} | {res['precision']:.2%} | {res['recall']:.2%} | {res['f1']:.2%} | {step}.0 | {step}.0 | {res['avg_damage']:.2f} |")
    print("="*80)
    
    # 3. Plotting results
    print("\nGenerating charts...")
    plt.rcParams['figure.facecolor'] = '#ffffff'
    plt.rcParams['axes.facecolor'] = '#ffffff'
    plt.rcParams['axes.edgecolor'] = '#333333'
    plt.rcParams['text.color'] = '#000000'
    plt.rcParams['axes.labelcolor'] = '#000000'
    plt.rcParams['xtick.color'] = '#333333'
    plt.rcParams['ytick.color'] = '#333333'
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy Plot
    static_accs = [r["accuracy"] for r in static_results]
    ax1.plot(static_steps, static_accs, marker='o', linestyle='-', color='#2b5c8f', linewidth=2, label='Static Stopping Baseline')
    ax1.scatter([rl_avg_stop_r], [rl_acc], color='#d9534f', marker='*', s=250, zorder=5, label='DES-RL Agent (Ours)')
    ax1.set_xlabel('Steps Observed (Ransomware)', fontsize=12)
    ax1.set_ylabel('Classification Accuracy', fontsize=12)
    ax1.set_title('Accuracy vs. Observation Length', fontsize=14, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(fontsize=10)
    
    # Damage Plot
    static_dmgs = [r["avg_damage"] for r in static_results]
    ax2.plot(static_steps, static_dmgs, marker='o', linestyle='-', color='#d9534f', linewidth=2, label='Static Stopping Baseline')
    ax2.scatter([rl_avg_stop_r], [rl_avg_dmg], color='#2b5c8f', marker='*', s=250, zorder=5, label='DES-RL Agent (Ours)')
    ax2.set_xlabel('Steps Observed (Ransomware)', fontsize=12)
    ax2.set_ylabel('Average File Damage Incurred', fontsize=12)
    ax2.set_title('Ransomware Damage vs. Observation Length', fontsize=14, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(fontsize=10)
    
    plt.tight_layout()
    
    # Save the plots
    plot_name = "results_comparison.png"
    workspace_plot = os.path.join(WORKSPACE_DIR, plot_name)
    artifact_plot = os.path.join(ARTIFACT_DIR, plot_name)
    
    plt.savefig(workspace_plot, dpi=300)
    plt.savefig(artifact_plot, dpi=300)
    print(f"Saved comparison plot to {workspace_plot}")
    print(f"Saved comparison plot to {artifact_plot}")
    
if __name__ == "__main__":
    main()
