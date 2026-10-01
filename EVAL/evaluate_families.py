import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score, accuracy_score

from data_loader import API_TO_ID, API_VOCAB, generate_single_sequence
from models import DESModel
from train import run_episode_batch, DEVICE, EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS

# Paths
WORKSPACE_DIR = r"c:\Users\kiosh\Downloads\aakash_2"
ARTIFACT_DIR = r"C:\Users\kiosh\.gemini\antigravity-ide\brain\f56de692-b05e-4bea-bbdc-adf51abd297d"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# Style setup for Matplotlib (slate/modern aesthetic)
def setup_plot_style():
    plt.rcParams['figure.facecolor'] = '#ffffff'
    plt.rcParams['axes.facecolor'] = '#ffffff'
    plt.rcParams['axes.edgecolor'] = '#333333'
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.color'] = '#cccccc'
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['grid.alpha'] = 0.5
    plt.rcParams['text.color'] = '#000000'
    plt.rcParams['axes.labelcolor'] = '#000000'
    plt.rcParams['xtick.color'] = '#333333'
    plt.rcParams['ytick.color'] = '#333333'
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['font.size'] = 11

def generate_family_sequence(family, max_len=100):
    """
    Generates simulated API call sequences representing specific ransomware family profiles
    with realistic noise and evasion decoys.
    """
    seq = [API_TO_ID["<SOS>"]]
    damage = [0.0]
    decoy_apis = [3, 4, 6, 7, 9]

    if family == "Cerber":
        # Fast, loud encryption: minimal setup/recon, immediate execution
        setup_apis = [6, 3]
        recon_apis = [8, 12]
        encrypt_loop = [9, 10, 11]
        setup_len = np.random.randint(2, 5)
        recon_len = np.random.randint(2, 5)
        noise_p = 0.015
    elif family == "WannaCry":
        # Drops file, opens network connection to killswitch / C2
        setup_apis = [8, 10, 6, 14, 15]
        recon_apis = [8, 12, 13]
        encrypt_loop = [9, 10, 11]
        setup_len = np.random.randint(6, 12)
        recon_len = np.random.randint(4, 10)
        noise_p = 0.025
    elif family == "Locky":
        # Registry heavy configuration for persistence & file hijacking
        setup_apis = [3, 4, 5, 3, 5]
        recon_apis = [8, 12, 13, 12, 13]
        encrypt_loop = [9, 10, 11]
        setup_len = np.random.randint(8, 15)
        recon_len = np.random.randint(5, 10)
        noise_p = 0.035
    elif family == "CryptoLocker":
        # Network heavy C2 key exchange before encrypting
        setup_apis = [14, 15, 14, 15, 3, 5]
        recon_apis = [8, 12, 13]
        encrypt_loop = [9, 10, 11]
        setup_len = np.random.randint(10, 18)
        recon_len = np.random.randint(4, 8)
        noise_p = 0.020
    elif family == "Ryuk":
        # Long evasion/delay phase (GetSystemTime queries), anti-sandbox checks
        setup_apis = [7, 7, 3, 4, 6]
        recon_apis = [8, 12, 13, 8]
        encrypt_loop = [9, 10, 11]
        setup_len = np.random.randint(15, 25)
        recon_len = np.random.randint(8, 15)
        noise_p = 0.045
    else:
        raise ValueError(f"Unknown family: {family}")

    length = np.random.randint(70, 95)

    # 1. Setup Phase
    for _ in range(setup_len):
        seq.append(np.random.choice(setup_apis))
        damage.append(0.0)
        if np.random.random() < noise_p:
            seq.append(np.random.choice(decoy_apis))
            damage.append(0.0)

    # 2. Recon Phase
    for _ in range(recon_len):
        seq.append(np.random.choice(recon_apis))
        damage.append(0.0)
        if np.random.random() < noise_p:
            seq.append(np.random.choice([14, 3, 7]))
            damage.append(0.0)

    # 3. Encryption Phase
    encrypt_len = max(10, length - len(seq) - 1)
    for i in range(encrypt_len):
        api = encrypt_loop[i % len(encrypt_loop)]
        seq.append(api)
        if api in [10, 11]:
            damage.append(1.0 if np.random.random() > noise_p else 0.0)
        else:
            damage.append(0.0)
        if np.random.random() < noise_p:
            seq.append(np.random.choice(decoy_apis))
            damage.append(0.0)

    # EOS and Padding
    seq.append(API_TO_ID["<EOS>"])
    damage.append(0.0)

    # Global noise substitution
    special_ids = {API_TO_ID["<PAD>"], API_TO_ID["<SOS>"], API_TO_ID["<EOS>"]}
    for idx in range(1, len(seq) - 1):
        if seq[idx] not in special_ids and np.random.random() < noise_p:
            seq[idx] = int(np.random.randint(3, 16))

    actual_len = len(seq)
    if actual_len < max_len:
        padding_len = max_len - actual_len
        seq.extend([API_TO_ID["<PAD>"]] * padding_len)
        damage.extend([0.0] * padding_len)
    else:
        seq = seq[:max_len]
        damage = damage[:max_len]
        actual_len = max_len

    return np.array(seq, dtype=np.int64), np.array(damage, dtype=np.float32), actual_len


def get_family_dataset(family_name, num_samples=300, max_len=100, seed=42):
    np.random.seed(seed)
    sequences = []
    damages = []
    lengths = []
    labels = []
    
    for _ in range(num_samples):
        seq, dmg, length = generate_family_sequence(family_name, max_len)
        sequences.append(seq)
        damages.append(dmg)
        lengths.append(length)
        labels.append(1.0) # Ransomware
        
    return {
        "sequence": torch.tensor(np.array(sequences), dtype=torch.long),
        "damage": torch.tensor(np.array(damages), dtype=torch.float32),
        "length": torch.tensor(np.array(lengths), dtype=torch.long),
        "label": torch.tensor(np.array(labels), dtype=torch.float32)
    }

def get_benign_dataset(num_samples=300, max_len=100, seed=42):
    np.random.seed(seed + 100)
    sequences = []
    damages = []
    lengths = []
    labels = []
    
    for _ in range(num_samples):
        seq, dmg, length = generate_single_sequence(is_ransomware=False, max_len=max_len)
        sequences.append(seq)
        damages.append(dmg)
        lengths.append(length)
        labels.append(0.0) # Benign
        
    return {
        "sequence": torch.tensor(np.array(sequences), dtype=torch.long),
        "damage": torch.tensor(np.array(damages), dtype=torch.float32),
        "length": torch.tensor(np.array(lengths), dtype=torch.long),
        "label": torch.tensor(np.array(labels), dtype=torch.float32)
    }

def evaluate_dataset(model, dataset, device):
    model.eval()
    
    sequences = dataset["sequence"]
    damages = dataset["damage"]
    lengths = dataset["length"]
    labels = dataset["label"]
    
    batch_size = 64
    num_samples = sequences.shape[0]
    
    all_preds = []
    all_stop_steps = []
    all_damages_incurred = []
    
    with torch.no_grad():
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            batch = {
                "sequence": sequences[start_idx:end_idx].to(device),
                "damage": damages[start_idx:end_idx].to(device),
                "length": lengths[start_idx:end_idx].to(device),
                "label": labels[start_idx:end_idx].to(device)
            }
            
            results = run_episode_batch(model, batch, device, train_mode=False)
            stop_steps = results["stop_steps"]
            
            # Re-evaluate classes at the stopping step
            policy_logits, _ = model(batch["sequence"])
            
            for i in range(len(batch["label"])):
                T = stop_steps[i].item()
                logits = policy_logits[i, T, :]
                pred = 1 if logits[1] > logits[2] else 2
                pred_class = 0 if pred == 1 else 1 # 0: Benign, 1: Ransomware
                
                all_preds.append(pred_class)
                all_stop_steps.append(T)
                
                dmg_acc = batch["damage"][i, :T+1].sum().item()
                all_damages_incurred.append(dmg_acc)
                
    return {
        "preds": np.array(all_preds),
        "labels": labels.numpy(),
        "stop_steps": np.array(all_stop_steps),
        "damages_incurred": np.array(all_damages_incurred)
    }

def main():
    setup_plot_style()
    
    # 1. Instantiate & load model
    vocab_size = 16
    model = DESModel(vocab_size=vocab_size, embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS)
    model_path = os.path.join(WORKSPACE_DIR, "des_model.pt")
    
    if not os.path.exists(model_path):
        print(f"Error: Model checkpoint not found at {model_path}. Please train the model first.")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    
    # 2. Define Ransomware Families
    families = ["Cerber", "WannaCry", "Locky", "CryptoLocker", "Ryuk"]
    results = {}
    
    # Evaluate Benign first
    print("Evaluating Benign dataset...")
    benign_ds = get_benign_dataset(num_samples=300)
    benign_eval = evaluate_dataset(model, benign_ds, DEVICE)
    # For benign, prediction accuracy is True Negative Rate
    benign_acc = accuracy_score(benign_eval["labels"], benign_eval["preds"])
    results["Benign"] = {
        "accuracy": benign_acc,
        "avg_stop_step": np.mean(benign_eval["stop_steps"]),
        "avg_damage": 0.0,
        "stop_steps": benign_eval["stop_steps"],
        "damages_incurred": benign_eval["damages_incurred"]
    }
    
    # Evaluate Ransomware Families
    for fam in families:
        print(f"Evaluating {fam} dataset...")
        fam_ds = get_family_dataset(fam, num_samples=300)
        fam_eval = evaluate_dataset(model, fam_ds, DEVICE)
        
        # Recall is TPR (how many ransomware classified as ransomware)
        recall = recall_score(fam_eval["labels"], fam_eval["preds"], zero_division=0)
        
        results[fam] = {
            "accuracy": recall, # detection rate
            "avg_stop_step": np.mean(fam_eval["stop_steps"]),
            "avg_damage": np.mean(fam_eval["damages_incurred"]),
            "stop_steps": fam_eval["stop_steps"],
            "damages_incurred": fam_eval["damages_incurred"]
        }
        
    # Write summary table to stdout and file
    summary_path = os.path.join(WORKSPACE_DIR, "family_evaluation_summary.md")
    summary_lines = [
        "# Ransomware Family Evaluation Summary",
        "",
        "This table presents the performance of the DES-RL agent across different ransomware families.",
        "",
        "| Family / Category | Family Characterization | Detection Rate (TPR) | Avg stopping step | Avg Damage (Files) | Protection Efficiency |",
        "| --- | --- | --- | --- | --- | --- |"
    ]
    
    charac = {
        "Benign": "Normal system activity (registry read/write, file load)",
        "Cerber": "Ultra-fast execution, heavy write/delete loop, low latency setup",
        "WannaCry": "C2 network beaconing (HTTP), drops helper binaries before encrypting",
        "Locky": "Registry-heavy persistence modifications before target search",
        "CryptoLocker": "High network traffic (key exchange) before file traversal",
        "Ryuk": "Stealthy evasion phase, system time queries, long latency before encryption"
    }
    
    for cat, res in results.items():
        # Protection Efficiency = 1 - (avg_damage / max_possible_damage)
        # Assuming maximum possible damage is around 30-40 operations on average if not stopped
        max_dmg = 35.0
        efficiency = max(0.0, 1.0 - res["avg_damage"]/max_dmg) if cat != "Benign" else 1.0
        
        summary_lines.append(
            f"| **{cat}** | {charac[cat]} | {res['accuracy']:.2%} | {res['avg_stop_step']:.1f} | {res['avg_damage']:.2f} | {efficiency:.1%} |"
        )
        
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines))
    print(f"\nWritten evaluation summary to {summary_path}")
    print("\n".join(summary_lines))
    
    # 3. Create Plots
    # Chart 1: Detection Recall by Ransomware Family
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    fam_names = families
    recalls = [results[f]["accuracy"] * 100 for f in families]
    colors = ['#ef4444', '#e11d48', '#f59e0b', '#0ea5e9', '#a855f7']
    
    bars = ax1.barh(fam_names, recalls, color=colors, height=0.55, edgecolor='#38bdf8', linewidth=0.5)
    ax1.set_xlim(0, 105)
    ax1.set_xlabel('Detection Rate / Recall (%)', fontweight='bold', fontsize=12)
    ax1.set_title('DES-RL Detection Rate (Recall) by Ransomware Family', pad=20, fontsize=14, fontweight='bold')
    
    # Add values on bars
    for bar in bars:
        width = bar.get_width()
        ax1.text(width + 1.5, bar.get_y() + bar.get_height()/2, f'{width:.1f}%', 
                 va='center', ha='left', color='#000000', fontweight='bold', fontsize=10)
                 
    plt.tight_layout()
    plt.savefig(os.path.join(WORKSPACE_DIR, "family_detection_rate.png"), dpi=300)
    plt.savefig(os.path.join(ARTIFACT_DIR, "family_detection_rate.png"), dpi=300)
    plt.close()
    
    # Chart 2: Average Stopping Step vs Average Damage Incurred
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    avg_steps = [results[f]["avg_stop_step"] for f in families]
    avg_damages = [results[f]["avg_damage"] for f in families]
    
    # Add Benign for comparison on step count
    avg_steps.append(results["Benign"]["avg_stop_step"])
    avg_damages.append(0.0)
    all_labels = families + ["Benign"]
    all_colors = colors + ["#22c55e"]
    
    scatter = ax2.scatter(avg_steps, avg_damages, c=all_colors, s=[250]*5 + [150], edgecolors='#333333', zorder=5)
    
    # Label each point
    for i, txt in enumerate(all_labels):
        offset = (5, 5) if txt != "Cerber" else (-45, 5)
        ax2.annotate(txt, (avg_steps[i], avg_damages[i]), textcoords="offset points", 
                     xytext=offset, ha='center', color='#000000', fontweight='bold', fontsize=10)
                     
    ax2.set_xlabel('Average Observation Step Before Stopping', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Average File Damage Incurred', fontweight='bold', fontsize=12)
    ax2.set_title('Trade-Off: Detection Latency vs. File Damage by Family', pad=20, fontsize=14, fontweight='bold')
    ax2.set_ylim(-1, max(avg_damages) * 1.2)
    ax2.set_xlim(0, max(avg_steps) * 1.2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(WORKSPACE_DIR, "family_stopping_latency_vs_damage.png"), dpi=300)
    plt.savefig(os.path.join(ARTIFACT_DIR, "family_stopping_latency_vs_damage.png"), dpi=300)
    plt.close()

    # Chart 3: Cumulative Damage Curves (Step by Step)
    fig3, ax3 = plt.subplots(figsize=(9, 6))
    steps = np.arange(100)
    
    # We will simulate the damage growth curve for a single trace of each ransomware family
    # and compare the agent's stopping step against the curve
    for idx, fam in enumerate(families):
        # Retrieve random sequence of damage for the family
        fam_ds = get_family_dataset(fam, num_samples=1, seed=42)
        dmg_seq = fam_ds["damage"][0].numpy()
        cum_dmg = np.cumsum(dmg_seq)
        
        # Stop step for this sequence from agent
        # We can approximate with the average stop step of the family
        stop_t = int(round(results[fam]["avg_stop_step"]))
        
        # Plot full potential damage (dotted line)
        line = ax3.plot(steps, cum_dmg, linestyle=':', color=colors[idx], alpha=0.6)
        # Plot prevented damage (solid line up to stop_t, then flat/stopped)
        actual_dmg = np.copy(cum_dmg)
        actual_dmg[stop_t:] = cum_dmg[stop_t] # Agent kills process here, damage stops!
        
        ax3.plot(steps[:stop_t+1], actual_dmg[:stop_t+1], linestyle='-', color=colors[idx], linewidth=2.5, label=f"{fam} (Agent Blocked)")
        ax3.scatter(stop_t, cum_dmg[stop_t], color=colors[idx], marker='X', s=120, edgecolors='#333333', zorder=5)
        
    ax3.set_xlabel('Execution Step (Telemetry Sequence)', fontweight='bold', fontsize=12)
    ax3.set_ylabel('Cumulative File Damage Incurred', fontweight='bold', fontsize=12)
    ax3.set_title('Cumulative Damage Prevention by DES-RL Agent', pad=20, fontsize=14, fontweight='bold')
    ax3.legend(loc='upper left', framealpha=0.8, facecolor='#ffffff', edgecolor='#333333')
    ax3.set_xlim(0, 100)
    ax3.set_ylim(-1, 45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(WORKSPACE_DIR, "family_damage_prevention_curves.png"), dpi=300)
    plt.savefig(os.path.join(ARTIFACT_DIR, "family_damage_prevention_curves.png"), dpi=300)
    plt.close()

    print("Successfully generated all three charts in workspace and artifact folders:")
    print("1. family_detection_rate.png")
    print("2. family_stopping_latency_vs_damage.png")
    print("3. family_damage_prevention_curves.png")

if __name__ == "__main__":
    main()
