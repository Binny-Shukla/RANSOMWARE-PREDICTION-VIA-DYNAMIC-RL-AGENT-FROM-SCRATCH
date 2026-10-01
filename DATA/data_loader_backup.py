import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Vocabulary definition for API calls
API_VOCAB = {
    0: "<PAD>",
    1: "<SOS>",
    2: "<EOS>",
    3: "RegOpenKey",
    4: "RegQueryValue",
    5: "RegSetValue",
    6: "LdrLoadDll",
    7: "GetSystemTime",
    8: "CreateFile",
    9: "ReadFile",
    10: "WriteFile",
    11: "DeleteFile",
    12: "FindFirstFile",
    13: "FindNextFile",
    14: "InternetOpen",
    15: "HttpSendRequest"
}

API_TO_ID = {v: k for k, v in API_VOCAB.items()}
VOCAB_SIZE = len(API_VOCAB)

def generate_single_sequence(is_ransomware, max_len=100):
    """
    Generates a single API call sequence along with its step-by-step data damage.
    Supports multiple ransomware families and benign processes.
    """
    seq = [API_TO_ID["<SOS>"]]
    damage = [0.0]
    
    if not is_ransomware:
        # Benign process simulation
        low_len = max(5, min(40, max_len - 10))
        high_len = max(10, max_len - 5)
        length = np.random.randint(low_len, high_len) if high_len > low_len else low_len
        
        # Normal process flow: opening registry, loading library, reading files, etc.
        benign_apis = [3, 4, 5, 6, 7, 8, 9, 6, 3, 4, 8, 9] # registry, read, load library
        for _ in range(length):
            api = np.random.choice(benign_apis)
            seq.append(api)
            damage.append(0.0)
    else:
        # Choose a ransomware family randomly to represent diverse behaviors
        family = np.random.choice(["WannaCry", "Locky", "Ryuk", "Cerber", "CryptoLocker", "Generic"])
        
        if family == "WannaCry":
            setup_apis = [8, 10, 6, 14, 15]
            recon_apis = [8, 12, 13]
            encrypt_loop = [9, 10, 11]
            setup_len = np.random.randint(6, 12)
            recon_len = np.random.randint(4, 10)
        elif family == "Locky":
            setup_apis = [3, 4, 5, 3, 5]
            recon_apis = [8, 12, 13, 12, 13]
            encrypt_loop = [9, 10, 11]
            setup_len = np.random.randint(8, 15)
            recon_len = np.random.randint(5, 10)
        elif family == "Ryuk":
            setup_apis = [7, 7, 3, 4, 6]
            recon_apis = [8, 12, 13, 8]
            encrypt_loop = [9, 10, 11]
            setup_len = np.random.randint(15, 25)
            recon_len = np.random.randint(8, 15)
        elif family == "Cerber":
            setup_apis = [6, 3]
            recon_apis = [8, 12]
            encrypt_loop = [9, 10, 11]
            setup_len = np.random.randint(2, 5)
            recon_len = np.random.randint(2, 5)
        elif family == "CryptoLocker":
            setup_apis = [14, 15, 14, 15, 3, 5]
            recon_apis = [8, 12, 13]
            encrypt_loop = [9, 10, 11]
            setup_len = np.random.randint(10, 18)
            recon_len = np.random.randint(4, 8)
        else: # Generic
            setup_apis = [3, 4, 6, 7, 8]
            recon_apis = [8, 12, 13, 12, 13]
            if np.random.rand() < 0.1:
                seq.append(np.random.choice([10, 11, 14, 15]))
            else:
                seq.append(np.random.choice(common_apis))
            damage.append(0.0)
    else:
        # Ransomware: Mix of common and sensitive APIs, plus specific loop patterns
        # Overlap noise: Inject some benign-like behavior
        setup_len = np.random.randint(10, 20)
        for _ in range(setup_len):
            seq.append(np.random.choice(common_apis + [14, 15]))
            damage.append(0.0)
            
        # Heavy payload
        payload_len = np.random.randint(30, 60)
        for _ in range(payload_len):
            api = np.random.choice([8, 9, 10, 11, 12, 13])
            seq.append(api)
            # Noise: encryption activity is high, but not 100% correlate
            is_damage = (api in [10, 11] and np.random.rand() > 0.1)
            damage.append(1.0 if is_damage else 0.0)
            
    # Append EOS
    seq.append(API_TO_ID["<EOS>"])
    damage.append(0.0)
    
    # Pad or truncate
    if len(seq) > max_len:
        seq = seq[:max_len]
        damage = damage[:max_len]
        actual_len = max_len
    else:
        actual_len = len(seq)
        padding = max_len - actual_len
        seq.extend([API_TO_ID["<PAD>"]] * padding)
        damage.extend([0.0] * padding)
        
    return np.array(seq, dtype=np.int64), np.array(damage, dtype=np.float32), actual_len

class RansomwareDataset(Dataset):
    def __init__(self, num_samples=2000, max_len=100, split_seed=42):
        super().__init__()
        self.num_samples = num_samples
        self.max_len = max_len
        
        np.random.seed(split_seed)
        
        self.sequences = []
        self.damages = []
        self.lengths = []
        self.labels = []
        
        for i in range(num_samples):
            # 50% benign, 50% ransomware
            is_ransomware = (i % 2 == 1)
            seq, damage, length = generate_single_sequence(is_ransomware, max_len)
            self.sequences.append(seq)
            self.damages.append(damage)
            self.lengths.append(length)
            self.labels.append(1 if is_ransomware else 0)
            
        self.sequences = torch.tensor(np.array(self.sequences), dtype=torch.long)
        self.damages = torch.tensor(np.array(self.damages), dtype=torch.float32)
        self.lengths = torch.tensor(np.array(self.lengths), dtype=torch.long)
        self.labels = torch.tensor(np.array(self.labels), dtype=torch.float32)
        
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        return {
            "sequence": self.sequences[idx],
            "damage": self.damages[idx],
            "length": self.lengths[idx],
            "label": self.labels[idx]
        }

def get_dataloaders(num_samples=2000, max_len=100, batch_size=64, train_split=0.8, seed=42):
    """
    Returns train and test dataloaders.
    """
    dataset = RansomwareDataset(num_samples=num_samples, max_len=max_len, split_seed=seed)
    
    train_size = int(train_split * len(dataset))
    test_size = len(dataset) - train_size
    
    generator = torch.Generator().manual_seed(seed)
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size], generator=generator
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

if __name__ == "__main__":
    train_loader, test_loader = get_dataloaders(num_samples=10, max_len=50, batch_size=2)
    for batch in train_loader:
        print("Sequence batch shape:", batch["sequence"].shape)
        print("Damage batch shape:", batch["damage"].shape)
        print("Length batch shape:", batch["length"].shape)
        print("Label batch shape:", batch["label"].shape)
        
        # Print first sample details
        first_seq = batch["sequence"][0].tolist()
        first_label = batch["label"][0].item()
        print(f"\nLabel: {'Ransomware' if first_label == 1.0 else 'Benign'}")
        print("API Sequence:")
        print([API_VOCAB[api] for api in first_seq if api != API_TO_ID["<PAD>"]])
        print("Damage sequence:", batch["damage"][0].tolist()[:batch["length"][0].item()])
        break
