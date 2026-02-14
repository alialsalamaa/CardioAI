#!/usr/bin/env python3
# compare_models.py
# Progressive comparison: CNN-only, GRU-only, CNN+GRU, CNN+GRU+Attention
# Uses same dataset and hyperparams as your original setup.

import os
import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
from tqdm.notebook import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import warnings
warnings.filterwarnings("ignore")

# -----------------------------
# Config (same as your original)
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

BATCH_SIZE = 64
NUM_EPOCHS = 5
LR = 1e-3
PATIENCE = 8
CLIP_NORM = 5.0
AUGMENT = True
NUM_WORKERS = 8
PIN_MEMORY = True

H5_PATH = "./processed/ptbxl_preprocessed_balanced.h5"
assert os.path.exists(H5_PATH), f"HDF5 file not found: {H5_PATH}"

VERBOSE_PER_EPOCH = True  # prints per-epoch logs for each model

# -----------------------------
# Dataset
# -----------------------------
class ECGDataset(Dataset):
    def __init__(self, h5_path, augment=False):
        self.h5_path = h5_path
        self.augment = augment
        self.h5_file = None
        self.signals = None
        self.labels = None

    def _init_h5(self):
        if self.h5_file is None:
            # read-only
            self.h5_file = h5py.File(self.h5_path, 'r')
            self.signals = self.h5_file["signals"]
            self.labels = self.h5_file["labels"]

    def __len__(self):
        self._init_h5()
        return self.signals.shape[0]

    def __getitem__(self, idx):
        self._init_h5()
        # NOTE: dataset assumed to be shaped (samples, time, leads) OR (samples, leads, time)
        # Based on your earlier code we expect (time, leads) per sample so stored layout should be [samples, time, leads]
        x = torch.tensor(self.signals[idx], dtype=torch.float32)  # shape: (time, leads)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)   # one-hot vector (n_classes,)
        if self.augment:
            x = x + 0.01 * torch.randn_like(x)
        return x, y

# create train/test split
dataset = ECGDataset(H5_PATH, augment=False)
n = len(dataset)
train_n = int(0.8 * n)
test_n = n - train_n
train_dataset, test_dataset = random_split(dataset, [train_n, test_n])
train_dataset.dataset.augment = AUGMENT

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

print(f"Dataset ready -> train: {len(train_dataset)}  test: {len(test_dataset)}")

# load labels to compute pos_weight
with h5py.File(H5_PATH, 'r') as f:
    labels_array = f['labels'][:]
label_counts = labels_array.sum(axis=0)
N = labels_array.shape[0]
pos_weight = torch.tensor((N - label_counts) / (label_counts + 1e-8), dtype=torch.float32).to(device)

# infer shapes
sample_x, sample_y = train_dataset[0]
# sample_x expected shape: (time, leads)
if sample_x.dim() != 2:
    raise RuntimeError("Unexpected signal shape from dataset. Expected (time, leads).")
TIME_DIM, N_LEADS = sample_x.shape
N_CLASSES = sample_y.shape[0]
print(f"Time steps: {TIME_DIM}, Leads: {N_LEADS}, Classes: {N_CLASSES}")

# -----------------------------
# Model building blocks
# -----------------------------
class AdditiveAttention(nn.Module):
    def __init__(self, hidden_dim, attn_dim):
        super().__init__()
        self.W = nn.Linear(hidden_dim, attn_dim)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, H):
        # H: (B, T, hidden_dim)
        score = torch.tanh(self.W(H))        # (B, T, attn_dim)
        score = self.v(score).squeeze(-1)    # (B, T)
        weights = torch.softmax(score, dim=1) # (B, T)
        context = torch.sum(H * weights.unsqueeze(-1), dim=1) # (B, hidden_dim)
        return context, weights

# Model 1: CNN-only (two conv layers like original, global avg pool -> FC)
class Model_CNN(nn.Module):
    def __init__(self, n_leads=N_LEADS, n_classes=N_CLASSES, cnn_channels=(64,128), dropout=0.3):
        super().__init__()
        ch1, ch2 = cnn_channels
        self.cnn = nn.Sequential(
            nn.Conv1d(n_leads, ch1, kernel_size=7, padding=3),
            nn.BatchNorm1d(ch1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
            nn.Conv1d(ch1, ch2, kernel_size=5, padding=2),
            nn.BatchNorm1d(ch2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
        )
        self.fc = nn.Linear(ch2, n_classes)

    def forward(self, x):
        # x: (B, T, leads)
        x = x.permute(0, 2, 1)    # -> (B, leads, T)
        x = self.cnn(x)          # -> (B, ch2, T')
        x = x.mean(dim=2)        # global average pool -> (B, ch2)
        out = self.fc(x)         # (B, n_classes)
        return out

# Model 2: GRU-only (operates on raw x: (B, T, leads))
class Model_GRU(nn.Module):
    def __init__(self, n_leads=N_LEADS, n_classes=N_CLASSES, hidden_dim=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(input_size=n_leads, hidden_size=hidden_dim, num_layers=num_layers,
                          batch_first=True, bidirectional=True, dropout=dropout if num_layers>1 else 0.0)
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim*2, n_classes)
        )

    def forward(self, x):
        # x: (B, T, leads)
        rnn_out, _ = self.gru(x)     # (B, T, hidden*2)
        pooled = rnn_out.mean(dim=1) # (B, hidden*2)
        out = self.fc(pooled)
        return out

# Model 3: CNN + GRU (no attention)
class Model_CNN_GRU(nn.Module):
    def __init__(self, n_leads=N_LEADS, n_classes=N_CLASSES, cnn_channels=(64,128),
                 gru_hidden=128, gru_layers=2, dropout=0.3):
        super().__init__()
        ch1, ch2 = cnn_channels
        self.cnn = nn.Sequential(
            nn.Conv1d(n_leads, ch1, kernel_size=7, padding=3),
            nn.BatchNorm1d(ch1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
            nn.Conv1d(ch1, ch2, kernel_size=5, padding=2),
            nn.BatchNorm1d(ch2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
        )
        self.gru = nn.GRU(input_size=ch2, hidden_size=gru_hidden, num_layers=gru_layers,
                          batch_first=True, bidirectional=True, dropout=dropout if gru_layers>1 else 0.0)
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(gru_hidden*2, n_classes)
        )

    def forward(self, x):
        # x: (B, T, leads)
        x = x.permute(0, 2, 1)  # (B, leads, T)
        x = self.cnn(x)         # (B, ch2, T')
        x = x.permute(0, 2, 1)  # (B, T', ch2)
        rnn_out, _ = self.gru(x)  # (B, T', hidden*2)
        pooled = rnn_out.mean(dim=1) # (B, hidden*2)
        out = self.fc(pooled)
        return out

# Model 4: CNN + GRU + Attention (your original architecture)
class Model_CNN_GRU_Attn(nn.Module):
    def __init__(self, n_leads=N_LEADS, n_classes=N_CLASSES, cnn_channels=(64,128),
                 gru_hidden=128, gru_layers=2, dropout=0.3, attn_dim=64):
        super().__init__()
        ch1, ch2 = cnn_channels
        self.cnn = nn.Sequential(
            nn.Conv1d(n_leads, ch1, kernel_size=7, padding=3),
            nn.BatchNorm1d(ch1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
            nn.Conv1d(ch1, ch2, kernel_size=5, padding=2),
            nn.BatchNorm1d(ch2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
        )
        self.gru = nn.GRU(input_size=ch2, hidden_size=gru_hidden, num_layers=gru_layers,
                          batch_first=True, bidirectional=True, dropout=dropout if gru_layers>1 else 0.0)
        self.attn = AdditiveAttention(hidden_dim=gru_hidden*2, attn_dim=attn_dim)
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(gru_hidden*2, n_classes)
        )

    def forward(self, x):
        # x: (B, T, leads)
        x = x.permute(0, 2, 1)  # (B, leads, T)
        x = self.cnn(x)         # (B, ch2, T')
        x = x.permute(0, 2, 1)  # (B, T', ch2)
        rnn_out, _ = self.gru(x)  # (B, T', hidden*2)
        context, weights = self.attn(rnn_out)  # context: (B, hidden*2)
        out = self.fc(context)
        return out

# -----------------------------
# Training & Evaluation utilities
# -----------------------------
def evaluate(model, loader, threshold=0.5, verbose=False):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model(X)
            probs = torch.sigmoid(logits)
            preds = (probs > threshold).int()
            all_preds.append(preds)
            all_labels.append(y)

    all_preds = torch.cat(all_preds, dim=0).cpu().numpy()
    all_labels = torch.cat(all_labels, dim=0).cpu().numpy()

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="samples", zero_division=0)

    if verbose:
        print(classification_report(all_labels, all_preds, zero_division=0))
    return acc, f1, all_preds, all_labels

def train_model(model, model_name, train_loader, val_loader, criterion, optimizer, scheduler,
                num_epochs=NUM_EPOCHS, patience=PATIENCE, clip_norm=CLIP_NORM, verbose=True):
    best_f1 = 0.0
    patience_ctr = 0
    best_path = f"best_{model_name}.pth"
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    for epoch in range(1, num_epochs+1):
        model.train()
        running_loss = 0.0
        n_samples = 0
        loop = tqdm(train_loader, desc=f"{model_name} Epoch {epoch}/{num_epochs}", leave=False)
        for X, y in loop:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()

            if scaler is not None:
                with torch.cuda.amp.autocast():
                    logits = model(X)
                    loss = criterion(logits, y)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(X)
                loss = criterion(logits, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                optimizer.step()

            running_loss += loss.item() * X.size(0)
            n_samples += X.size(0)
            loop.set_postfix(train_loss=(running_loss / n_samples))

        train_loss = running_loss / n_samples
        val_acc, val_f1, _, _ = evaluate(model, val_loader, verbose=False)

        if VERBOSE_PER_EPOCH and verbose:
            print(f"{model_name} Epoch {epoch}: TrainLoss={train_loss:.4f} | ValAcc={val_acc*100:.2f}% | ValF1={val_f1:.4f}")

        scheduler.step(val_f1)

        if val_f1 > best_f1 + 1e-5:
            best_f1 = val_f1
            patience_ctr = 0
            torch.save(model.state_dict(), best_path)
            if verbose:
                print(f"   {model_name}: new best (F1={best_f1:.4f}) saved -> {best_path}")
        else:
            patience_ctr += 1
            if verbose:
                print(f"   {model_name}: no improvement. patience {patience_ctr}/{patience}")
            if patience_ctr >= patience:
                if verbose:
                    print(f"   {model_name}: early stopping.")
                break

    # load best
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))
    return model, best_f1

# -----------------------------
# Run experiments
# -----------------------------
def run_experiment():
    results = []

    model_constructors = [
        ("cnn", Model_CNN),
        ("gru", Model_GRU),
        ("cnn_gru", Model_CNN_GRU),
        ("cnn_gru_attn", Model_CNN_GRU_Attn),
    ]

    for name, ModelClass in model_constructors:
        print("\n" + "="*60)
        print(f"Training & evaluating model: {name}")
        print("="*60)
        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)

        model = ModelClass().to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

        model, best_f1 = train_model(model, name, train_loader, test_loader, criterion, optimizer, scheduler,
                                     num_epochs=NUM_EPOCHS, patience=PATIENCE, clip_norm=CLIP_NORM, verbose=True)

        print(f"\n{name} -> Best Val F1: {best_f1:.4f}")
        acc, f1, preds, targets = evaluate(model, test_loader, verbose=True)
        print(f"{name} final -> Acc: {acc*100:.2f}% | F1(samples): {f1:.4f}")

        # per-class f1
        try:
            per_class_f1 = f1_score(targets, preds, average=None, zero_division=0)
        except Exception:
            per_class_f1 = None

        # store result
        results.append({
            "model": name,
            "acc": acc,
            "f1": f1,
            "per_class_f1": per_class_f1,
            "preds": preds,
            "targets": targets
        })

        # show confusion matrix
        try:
            cm = confusion_matrix(targets.argmax(axis=1), preds.argmax(axis=1))
            plt.figure(figsize=(6,5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
            plt.title(f"Confusion Matrix ({name})")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.show()
        except Exception as e:
            print("Could not plot confusion matrix:", e)

    # Summary table
    summary = pd.DataFrame([{"Model": r["model"], "Acc": r["acc"], "F1_samples": r["f1"]} for r in results])
    summary = summary.sort_values("F1_samples", ascending=False).reset_index(drop=True)
    print("\n" + "="*60)
    print("Final summary (sorted by F1):")
    print(summary.to_string(index=False))
    # also save to CSV
    summary.to_csv("compare_models_summary.csv", index=False)
    print("\nSaved summary to compare_models_summary.csv")

if __name__ == "__main__":
    run_experiment()

import requests
import json
import pandas as pd
import numpy as np
import torch
import random
import math
from datetime import datetime

# Check Ollama connection
try:
    response = requests.get("http://localhost:11434/api/tags", timeout=5)
    if response.status_code == 200:
        print("✅ Ollama server is running")
        models = [m['name'] for m in response.json().get('models', [])]
        print(f"Available models: {models}")
    else:
        print("  Ollama server issue")
except:
    print(" Ollama not running. Start with: nohup ollama serve &")
class LLMOptimizer:
    """LLM-based hyperparameter optimizer using Ollama"""
    
    def __init__(self, model_name="llama3", temperature=0.7):
        self.model_name = model_name
        self.base_url = "http://localhost:11434"
        self.temperature = temperature
        self.trial_history = []
        
    def suggest_hyperparameters(self, search_space, previous_trials=None):
        """Get hyperparameter suggestions from LLM"""
        prompt = self._build_prompt(search_space, previous_trials)
        response = self._call_ollama(prompt)
        suggestions = self._parse_response(response, search_space)
        return suggestions
    
    def _call_ollama(self, prompt):
        """Call Ollama API"""
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "temperature": self.temperature
        }
        
        try:
            response = requests.post(url, json=payload, timeout=180)
            response.raise_for_status()
            return response.json().get("response", "")
        except Exception as e:
            print(f"  Ollama error: {e}")
            return ""
    
    def _build_prompt(self, search_space, previous_trials):
        """Build optimization prompt for ECG models"""
        prompt = """You are an expert in deep learning hyperparameter optimization for ECG classification.

Task: Suggest optimal hyperparameters for training an ECG classifier on the PTB-XL dataset.

Search Space:
"""
        for param, config in search_space.items():
            if config["type"] == "loguniform":
                prompt += f"\n- {param}: log-uniform in [{config['low']}, {config['high']}]"
            elif config["type"] == "uniform":
                prompt += f"\n- {param}: uniform in [{config['low']}, {config['high']}]"
            elif config["type"] == "categorical":
                prompt += f"\n- {param}: choose from {config['choices']}"
            elif config["type"] == "int":
                prompt += f"\n- {param}: integer in [{config['low']}, {config['high']}]"
        
        if previous_trials and len(previous_trials) > 0:
            prompt += "\n\nPrevious Trial Results (last 5):\n"
            for i, trial in enumerate(previous_trials[-5:], 1):
                prompt += f"\nTrial {i}:\n"
                prompt += f"  Parameters: {trial['params']}\n"
                prompt += f"  F1-Score: {trial.get('f1_score', 'N/A'):.4f}\n"
                prompt += f"  Accuracy: {trial.get('accuracy', 'N/A'):.4f}\n"
        
        prompt += """\n\nBased on the search space and previous results, suggest the NEXT set of hyperparameters to try.
Consider:
- Learning rate should balance convergence speed and stability
- Dropout prevents overfitting but too much hurts performance
- Larger models (more channels/units) may perform better but risk overfitting
- More epochs help but watch for early stopping

CRITICAL: Respond with ONLY a valid JSON object, nothing else:
{
    "learning_rate": <value>,
    "dropout": <value>,
    "batch_size": <value>,
    ...
}

Your suggestions:"""
        return prompt
    
    def _parse_response(self, response, search_space):
        """Parse LLM response and extract hyperparameters"""
        try:
            # Find JSON in response
            start = response.find('{')
            end = response.rfind('}') + 1
            
            if start != -1 and end > start:
                json_str = response[start:end]
                suggestions = json.loads(json_str)
                
                # Validate and clip to search space
                cleaned = {}
                for param, value in suggestions.items():
                    if param in search_space:
                        config = search_space[param]
                        
                        if config["type"] in ["uniform", "loguniform"]:
                            value = float(value)
                            value = np.clip(value, config["low"], config["high"])
                        elif config["type"] == "int":
                            value = int(value)
                            value = np.clip(value, config["low"], config["high"])
                        elif config["type"] == "categorical":
                            if value not in config["choices"]:
                                value = config["choices"][0]
                        
                        cleaned[param] = value
                
                return cleaned
        except Exception as e:
            print(f"  Parsing error: {e}. Using random fallback.")
        
        # Fallback: random sampling
        return self._random_sample(search_space)
    
    def _random_sample(self, search_space):
        """Random sampling fallback"""
        samples = {}
        for param, config in search_space.items():
            if config["type"] == "loguniform":
                log_low = math.log(config["low"])
                log_high = math.log(config["high"])
                samples[param] = math.exp(random.uniform(log_low, log_high))
            elif config["type"] == "uniform":
                samples[param] = random.uniform(config["low"], config["high"])
            elif config["type"] == "int":
                samples[param] = random.randint(config["low"], config["high"])
            elif config["type"] == "categorical":
                samples[param] = random.choice(config["choices"])
        return samples
    
    def record_trial(self, params, results):
        """Record trial results"""
        self.trial_history.append({
            "params": params,
            **results
        })
    
    def get_best_trial(self, metric="f1_score"):
        """Get best trial by metric"""
        if not self.trial_history:
            return None
        return max(self.trial_history, key=lambda x: x.get(metric, 0))

def get_search_space_for_model(model_type):
    """
    Define hyperparameter search spaces for each model
    
    Args:
        model_type: 'cnn', 'gru', 'cnn_gru', or 'cnn_gru_attn'
    """
    # Base hyperparameters (all models)
    search_space = {
        "learning_rate": {
            "type": "loguniform",
            "low": 1e-5,
            "high": 1e-2
        },
        "dropout": {
            "type": "uniform",
            "low": 0.1,
            "high": 0.6
        },
        "batch_size": {
            "type": "categorical",
            "choices": [32, 64, 128]
        },
        "num_epochs": {
            "type": "categorical",
            "choices": [3]          
        }
    }
    
    # CNN-specific parameters
    if model_type in ['cnn', 'cnn_gru', 'cnn_gru_attn']:
        search_space["cnn_ch1"] = {
            "type": "categorical",
            "choices": [32, 64, 128]
        }
        search_space["cnn_ch2"] = {
            "type": "categorical",
            "choices": [64, 128, 256]
        }
    
    # GRU-specific parameters
    if model_type in ['gru', 'cnn_gru', 'cnn_gru_attn']:
        search_space["gru_hidden"] = {
            "type": "categorical",
            "choices": [64, 128, 256]
        }
        search_space["gru_layers"] = {
            "type": "int",
            "low": 1,
            "high": 3
        }
    
    # Attention-specific parameters
    if model_type == 'cnn_gru_attn':
        search_space["attn_dim"] = {
            "type": "categorical",
            "choices": [32, 64, 128]
        }
    
    return search_space

def optimize_model_with_llm(
    model_class,
    model_name,
    search_space,
    n_trials=10,
    verbose=True
):
    """
    Optimize a model using LLM-based hyperparameter search
    
    Args:
        model_class: Your model class (Model_CNN, Model_GRU, etc.)
        model_name: String name ('cnn', 'gru', 'cnn_gru', 'cnn_gru_attn')
        search_space: Hyperparameter search space
        n_trials: Number of optimization trials
        verbose: Print detailed logs
    
    Returns:
        dict with best_params, best_results, all_trials
    """
    
    optimizer = LLMOptimizer(model_name="llama3")
    
    print(f"\n{'='*70}")
    print(f"OPTIMIZING {model_name.upper()} MODEL WITH LLM")
    print(f"{'='*70}")
    print(f"Trials: {n_trials}")
    print(f"Search space: {list(search_space.keys())}\n")
    
    for trial_num in range(n_trials):
        print(f"\n{'─'*70}")
        print(f"Trial {trial_num + 1}/{n_trials}")
        print(f"{'─'*70}")
        
        # Get LLM suggestions
        params = optimizer.suggest_hyperparameters(
            search_space=search_space,
            previous_trials=optimizer.trial_history
        )
        
        print(f"\n LLM Suggested Hyperparameters:")
        for k, v in params.items():
            print(f"   {k}: {v}")
        
        # Build model with suggested hyperparameters
        try:
            print(f"\n🔨 Building model...")
            
            # Extract parameters
            lr = params.get('learning_rate', LR)
            dropout = params.get('dropout', 0.3)
            batch_size = params.get('batch_size', BATCH_SIZE)
            num_epochs_trial = params.get('num_epochs', NUM_EPOCHS)
            
            # Model-specific parameters
            if model_name == 'cnn':
                ch1 = params.get('cnn_ch1', 64)
                ch2 = params.get('cnn_ch2', 128)
                model = model_class(
                    n_leads=N_LEADS,
                    n_classes=N_CLASSES,
                    cnn_channels=(ch1, ch2),
                    dropout=dropout
                ).to(device)
                
            elif model_name == 'gru':
                gru_hidden = params.get('gru_hidden', 128)
                gru_layers = params.get('gru_layers', 2)
                model = model_class(
                    n_leads=N_LEADS,
                    n_classes=N_CLASSES,
                    hidden_dim=gru_hidden,
                    num_layers=gru_layers,
                    dropout=dropout
                ).to(device)
                
            elif model_name == 'cnn_gru':
                ch1 = params.get('cnn_ch1', 64)
                ch2 = params.get('cnn_ch2', 128)
                gru_hidden = params.get('gru_hidden', 128)
                gru_layers = params.get('gru_layers', 2)
                model = model_class(
                    n_leads=N_LEADS,
                    n_classes=N_CLASSES,
                    cnn_channels=(ch1, ch2),
                    gru_hidden=gru_hidden,
                    gru_layers=gru_layers,
                    dropout=dropout
                ).to(device)
                
            elif model_name == 'cnn_gru_attn':
                ch1 = params.get('cnn_ch1', 64)
                ch2 = params.get('cnn_ch2', 128)
                gru_hidden = params.get('gru_hidden', 128)
                gru_layers = params.get('gru_layers', 2)
                attn_dim = params.get('attn_dim', 64)
                model = model_class(
                    n_leads=N_LEADS,
                    n_classes=N_CLASSES,
                    cnn_channels=(ch1, ch2),
                    gru_hidden=gru_hidden,
                    gru_layers=gru_layers,
                    dropout=dropout,
                    attn_dim=attn_dim
                ).to(device)
            
            # Setup training
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optim, mode='max', factor=0.5, patience=3
            )
            
            # Train model using your existing train_model function
            print(f" Training for {num_epochs_trial} epochs...")
            model, best_f1 = train_model(
                model, 
                f"{model_name}_trial{trial_num+1}",
                train_loader,
                test_loader,
                criterion,
                optim,
                scheduler,
                num_epochs=num_epochs_trial,
                patience=PATIENCE,
                clip_norm=CLIP_NORM,
                verbose=False  # Suppress per-epoch output
            )
            
            # Evaluate on test set
            print(f" Evaluating...")
            acc, f1, preds, targets = evaluate(model, test_loader, verbose=False)
            
            results = {
                'accuracy': acc,
                'f1_score': f1
            }
            
            print(f"\n Results:")
            print(f"   Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
            print(f"   F1-Score:  {f1:.4f}")
            
            # Record trial
            optimizer.record_trial(params, results)
            
        except Exception as e:
            print(f"\n Trial failed: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Get best trial
    best = optimizer.get_best_trial(metric='f1_score')
    
    if best is None:
        print(f"\n No successful trials for {model_name}")
        return None
    
    print(f"\n{'='*70}")
    print(f"OPTIMIZATION COMPLETE - {model_name.upper()}")
    print(f"{'='*70}")
    print(f"\n Best hyperparameters:")
    for k, v in best['params'].items():
        print(f"   {k}: {v}")
    print(f"\n Best results:")
    print(f"   F1-Score:  {best['f1_score']:.4f}")
    print(f"   Accuracy:  {best['accuracy']:.4f}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"llm_optimization_{model_name}_{timestamp}.json"
    
    results_dict = {
        'model_name': model_name,
        'best_params': best['params'],
        'best_results': {k: v for k, v in best.items() if k != 'params'},
        'all_trials': optimizer.trial_history,
        'search_space': search_space,
        'n_trials': n_trials
    }
    
    with open(results_file, 'w') as f:
        json.dump(results_dict, f, indent=2, default=str)
    
    print(f"\n Results saved to: {results_file}")
    
    return results_dict
def run_llm_optimization_experiment(n_trials_per_model=3):
    """
    Run LLM-based optimization for all 4 models
    """
    
    models_to_optimize = [
        ('cnn', Model_CNN),
        ('gru', Model_GRU),
        ('cnn_gru', Model_CNN_GRU),
        ('cnn_gru_attn', Model_CNN_GRU_Attn),
    ]
    
    all_results = {}
    
    for model_name, model_class in models_to_optimize:
        print(f"\n\n{'#'*70}")
        print(f"# STARTING LLM OPTIMIZATION: {model_name.upper()}")
        print(f"{'#'*70}\n")
        
        # Get search space for this model
        search_space = get_search_space_for_model(model_name)
        
        # Run optimization
        results = optimize_model_with_llm(
            model_class=model_class,
            model_name=model_name,
            search_space=search_space,
            n_trials=n_trials_per_model,
            verbose=True
        )
        
        if results:
            all_results[model_name] = results
    
    # Create comparison DataFrame
    print(f"\n\n{'='*70}")
    print("FINAL LLM OPTIMIZATION COMPARISON")
    print(f"{'='*70}\n")
    
    comparison_data = []
    for model_name, results in all_results.items():
        comparison_data.append({
            'Model': model_name.upper(),
            'Best F1-Score': results['best_results']['f1_score'],
            'Best Accuracy': results['best_results']['accuracy'],
            'Trials Completed': len(results['all_trials'])
        })
    
    df_comparison = pd.DataFrame(comparison_data)
    df_comparison = df_comparison.sort_values('Best F1-Score', ascending=False)
    
    print(df_comparison.to_string(index=False))
    
    # Save comparison
    comparison_file = f"llm_optimization_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df_comparison.to_csv(comparison_file, index=False)
    print(f"\n Comparison saved to: {comparison_file}")
    
    return all_results, df_comparison
# Run LLM optimization for all models
# Start with 5 trials per model for testing, increase to 10-15 for final results
llm_results, llm_comparison = run_llm_optimization_experiment(n_trials_per_model=3)

print("\n✨ LLM optimization complete!")
def compare_llm_vs_baseline(llm_results, baseline_results_file="compare_models_summary.csv"):
    """
    Compare LLM-optimized results with baseline results
    """
    
    print(f"\n{'='*70}")
    print("COMPARISON: LLM-OPTIMIZED vs BASELINE")
    print(f"{'='*70}\n")
    
    # Load baseline results if they exist
    try:
        baseline_df = pd.read_csv(baseline_results_file)
        print(" Baseline Results:")
        print(baseline_df.to_string(index=False))
        print()
    except:
        print("  No baseline results file found. Run the baseline experiment first.\n")
        baseline_df = None
    
    # LLM results
    llm_data = []
    for model_name, results in llm_results.items():
        llm_data.append({
            'Model': model_name.upper(),
            'F1_samples': results['best_results']['f1_score'],
            'Acc': results['best_results']['accuracy']
        })
    
    llm_df = pd.DataFrame(llm_data)
    print(" LLM-Optimized Results:")
    print(llm_df.to_string(index=False))
    
    # Combined comparison if baseline exists
    if baseline_df is not None:
        print(f"\n{'─'*70}")
        print("IMPROVEMENT ANALYSIS")
        print(f"{'─'*70}\n")
        
        for _, row in llm_df.iterrows():
            model = row['Model']
            llm_f1 = row['F1_samples']
            
            baseline_row = baseline_df[baseline_df['Model'].str.upper() == model]
            if not baseline_row.empty:
                baseline_f1 = baseline_row['F1_samples'].values[0]
                improvement = ((llm_f1 - baseline_f1) / baseline_f1) * 100
                
                print(f"{model}:")
                print(f"  Baseline F1:  {baseline_f1:.4f}")
                print(f"  LLM F1:       {llm_f1:.4f}")
                print(f"  Improvement:  {improvement:+.2f}%")
                print()

# Run comparison
compare_llm_vs_baseline(llm_results)

def visualize_best_hyperparameters(llm_results):
    """
    Visualize the best hyperparameters found for each model
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    model_names = list(llm_results.keys())
    
    for idx, model_name in enumerate(model_names):
        results = llm_results[model_name]
        best_params = results['best_params']
        
        ax = axes[idx]
        
        # Extract key parameters
        params_to_plot = {}
        for k, v in best_params.items():
            if k in ['learning_rate', 'dropout', 'batch_size', 'num_epochs',
                     'cnn_ch1', 'cnn_ch2', 'gru_hidden', 'gru_layers', 'attn_dim']:
                params_to_plot[k] = v
        
        # Plot
        ax.barh(list(params_to_plot.keys()), list(params_to_plot.values()))
        ax.set_xlabel('Value')
        ax.set_title(f'{model_name.upper()} - Best Hyperparameters')
        ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('llm_best_hyperparameters.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(" Visualization saved to: llm_best_hyperparameters.png")

# Run visualization
visualize_best_hyperparameters(llm_results)

print("\n🎉 All done! Check the generated files for detailed results.")