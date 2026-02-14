#!/usr/bin/env python3
# llm_optimize_full_model.py

import os
import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
from tqdm.notebook import tqdm
import requests
import json
import pandas as pd
import random
import math
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# 1. CONFIGURATION

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Base hyperparameters
NUM_EPOCHS = 40
LR = 1e-3
PATIENCE = 4
CLIP_NORM = 5.0
AUGMENT = True
NUM_WORKERS = 4
PIN_MEMORY = True

H5_PATH = "./processed/ptbxl_preprocessed_balanced.h5"
assert os.path.exists(H5_PATH), f"HDF5 file not found: {H5_PATH}"

# Check Ollama connection
try:
    response = requests.get("http://localhost:11434/api/tags", timeout=5)
    if response.status_code == 200:
        print("Ollama server is running")
        models = [m['name'] for m in response.json().get('models', [])]
        print(f"Available models: {models}")
    else:
        print("Ollama server issue")
except:
    print("Ollama not running. Start with: nohup ollama serve &")


# 2. DATASET

class ECGDataset(Dataset):
    def __init__(self, h5_path, augment=False):
        self.h5_path = h5_path
        self.augment = augment
        self.h5_file = None
        self.signals = None
        self.labels = None

    def _init_h5(self):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'r')
            self.signals = self.h5_file["signals"]
            self.labels = self.h5_file["labels"]

    def __len__(self):
        self._init_h5()
        return len(self.signals)

    def __getitem__(self, idx):
        self._init_h5()
        x = torch.tensor(self.signals[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        if self.augment:
            x = x + 0.01 * torch.randn_like(x)
        return x, y

# Create dataset and split
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

print(f"Dataset loaded: train={len(train_dataset)}, test={len(test_dataset)}")

# Load labels for pos_weight
with h5py.File(H5_PATH, 'r') as f:
    labels_array = f['labels'][:]
label_counts = labels_array.sum(axis=0)
N = labels_array.shape[0]
pos_weight = torch.tensor((N - label_counts) / (label_counts + 1e-8), dtype=torch.float32).to(device)


sample_x, sample_y = train_dataset[0]
n_leads = sample_x.shape[1]
n_classes = sample_y.shape[0]
print(f"Leads: {n_leads}, Classes: {n_classes}")

# 3. MODEL ARCHITECTURE 

class AdditiveAttention(nn.Module):
    def __init__(self, hidden_dim, attn_dim):
        super().__init__()
        self.W = nn.Linear(hidden_dim, attn_dim)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, H):
        score = torch.tanh(self.W(H))
        score = self.v(score).squeeze(-1)
        weights = torch.softmax(score, dim=1)
        context = torch.sum(H * weights.unsqueeze(-1), dim=1)
        return context, weights

class ECG_CNN_GRU_Attn(nn.Module):
    def __init__(self, n_leads=12, n_classes=5, cnn_channels=(64,128),
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
                          batch_first=True, bidirectional=True,
                          dropout=dropout if gru_layers>1 else 0.0)
        self.attn = AdditiveAttention(hidden_dim=gru_hidden*2, attn_dim=attn_dim)
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(gru_hidden*2, n_classes)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)  
        x = self.cnn(x)         
        x = x.permute(0, 2, 1)  
        rnn_out, _ = self.gru(x)  
        context, weights = self.attn(rnn_out)
        out = self.fc(context)
        return out, weights


# 4. LLM OPTIMIZER

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
            print(f" Ollama error: {e}")
            return ""

    def _build_prompt(self, search_space, previous_trials):
        """Build optimization prompt"""
        prompt = """You are an expert in deep learning hyperparameter optimization for ECG classification.

Task: Suggest optimal hyperparameters for the FULL CNN+GRU+Attention model on PTB-XL dataset.

Architecture: CNN (2 layers) → Bidirectional GRU → Additive Attention → FC

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

        prompt += """\n\nBased on the search space and previous results, suggest the NEXT set of hyperparameters.

Guidelines:
- Learning rate: balance convergence speed and stability (typical range: 1e-4 to 1e-3)
- Dropout: prevent overfitting but not too aggressive (0.2-0.4 is common)
- CNN channels: larger = more capacity but risk overfitting
- GRU hidden/layers: balance expressiveness and training time
- Attention dimension: typically 32-128
- Batch size: larger = faster but needs more memory

CRITICAL: Respond with ONLY a valid JSON object, nothing else:
{
    "learning_rate": <value>,
    "dropout": <value>,
    "batch_size": <value>,
    "cnn_ch1": <value>,
    "cnn_ch2": <value>,
    "gru_hidden": <value>,
    "gru_layers": <value>,
    "attn_dim": <value>,
    "num_epochs": <value>
}

Your suggestions:"""
        return prompt

    def _parse_response(self, response, search_space):
        """Parse LLM response"""
        try:
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
            print(f" Parsing error: {e}. Using random fallback.")

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


# 5. SEARCH SPACE FOR FULL MODEL

def get_full_model_search_space():
    """Define search space for full CNN+GRU+Attention model"""
    return {
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
            "choices": [128,256]
        },
        "cnn_ch1": {
            "type": "categorical",
            "choices": [32, 64, 128]
        },
        "cnn_ch2": {
            "type": "categorical",
            "choices": [64, 128, 256]
        },
        "gru_hidden": {
            "type": "categorical",
            "choices": [64, 128, 256]
        },
        "gru_layers": {
            "type": "int",
            "low": 1,
            "high": 3
        },
        "attn_dim": {
            "type": "categorical",
            "choices": [32, 64, 128]
        },
        "num_epochs": {
            "type": "categorical",
            "choices": [30, 40, 50]
        }
    }


# 6. TRAINING & EVALUATION

def evaluate(model, loader, threshold=0.5, verbose=False):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits, _ = model(X)
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

def train_model(model, model_name, train_loader, val_loader, criterion, optimizer,
                scheduler, num_epochs=50, patience=8, clip_norm=5.0, verbose=True):
    best_f1 = 0.0
    patience_ctr = 0
    best_path = f"best_{model_name}.pth"

    # --- REFINED RESUME LOGIC ---
    if os.path.exists(best_path):
        print(f"  Checkpoint found for {model_name}. Resuming training...")
        try:
            model.load_state_dict(torch.load(best_path, map_location=device))
            _, best_f1, _, _ = evaluate(model, val_loader)
            print(f" Resuming with starting F1: {best_f1:.4f}")
        except Exception as e:
            print(f" Could not load checkpoint: {e}. Starting fresh.")
    # ----------------------------

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
                    logits, _ = model(X)
                    loss = criterion(logits, y)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits, _ = model(X)
                loss = criterion(logits, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                optimizer.step()

            running_loss += loss.item() * X.size(0)
            n_samples += X.size(0)
            loop.set_postfix(train_loss=(running_loss / n_samples))

        train_loss = running_loss / n_samples
        val_acc, val_f1, _, _ = evaluate(model, val_loader, verbose=False)

        if verbose:
            print(f"{model_name} Epoch {epoch}: Loss={train_loss:.4f} | ValAcc={val_acc*100:.2f}% | ValF1={val_f1:.4f}")

        scheduler.step(val_f1)

        if val_f1 > best_f1 + 1e-5:
            best_f1 = val_f1
            patience_ctr = 0
            torch.save(model.state_dict(), best_path)
            if verbose:
                print(f"    New best (F1={best_f1:.4f}) saved")
        else:
            patience_ctr += 1
            if verbose:
                print(f"   No improvement. Patience {patience_ctr}/{patience}")
            if patience_ctr >= patience:
                if verbose:
                    print(f"   Early stopping triggered")
                break

    # Load best model
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))
    return model, best_f1



# 7. LLM OPTIMIZATION LOOP

def optimize_full_model_with_llm(n_trials=10):
    """
    Run LLM-based optimization for the full CNN+GRU+Attention model
    """

    optimizer = LLMOptimizer(model_name="llama3")
    search_space = get_full_model_search_space()

    print(f"\n{'='*70}")
    print(f"LLM OPTIMIZATION: FULL CNN+GRU+ATTENTION MODEL")
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

        try:
            # Extract parameters
            lr = params.get('learning_rate', LR)
            dropout = params.get('dropout', 0.3)
            batch_size = params.get('batch_size', BATCH_SIZE)
            ch1 = params.get('cnn_ch1', 64)
            ch2 = params.get('cnn_ch2', 128)
            gru_hidden = params.get('gru_hidden', 128)
            gru_layers = params.get('gru_layers', 2)
            attn_dim = params.get('attn_dim', 64)
            num_epochs_trial = params.get('num_epochs', NUM_EPOCHS)

            
            current_train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True,
                num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
            )
            current_test_loader = DataLoader(
                test_dataset, batch_size=batch_size, shuffle=False,
                num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
            )
          

            print(f"\n Building model...")

            # Build model
            model = ECG_CNN_GRU_Attn(
                n_leads=n_leads,
                n_classes=n_classes,
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

            print(f" Training for {num_epochs_trial} epochs...")
            model, best_f1 = train_model(
                model,
                f"full_model_trial{trial_num+1}",
                current_train_loader,  
                current_test_loader,   
                criterion,
                optim,
                scheduler,
                num_epochs=num_epochs_trial,
                patience=PATIENCE,
                clip_norm=CLIP_NORM,
                verbose=False
            )

            print(f" Evaluating...")
            acc, f1, preds, targets = evaluate(model, current_test_loader, verbose=False)

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
        print(f"\n No successful trials")
        return None

    print(f"\n{'='*70}")
    print(f"OPTIMIZATION COMPLETE")
    print(f"{'='*70}")
    print(f"\n Best hyperparameters:")
    for k, v in best['params'].items():
        print(f"   {k}: {v}")
    print(f"\n Best results:")
    print(f"   F1-Score:  {best['f1_score']:.4f}")
    print(f"   Accuracy:  {best['accuracy']:.4f}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"llm_optimization_full_model_{timestamp}.json"

    results_dict = {
        'model_name': 'full_cnn_gru_attn',
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


# 8. RUN OPTIMIZATION

if __name__ == "__main__":
    
    results = optimize_full_model_with_llm(n_trials=10)

    if results:
        print("\n Optimization complete! Best configuration found.")
        print("\n To use these hyperparameters, train your model with:")
        print(f"   {results['best_params']}")
    else:
        print("\n Optimization failed. Check logs above.")