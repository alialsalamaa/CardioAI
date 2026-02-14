# 1. Libraries
import os
import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
from tqdm.notebook import tqdm

## 2. Device & Hyperparameters 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

BATCH_SIZE = 64          # Increase for faster GPU utilization
NUM_EPOCHS = 50          # Fewer epochs, early stopping will handle the rest
LR = 1e-3
PATIENCE = 8
CLIP_NORM = 5.0
AUGMENT = True
NUM_WORKERS = 8          # Increase for HDF5 reading speed
PIN_MEMORY = True

H5_PATH = "./processed/ptbxl_preprocessed_balanced.h5"
assert os.path.exists(H5_PATH), f"HDF5 file not found: {H5_PATH}"
# 3. PyTorch Dataset
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

# ----------------------
# Create dataset and split
# ----------------------
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

print("Dataset loaded successfully!")
print("train:", len(train_dataset), "test:", len(test_dataset), "batch_size:", BATCH_SIZE)
# 5. 1D-CNN + GRU Model+attn
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
    def __init__(self, n_leads=12, n_classes=5, cnn_channels=(64,128), gru_hidden=128, gru_layers=2, dropout=0.3, attn_dim=64):
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
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        rnn_out, _ = self.gru(x)
        context, weights = self.attn(rnn_out)
        out = self.fc(context)
        return out, weights

n_leads = train_dataset[0][0].shape[1]
n_classes = train_dataset[0][1].shape[0]
model = ECG_CNN_GRU_Attn(n_leads=n_leads, n_classes=n_classes).to(device)
print(model)

import torch
from torchsummary import summary

class Wrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        # torchsummary gives (batch, channels, length), so we transpose
        x = x.permute(0, 2, 1)  # (B, L, C)
        return self.model(x)[0]

wrapped_model = Wrapper(model)
summary(wrapped_model, input_size=(12, 5000))
with h5py.File(H5_PATH, 'r') as f:
    labels_array = f['labels'][:]

label_counts = labels_array.sum(axis=0)
N = labels_array.shape[0]
pos_weight = torch.tensor((N - label_counts) / (label_counts + 1e-8), dtype=torch.float32).to(device)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
# 6. Training & Evaluation Functions
def evaluate(model, loader, threshold=0.5, verbose=True):
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

# ----------------------
# Training Loop (with Mixed Precision)
# ----------------------
scaler = torch.cuda.amp.GradScaler()  # AMP

def train_loop(model, train_loader, val_loader, num_epochs=50, patience=8, clip_norm=5.0):
    best_f1 = 0.0
    patience_ctr = 0
    best_path = "best_cnn_gru_attn.pth"

    for epoch in range(1, num_epochs+1):
        model.train()
        running_loss = 0.0
        n_samples = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}", leave=False)

        for X, y in pbar:
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                logits, _ = model(X)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * X.size(0)
            n_samples += X.size(0)
            pbar.set_postfix(train_loss=(running_loss / n_samples))

        train_loss = running_loss / n_samples

        val_acc, val_f1, _, _ = evaluate(model, val_loader, verbose=False)
        print(f"Epoch {epoch} -> TrainLoss: {train_loss:.4f} | Val Acc: {val_acc*100:.2f}% | Val F1: {val_f1:.4f}")

        scheduler.step(val_f1)

        if val_f1 > best_f1 + 1e-5:
            best_f1 = val_f1
            patience_ctr = 0
            torch.save(model.state_dict(), best_path)
            print(f"   New best model (F1={best_f1:.4f}) saved.")
        else:
            patience_ctr += 1
            print(f"   No improvement. patience {patience_ctr}/{patience}")
            if patience_ctr >= patience:
                print("   Early stopping triggered.")
                break

    model.load_state_dict(torch.load(best_path, map_location=device))
    return model
# 7. Train the Model
model = train_loop(model, train_loader, test_loader, num_epochs=NUM_EPOCHS, patience=PATIENCE, clip_norm=CLIP_NORM)
## 8. Final Evaluation
print("\nFinal evaluation on test set:")
acc, f1, preds, targets = evaluate(model, test_loader, verbose=True)
print(f"Final Acc: {acc*100:.2f}%  Final F1(samples): {f1:.4f}")

per_class_f1 = f1_score(targets, preds, average=None, zero_division=0)
for i, score in enumerate(per_class_f1):
    print(f"Class {i} F1: {score:.4f}")

# --- Visual Classification Report and Confusion Matrix ---
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Create classification report dataframe
report = classification_report(targets, preds, output_dict=True, zero_division=0)
report_df = pd.DataFrame(report).transpose()

print("\n=== Classification Report ===")
display(report_df.style.background_gradient(cmap='YlGnBu', axis=0))

# --- Confusion Matrix ---
cm = confusion_matrix(targets.argmax(axis=1), preds.argmax(axis=1))
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()
#  Final sanity check after training
print("\n===== Model Output Information (Post-Training Check) =====")

#  From model final layer
if hasattr(model.fc[-1], "out_features"):
    print(f"Number of labels (from model final layer): {model.fc[-1].out_features}")

#  From dataset labels
try:
    with h5py.File(H5_PATH, "r") as f:
        labels = f["labels"][:]
        print(f"Number of labels (from dataset): {labels.shape[1]}")
except Exception as e:
    print(f"Could not load labels from file: {e}")

#  From a test batch
sample_input, _ = next(iter(test_loader))
sample_input = sample_input.to(device)
with torch.no_grad():
    output, _ = model(sample_input)
print(f"Model output shape: {tuple(output.shape)}  -->  Predicted labels per sample: {output.shape[1]}")
#  Save the entire model (architecture + weights)
FULL_MODEL_PATH = "ecg_cnn_gru_attn_full.pth"
torch.save(model, FULL_MODEL_PATH)
print(f"✅ Full model (structure + weights) saved at: {FULL_MODEL_PATH}")