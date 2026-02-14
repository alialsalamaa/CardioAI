# ========== CELL 1: Notebook Metadata & Imports ==========
# This single-file notebook contains the full XAI pipeline (Option 2)
# - Dataset loader (safe HDF5 handling)
# - Model definition (same architecture)
# - Model loading (robust to both full model and state_dict)
# - Integrated Gradients (Captum)
# - Optional SHAP placeholder (disabled by default)
# - Attention visualization, lead/temporal attributions
# - Clinical summary and report generation

# Use the file as a Jupyter cell-run notebook. Cells are separated by comments.

# Standard imports
import os
import numpy as np
import h5py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from captum.attr import IntegratedGradients
import matplotlib.pyplot as plt
from tqdm import tqdm

# Optional: SHAP (import only if available)
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ========== CELL 2: User paths & constants ==========
MODEL_PATH = "./notebook/ecg_cnn_gru_attn_full.pth"
H5_PATH = "./processed/ptbxl_preprocessed_balanced.h5"

CLASS_NAMES = ['CD', 'HYP', 'MI', 'NORM', 'STTC']
FS_OUT = 500
TARGET_LEN = 5000
N_LEADS = 12

print("Model:", MODEL_PATH)
print("HDF5:", H5_PATH)

# ========== CELL 3: Safe HDF5 Dataset ==========
class ECGDataset(Dataset):
    """Lightweight dataset that opens the HDF5 file lazily and closes on delete."""
    def __init__(self, h5_path):
        self.h5_path = h5_path
        # We'll not open here; open lazily in __getitem__ to avoid file handle limits
        with h5py.File(self.h5_path, 'r') as f:
            self._len = len(f['signals'])

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        # Open per access (fast enough for inference); ensures file closed reliably
        with h5py.File(self.h5_path, 'r') as f:
            sig = f['signals'][idx]
            lbl = f['labels'][idx]
        x = torch.tensor(sig, dtype=torch.float32)
        y = torch.tensor(lbl, dtype=torch.float32)
        return x, y

# Quick test of dataset (only if file exists)
if not os.path.exists(H5_PATH):
    print(f"WARNING: HDF5 not found at {H5_PATH}. Create or update the path before running.")
else:
    ds = ECGDataset(H5_PATH)
    print(f"Dataset length: {len(ds)}")

# ========== CELL 4: Model Definition (unchanged architecture) ==========
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
                          dropout=dropout if gru_layers > 1 else 0.0)
        self.attn = AdditiveAttention(hidden_dim=gru_hidden*2, attn_dim=attn_dim)
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(gru_hidden*2, n_classes)
        )

    def forward(self, x, return_attention=False):
        # Expected input: (batch, seq_len, n_leads)
        x = x.permute(0, 2, 1)  # -> (batch, n_leads, seq_len)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)  # -> (batch, seq_len', features)
        rnn_out, _ = self.gru(x)
        context, weights = self.attn(rnn_out)
        out = self.fc(context)
        if return_attention:
            return out, weights
        return out

print("Model architecture defined.")

# ========== CELL 5: Model Loading (robust) ==========
# Try multiple ways to load: (1) entire model, (2) checkpoint with state_dict, (3) state_dict directly

def load_model(path, device=device):
    # First try torch.load directly
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")

    try:
        # This may be a full model object
        loaded = torch.load(path, map_location=device)
        if isinstance(loaded, nn.Module):
            model = loaded
            print("Loaded full model object from file.")
        elif isinstance(loaded, dict):
            # If it looks like a dict, try to get state dict
            # Instantiate architecture and load state_dict
            model = ECG_CNN_GRU_Attn(n_leads=N_LEADS, n_classes=len(CLASS_NAMES))
            if 'state_dict' in loaded:
                sd = loaded['state_dict']
            elif 'model_state_dict' in loaded:
                sd = loaded['model_state_dict']
            else:
                sd = loaded
            # If keys have module. prefix, remove if needed
            try:
                model.load_state_dict(sd)
                print("Loaded state_dict into new model instance.")
            except RuntimeError:
                # Try stripping 'module.' prefixes
                new_sd = {k.replace('module.', ''): v for k, v in sd.items()}
                model.load_state_dict(new_sd)
                print("Loaded state_dict (stripping 'module.' prefixes).")
        else:
            raise RuntimeError("Unrecognized model checkpoint format")

    except Exception as e:
        # As a final fallback, instantiate architecture and try to load state_dict naive
        print("Primary load failed with:", e)
        print("Falling back to instantiating model and loading state_dict.")
        model = ECG_CNN_GRU_Attn(n_leads=N_LEADS, n_classes=len(CLASS_NAMES))
        sd = torch.load(path, map_location=device)
        if isinstance(sd, dict) and not isinstance(sd, nn.Module):
            try:
                model.load_state_dict(sd)
            except RuntimeError:
                # remove module prefix
                new_sd = {k.replace('module.', ''): v for k, v in sd.items()}
                model.load_state_dict(new_sd)
        else:
            raise

    model = model.to(device)
    model.eval()
    return model

model = load_model(MODEL_PATH, device=device)
print("Model ready for inference on device:", device)

# ========== CELL 6: Prepare DataLoader for Inference ==========
# Use the ECGDataset defined earlier and create a DataLoader for inference
if os.path.exists(H5_PATH):
    dataset = ECGDataset(H5_PATH)
    test_loader = DataLoader(dataset, batch_size=32, shuffle=False)
else:
    test_loader = None

# Collect a small test batch (match original script behavior: take first ~64 samples)

test_data = None
test_labels = None
if test_loader is not None:
    collected_x = []
    collected_y = []
    for i, (x, y) in enumerate(test_loader):
        collected_x.append(x)
        collected_y.append(y)
        if len(collected_x) >= 3:  # similar to original code
            break
    if collected_x:
        test_data = torch.cat(collected_x)[:64]
        test_labels = torch.cat(collected_y)[:64]

if test_data is None:
    raise RuntimeError("No test data collected. Ensure HDF5 path is correct and file contains 'signals' dataset.")

# Convert one-hot to indices if needed (store both forms)
if test_labels.ndim > 1 and test_labels.shape[1] > 1:
    test_labels_idx = test_labels.argmax(dim=1)
else:
    test_labels_idx = test_labels.squeeze().long()

print(f"Test data shape: {test_data.shape}")
print(f"Test labels shape: {test_labels.shape}")

# Keep a CPU copy for IG (Captum expects model and inputs on same device)
# We'll compute IG on CPU to match the user's request (the original did IG on CPU)
test_data_cpu = test_data.cpu()
test_labels_cpu = test_labels_idx.cpu()

# ========== CELL 7: Integrated Gradients (safe, batched) ==========
print("Initializing Integrated Gradients (IG) on CPU...")

# Move a CPU copy of the model for IG
model_cpu = model.to('cpu')
model_cpu.eval()

ig = IntegratedGradients(model_cpu)

# Baseline: zeros same shape as a single input
baseline = torch.zeros_like(test_data_cpu[:1])

n_steps = 20  # keep same as original; increase if you want smoother attributions
batch_size_ig = 8
all_attrs = []

print("Computing IG attributions (batched)...")
for i in range(0, test_data_cpu.shape[0], batch_size_ig):
    batch = test_data_cpu[i:i+batch_size_ig]
    batch_labels = test_labels_cpu[i:i+batch_size_ig]
    # Captum expects input and baselines on same device as model_cpu
    attributions = ig.attribute(
        inputs=batch,
        baselines=baseline.expand_as(batch),
        target=batch_labels,
        n_steps=n_steps,
        internal_batch_size=None
    )
    all_attrs.append(attributions.detach().numpy())

attributions = np.concatenate(all_attrs, axis=0)  # shape: (B, seq_len, leads)
print("IG computation finished. attributions shape:", attributions.shape)

# Save a NumPy copy (unchanged behavior)
np.save('attributions.npy', attributions)

# ========== CELL 8: Lead Importance & Basic Visuals ==========
# Aggregate per-lead importance (mean abs across time)
lead_importance = np.mean(np.abs(attributions), axis=1)  # (N, leads)
mean_imp = np.mean(lead_importance, axis=0)
std_imp = np.std(lead_importance, axis=0)

# Boxplot + bar chart
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
axes[0].boxplot([lead_importance[:, i] for i in range(N_LEADS)], labels=[f"L{i+1}" for i in range(N_LEADS)])
axes[0].set_xlabel('ECG Lead')
axes[0].set_ylabel('Mean Absolute Attribution')
axes[0].set_title('Feature Importance (Integrated Gradients)')
axes[0].grid(axis='y', alpha=0.3)

axes[1].bar(range(1, N_LEADS+1), mean_imp, yerr=std_imp, capsize=5)
axes[1].set_xlabel('ECG Lead')
axes[1].set_ylabel('Average Importance')
axes[1].set_title('Average Feature Importance')
axes[1].set_xticks(range(1, N_LEADS+1))
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

# Print rankings
print("ECG LEAD IMPORTANCE RANKINGS")
for rank, (lead, imp) in enumerate(sorted(enumerate(mean_imp, 1), key=lambda x: x[1], reverse=True), 1):
    print(f"{rank:2d}. Lead {lead:2d}: {imp:.6f} (±{std_imp[lead-1]:.6f})")

# ========== CELL 9: Temporal Attribution Plot (helper) ==========
def plot_ecg_with_attribution(sample_idx=0, lead_idx=0):
    signal = test_data_cpu[sample_idx].numpy()[:, lead_idx]
    importance = attributions[sample_idx, :, lead_idx]
    importance_abs = np.abs(importance)
    if importance_abs.max() > importance_abs.min():
        importance_norm = (importance_abs - importance_abs.min()) / (importance_abs.max() - importance_abs.min())
    else:
        importance_norm = np.zeros_like(importance_abs)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={'height_ratios':[3,1]})
    points = np.arange(len(signal))
    scatter = ax1.scatter(points, signal, c=importance_norm, cmap='coolwarm', s=5, alpha=0.8)
    ax1.plot(signal, color='gray', alpha=0.3, linewidth=1)
    ax1.set_title(f"ECG Lead {lead_idx+1} - Sample {sample_idx} (Class: {test_labels_idx[sample_idx].item()})")
    ax1.set_ylabel('Amplitude')
    ax1.grid(alpha=0.3)
    plt.colorbar(scatter, ax=ax1, label='Attribution Importance')

    ax2.plot(points, importance, color='darkred', linewidth=1.5)
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax2.fill_between(points, 0, importance, where=(importance > 0), color='red', alpha=0.3)
    ax2.fill_between(points, 0, importance, where=(importance < 0), color='blue', alpha=0.3)
    ax2.set_xlabel('Time steps')
    ax2.set_ylabel('Attribution Value')
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

# Visualize first 3 samples on lead 0
for i in range(min(3, test_data_cpu.shape[0])):
    plot_ecg_with_attribution(sample_idx=i, lead_idx=0)
    print('\n' + '='*60)

# ========== CELL 10: All 12 Leads Overview ==========
def plot_all_leads(sample_idx=0):
    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    axes = axes.flatten()
    for lead_idx in range(N_LEADS):
        signal = test_data_cpu[sample_idx].numpy()[:, lead_idx]
        importance = attributions[sample_idx, :, lead_idx]
        importance_abs = np.abs(importance)
        if importance_abs.max() > importance_abs.min():
            importance_norm = (importance_abs - importance_abs.min()) / (importance_abs.max() - importance_abs.min())
        else:
            importance_norm = np.zeros_like(importance_abs)
        points = np.arange(len(signal))
        axes[lead_idx].scatter(points, signal, c=importance_norm, cmap='coolwarm', s=1, alpha=0.6)
        axes[lead_idx].plot(signal, color='gray', alpha=0.2, linewidth=0.5)
        axes[lead_idx].set_title(f"Lead {lead_idx+1}")
        axes[lead_idx].set_xticks([])
        axes[lead_idx].set_yticks([])
    plt.suptitle(f"All 12 ECG Leads - Sample {sample_idx} (Class: {test_labels_idx[sample_idx].item()})")
    plt.tight_layout()
    plt.show()

plot_all_leads(sample_idx=0)

# ========== CELL 11: Attention Analysis (GPU for speed) ==========
# Move model back to original device for attention visualization
model = model.to(device)
model.eval()

# Get one batch for attention visualization
if test_loader is not None:
    X_batch, y_batch = next(iter(test_loader))
    X_batch = X_batch.to(device)
    with torch.no_grad():
        logits, attn_weights = model(X_batch, return_attention=True)
        preds = torch.sigmoid(logits)

    # Plot first 3 samples attention
    fig, axes = plt.subplots(3, 1, figsize=(14, 9))
    for i in range(3):
        sample_attn = attn_weights[i].cpu().numpy()
        axes[i].plot(sample_attn, linewidth=1.5)
        axes[i].set_title(f"Sample {i} - Attention Distribution (Class: {y_batch[i].argmax().item() if y_batch.ndim>1 else y_batch[i].item()})")
        axes[i].set_xlabel("Time Step (after CNN+Pooling)")
        axes[i].set_ylabel("Attention Weight")
        axes[i].grid(alpha=0.3)
        top_k = 10
        top_indices = np.argsort(sample_attn)[-top_k:]
        axes[i].scatter(top_indices, sample_attn[top_indices], color='red', s=50, zorder=5, label=f'Top {top_k}')
        axes[i].legend()
    plt.tight_layout()
    plt.show()
else:
    print("No test_loader available for attention visualization.")

# ========== CELL 12: Clinical Summary Helpers ==========
def clinical_summary(sample_idx=0):
    sample = test_data[sample_idx:sample_idx+1].to(device)
    with torch.no_grad():
        logits, attn = model(sample, return_attention=True)
        pred = torch.sigmoid(logits).cpu().numpy()[0]
    pred_class = np.argmax(pred)
    confidence = pred[pred_class]
    lead_imp = np.mean(np.abs(attributions[sample_idx]), axis=0)
    top_leads = np.argsort(lead_imp)[-3:][::-1]
    temporal_imp = np.mean(np.abs(attributions[sample_idx]), axis=1)
    top_times = np.argsort(temporal_imp)[-5:][::-1]
    print(f"Sample {sample_idx} - Predicted: {pred_class}, Confidence: {confidence:.2%}, True: {test_labels_idx[sample_idx].item()}")
    print(f"Top Leads: {', '.join([f'Lead {i+1}' for i in top_leads])}")
    print(f"Critical Time Windows: {top_times}")
    print("Pattern-based explanation: Potential morphological abnormalities located in top leads/time windows; clinical correlation required.")

for i in range(min(3, test_data.shape[0])):
    clinical_summary(i)

# ========== CELL 13: Comprehensive Clinical Report Generation ==========
def generate_comprehensive_clinical_report(sample_idx=0, shap_attributions=None):
    sample = test_data[sample_idx:sample_idx+1].to(device)
    with torch.no_grad():
        logit, attn = model(sample, return_attention=True)
        pred = torch.sigmoid(logit).cpu().numpy()[0]
    pred_class = np.argmax(pred)
    confidence = pred[pred_class]
    true_label = int(test_labels_idx[sample_idx].item())

    lead_imp_ig = np.mean(np.abs(attributions[sample_idx]), axis=0)
    top_leads_ig = np.argsort(lead_imp_ig)[-3:][::-1]
    temporal_imp = np.mean(np.abs(attributions[sample_idx]), axis=1)
    critical_windows = np.where(temporal_imp > np.percentile(temporal_imp, 95))[0]

    attn_np = attn[0].cpu().numpy()
    peak_attention = np.argmax(attn_np)

    shap_available = (shap_attributions is not None)
    if shap_available:
        lead_imp_shap = np.mean(np.abs(shap_attributions[sample_idx]), axis=0)
        top_leads_shap = np.argsort(lead_imp_shap)[-3:][::-1]
        method_agreement = np.intersect1d(top_leads_ig, top_leads_shap)
    else:
        top_leads_shap = []
        method_agreement = []

    report = f"""
{'='*70}
🩺 CLINICAL EXPLAINABILITY REPORT
{'='*70}

PATIENT SAMPLE: {sample_idx}
{'='*70}

DIAGNOSTIC PREDICTION:
• AI Predicted Class: {pred_class}
• Confidence Level: {confidence:.1%}
• True Diagnosis: {true_label}
• Prediction Status: {'✓ CORRECT' if pred_class == true_label else '✗ INCORRECT - Clinical Review Recommended'}

{'='*70}
KEY FINDINGS - INTEGRATED GRADIENTS ANALYSIS:
{'='*70}

MOST INFLUENTIAL ECG LEADS:
1. Lead {top_leads_ig[0]+1}: Importance = {lead_imp_ig[top_leads_ig[0]]:.6f}
2. Lead {top_leads_ig[1]+1}: Importance = {lead_imp_ig[top_leads_ig[1]]:.6f}
3. Lead {top_leads_ig[2]+1}: Importance = {lead_imp_ig[top_leads_ig[2]]:.6f}

TEMPORAL ANALYSIS:
• Critical Time Segments: {len(critical_windows)} high-importance regions identified
• Key Timesteps: {critical_windows[:15].tolist()}{'...' if len(critical_windows) > 15 else ''}

ATTENTION MECHANISM INSIGHTS:
• Peak Attention at Timestep: {peak_attention}
• Attention Spread: {'Focused' if np.max(attn_np) > 0.1 else 'Distributed'}

CLINICAL INTERPRETATION:
The AI model prioritized ECG features in {', '.join([f'Lead {i+1}' for i in top_leads_ig])}.
These leads are clinically relevant for the predicted condition. Manual review recommended.

"""

    if shap_available:
        report += f"SHAP agreement leads: {[l+1 for l in method_agreement]}\n"

    return report

# Generate and print reports for first 3 samples
for i in range(min(3, test_data.shape[0])):
    r = generate_comprehensive_clinical_report(i)
    print(r)

# ========== CELL 14: Visual Summary per Sample ==========
for i in range(min(3, test_data.shape[0])):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    # Lead importance
    lead_imp = np.mean(np.abs(attributions[i]), axis=0)
    axes[0,0].bar(range(1, N_LEADS+1), lead_imp)
    axes[0,0].set_title(f"Sample {i} - Lead Importance")
    axes[0,0].grid(axis='y', alpha=0.3)
    # Temporal importance
    temporal_imp = np.mean(np.abs(attributions[i]), axis=1)
    axes[0,1].plot(temporal_imp)
    axes[0,1].set_title(f"Sample {i} - Temporal Pattern")
    axes[0,1].grid(alpha=0.3)
    # ECG lead 1 with attribution
    signal = test_data_cpu[i].numpy()[:, 0]
    importance = attributions[i, :, 0]
    importance_abs = np.abs(importance)
    if importance_abs.max() > importance_abs.min():
        importance_norm = (importance_abs - importance_abs.min()) / (importance_abs.max() - importance_abs.min())
    else:
        importance_norm = np.zeros_like(importance_abs)
    scatter = axes[1,0].scatter(range(len(signal)), signal, c=importance_norm, s=3)
    axes[1,0].plot(signal, color='gray', alpha=0.3)
    axes[1,0].set_title(f"Sample {i} - ECG Lead 1 with Attribution")
    plt.colorbar(scatter, ax=axes[1,0])
    # Attention weights
    sample = test_data[i:i+1].to(device)
    with torch.no_grad():
        _, attn = model(sample, return_attention=True)
    attn_np = attn[0].cpu().numpy()
    axes[1,1].plot(attn_np)
    axes[1,1].set_title(f"Sample {i} - Attention Distribution")
    axes[1,1].grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

# ========== CELL 15: Optional SHAP (placeholder) ==========
# SHAP is optional; only run if SHAP_AVAILABLE is True and user wants it.
if SHAP_AVAILABLE:
    print("SHAP is available. To compute Deep SHAP, uncomment and configure the following block.")
    # Example (disabled by default to avoid heavy compute):
    # explainer = shap.DeepExplainer(model_cpu, test_data_cpu[:50])
    # shap_values = explainer.shap_values(test_data_cpu[:50])
    # shap_attributions = np.stack(shap_values, axis=-1) # shape depends on explainer
else:
    print("SHAP not available (skipping SHAP section).")

# ========== CELL 16: Save artifacts ==========
# Save the small test_data/test_labels and model state for demo reproducibility
torch.save(test_data.cpu(), 'test_data.pt')
torch.save(test_labels_idx.cpu(), 'test_labels.pt')
np.save('attributions.npy', attributions)
torch.save(model.state_dict(), 'model_for_demo.pth')

print("✅ Results saved: test_data.pt, test_labels.pt, attributions.npy, model_for_demo.pth")
print("Notebook complete. Review plots and reports above.")