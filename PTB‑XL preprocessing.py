## 0.library
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, iirnotch, resample_poly
import wfdb
from tqdm.notebook import tqdm
import h5py
import seaborn as sns
from collections import Counter
from sklearn.preprocessing import MultiLabelBinarizer
import ast
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
import warnings
warnings.filterwarnings("ignore")
from imblearn.over_sampling import RandomOverSampler

# 1.Configuration
DATA_PATH = "./Data/ptb-xl-1.0.3"
OUTPUT_PATH = "./processed"
REPORT_PATH = "./preprocessing_reports"
os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(REPORT_PATH, exist_ok=True)
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
# 2.Load Metadata
meta_file = os.path.join(DATA_PATH, "ptbxl_database.csv")
scp_file = os.path.join(DATA_PATH, "scp_statements.csv")
meta = pd.read_csv(meta_file)
scp = pd.read_csv(scp_file)
print("Shape of metadata:", meta.shape)
# 3.Metadata Exploration
print(meta.info())
print(meta.describe(include="all"))
plt.figure(figsize=(12,5))
sns.heatmap(meta.isnull(), cbar=False)
plt.title("Missing values BEFORE cleaning")
plt.show()

# Optional: check number of missing per column
print(meta.isnull().sum())
meta_clean = meta.copy()
meta_clean.dropna(subset=["age","sex"], inplace=True)
cols_to_drop = [
    "height","weight","infarction_stadium1","infarction_stadium2",
    "baseline_drift","static_noise","burst_noise",
    "electrodes_problems","extra_beats","pacemaker"
]
meta_clean.drop(columns=cols_to_drop, inplace=True, errors='ignore')
print("Dropped columns:", cols_to_drop)

# Convert heart_axis to string, replace 'nan', then fill with mode
meta_clean["heart_axis"] = meta_clean["heart_axis"].astype(str).replace("nan", np.nan)
meta_clean["heart_axis"] = meta_clean["heart_axis"].fillna(meta_clean["heart_axis"].mode()[0])
# Fill categorical missing values
meta_clean["nurse"] = meta_clean["nurse"].fillna("unknown")
meta_clean["site"] = meta_clean["site"].fillna(meta_clean["site"].mode()[0])
meta_clean["validated_by"] = meta_clean["validated_by"].fillna("unknown")
print("Missing values AFTER cleaning:\n", meta_clean.isnull().sum())
plt.figure(figsize=(12,5))
sns.heatmap(meta_clean.isnull(), cbar=False)
plt.title("Missing values AFTER cleaning")
plt.show()
print(scp.columns)
scp.head()
meta_clean['scp_codes'] = meta_clean['scp_codes'].apply(ast.literal_eval)
scp_df = pd.read_csv(scp_file, index_col=0)
diagnostic_classes = scp_df[scp_df.diagnostic == 1].index

meta_clean['diagnostic_codes'] = meta_clean['scp_codes'].apply(
    lambda x: {k: v for k, v in x.items() if k in diagnostic_classes}
)
# Drop rows with no diagnostic codes
meta_clean = meta_clean[meta_clean['diagnostic_codes'].map(len) > 0]

print("Remaining rows after filtering for diagnostic labels:", meta_clean.shape[0])
label_mapping = scp_df.loc[diagnostic_classes]['diagnostic_class'].to_dict()

meta_clean['diagnostic_superclass'] = meta_clean['diagnostic_codes'].apply(
    lambda x: list(set(label_mapping[k] for k in x.keys()))
)
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(meta_clean['diagnostic_superclass'])
# Split using PTB-XL recommended folds
train_set = meta_clean[meta_clean['strat_fold'] < 9].reset_index(drop=True)
val_set   = meta_clean[meta_clean['strat_fold'] == 9].reset_index(drop=True)
test_set  = meta_clean[meta_clean['strat_fold'] == 10].reset_index(drop=True)
y_train_flat = ['_'.join(sorted(label)) for label in train_set['diagnostic_superclass']]
ros = RandomOverSampler(random_state=RANDOM_SEED)
train_idx_resampled, _ = ros.fit_resample(np.arange(len(train_set)).reshape(-1,1), y_train_flat)
train_set_balanced = train_set.iloc[train_idx_resampled.flatten()].reset_index(drop=True)
class_counts_before = Counter([c for classes in meta_clean['diagnostic_superclass'] for c in classes])

plt.figure(figsize=(10, 4))
plt.bar(class_counts_before.keys(), class_counts_before.values(), color="steelblue")
plt.title("Class Distribution BEFORE Split")
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(sig, lowcut=0.5, highcut=45.0, fs=500):
    b, a = butter_bandpass(lowcut, highcut, fs)
    return filtfilt(b, a, sig)

def notch_filter(sig, fs=500, freq=50.0, Q=30):
    b, a = iirnotch(freq, Q, fs)
    return filtfilt(b, a, sig)

def preprocess_signal(sig, fs_in, fs_out=500, target_len=5000, augment=False):
    sig = bandpass_filter(sig, 0.5, 45, fs=fs_in)
    sig = notch_filter(sig, fs=fs_in)
    sig = resample_poly(sig, fs_out, fs_in)
    
    # Normalize
    mean = np.mean(sig, axis=0)
    std = np.std(sig, axis=0) + 1e-8
    sig = (sig - mean) / std
    
    # Augmentation for rare classes
    if augment:
        sig += np.random.normal(0, 0.01, size=sig.shape)
        scale = np.random.uniform(0.95, 1.05)
        sig *= scale
        shift = np.random.randint(-10, 10)
        sig = np.roll(sig, shift)
        stretch = np.interp(np.linspace(0, len(sig)-1, len(sig)), np.arange(len(sig)), sig)
        sig = stretch

    # Fixed length
    if len(sig) < target_len:
        sig = np.pad(sig, (0, target_len - len(sig)), mode="constant")
    elif len(sig) > target_len:
        start = (len(sig) - target_len) // 2
        sig = sig[start:start + target_len]

    return sig
leads = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]
def plot_ecg(signal, fs, leads, n_samples=1000, title="ECG"):
    t = np.arange(signal.shape[0]) / fs
    n_leads = signal.shape[1]
    n_rows = int(np.ceil(n_leads / 2))
    plt.figure(figsize=(12, 3*n_rows))
    for i, lead in enumerate(leads):
        plt.subplot(n_rows, 2, i+1)
        plt.plot(t[:n_samples], signal[:n_samples, i])
        plt.title(f"{title} - {lead}")
        plt.tight_layout()
    plt.show()
    sample_idx = 0
record_path = os.path.join(DATA_PATH, meta_clean.iloc[sample_idx]["filename_hr"].replace(".mat",""))

# Unpack wfdb.rdsamp
sig_raw, fields = wfdb.rdsamp(record_path)
fs_in = int(fields['fs'] if 'fs' in fields else meta_clean.iloc[sample_idx].get("sampling_frequency", 500))

print("Raw signal shape:", sig_raw.shape)
plot_ecg(sig_raw, fs_in, leads, n_samples=1000, title="Raw ECG")
sig_preprocessed = np.stack([preprocess_signal(sig_raw[:, i], fs_in, fs_out=500) 
                             for i in range(sig_raw.shape[1])], axis=-1)
plot_ecg(sig_preprocessed, 500, leads, n_samples=1000, title="Preprocessed ECG")
def check_nan_inf(signal):
    return np.isnan(signal).any() or np.isinf(signal).any()

def check_flatline(signal, tol=1e-6):
    return np.all(np.abs(np.diff(signal, axis=0)) < tol)

def check_saturation(signal, max_val=32767, min_val=-32768):
    return np.any(signal >= max_val) or np.any(signal <= min_val)
fs_out = 500
target_len = 5000
h5_path = os.path.join(OUTPUT_PATH, "ptbxl_preprocessed_balanced.h5")
batch_size = 1000

if os.path.exists(h5_path):
    os.remove(h5_path)
    print("Old file removed, creating a new one...")

skipped_records = []
n_written = 0

# Temporary buffers
batch_signals = []
batch_labels = []
batch_ids = []

with h5py.File(h5_path, "w") as f:
    signals_ds = f.create_dataset(
        "signals",
        shape=(0, target_len, 12),
        maxshape=(None, target_len, 12),
        dtype="float32",
        compression="gzip"
    )
    labels_ds = f.create_dataset(
        "labels",
        shape=(0, len(mlb.classes_)),
        maxshape=(None, len(mlb.classes_)),
        dtype="int8",
        compression="gzip"
    )
    ids_ds = f.create_dataset(
        "ids",
        shape=(0,),
        maxshape=(None,),
        dtype=h5py.string_dtype(encoding="utf-8"),
        compression="gzip"
    )

    for idx, row in tqdm(train_set_balanced.iterrows(), total=train_set_balanced.shape[0]):
        try:
            record_path = os.path.join(DATA_PATH, row["filename_hr"].replace(".mat", ""))
            sig, fields = wfdb.rdsamp(record_path)
            fs_in = int(fields.get('fs', row.get("sampling_frequency", 500)))

            rare_class = any(row['diagnostic_superclass'].count(c) < 50 for c in row['diagnostic_superclass'])
            processed_leads = [
                preprocess_signal(sig[:, i], fs_in, fs_out, target_len, augment=rare_class).astype("float32")
                for i in range(sig.shape[1])
            ]
            processed = np.stack(processed_leads, axis=-1)

            if check_nan_inf(processed) or check_flatline(processed) or check_saturation(processed):
                skipped_records.append(row["ecg_id"])
                continue

            batch_signals.append(processed)
            batch_labels.append(mlb.transform([row["diagnostic_superclass"]])[0].astype("int8"))
            batch_ids.append(str(row["ecg_id"]))

            # When batch full, write it to HDF5
            if len(batch_signals) == batch_size:
                batch_signals = np.stack(batch_signals)
                batch_labels = np.stack(batch_labels)

                new_size = n_written + batch_signals.shape[0]
                signals_ds.resize((new_size, target_len, 12))
                labels_ds.resize((new_size, len(mlb.classes_)))
                ids_ds.resize((new_size,))

                signals_ds[n_written:new_size] = batch_signals
                labels_ds[n_written:new_size] = batch_labels
                ids_ds[n_written:new_size] = batch_ids

                n_written = new_size
                batch_signals, batch_labels, batch_ids = [], [], []

        except Exception as e:
            skipped_records.append(row["ecg_id"])
            continue

    # Write remaining samples
    if len(batch_signals) > 0:
        batch_signals = np.stack(batch_signals)
        batch_labels = np.stack(batch_labels)
        new_size = n_written + batch_signals.shape[0]

        signals_ds.resize((new_size, target_len, 12))
        labels_ds.resize((new_size, len(mlb.classes_)))
        ids_ds.resize((new_size,))

        signals_ds[n_written:new_size] = batch_signals
        labels_ds[n_written:new_size] = batch_labels
        ids_ds[n_written:new_size] = batch_ids

        n_written = new_size

print(f" Processed {n_written} signals, skipped {len(skipped_records)}.")
with h5py.File(h5_path, "r") as f:
    all_ids = list(f["ids"][:])
    all_ids = [id.decode() if isinstance(id, bytes) else id for id in all_ids]  # decode bytes if needed

qc_df = pd.DataFrame({"ecg_id": all_ids})
qc_csv_path = os.path.join(REPORT_PATH, "ptbxl_qc_summary_balanced.csv")
qc_df.to_csv(qc_csv_path, index=False)
print(f"QC summary saved to: {qc_csv_path}")
def load_ptbxl_flat(h5_path):
    with h5py.File(h5_path, "r") as f:
        signals = f["signals"][:]
        labels = f["labels"][:]
        ids = [id.decode() if isinstance(id, bytes) else id for id in f["ids"][:]]
    return signals, labels, ids
# Example load
signals, labels, ids = load_ptbxl_flat(h5_path)
print("Signals shape:", signals.shape)
print("Labels shape:", labels.shape)
print("Number of IDs:", len(ids))