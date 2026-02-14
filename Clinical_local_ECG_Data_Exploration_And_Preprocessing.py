import os
import numpy as np
import h5py
import xml.etree.ElementTree as ET
from scipy.signal import resample_poly, butter, filtfilt, iirnotch
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import base64
import struct

XML_PATH = "./Data/local-data/local_ecg_data.xml"
OUTPUT_DIR = "./processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_H5 = os.path.join(OUTPUT_DIR, "clinical_preprocessed.h5")
TARGET_LEN = 5000
FS_OUT = 500
LEADS = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]

print("XML File:", XML_PATH)
print("Output file will be saved to:", OUTPUT_H5)
# ========== CELL 2: Parse XML ==========
tree = ET.parse(XML_PATH)
root = tree.getroot()

print("Root Tag:", root.tag)
print("Number of children:", len(list(root)))
print("\nXML Structure:")
for child in root:
    print(f"  - {child.tag}")

# Find waveform container
waveform_block = None
for elem in root.iter():
    if "waveform" in elem.tag.lower():
        waveform_block = elem
        break

if waveform_block is None:
    raise ValueError("Waveform section not detected")

records = list(waveform_block)
print(f"\n Total waveform elements found: {len(records)}")
print("   (Note: 1 = full ECG, 2 = full ECG + representative beats)")
# ========== CELL 3: Create Index File ==========
# We only have 1 actual ECG recording
index_data = [{
    "index": 0,
    "type": "parsedwaveforms",
    "date": "2025-11-25",
    "diagnosis": "STTC (ST elevation - probable early repolarization)"
}]

df_index = pd.DataFrame(index_data)
index_csv_path = os.path.join(OUTPUT_DIR, "clinical_ecg_index.csv")
df_index.to_csv(index_csv_path, index=False)

print(" ECG Index file created:", index_csv_path)
print("\nPatient Information:")
print(df_index.to_string(index=False))
# ========== CELL 4: Select Indices ==========
# Only use index 0 (parsedwaveforms - the full ECG recording)
SELECTED_INDICES = [0]
print(f"Selected indices for processing: {SELECTED_INDICES}")
print("(Using only the full ECG waveform, not the representative beats)")
# ========== CELL 5: Preprocessing Functions (FIXED) ==========
from scipy.signal import medfilt

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut/nyq, highcut/nyq], btype='band')
    return b, a

def bandpass_filter_clinical(sig, fs, lowcut=0.05, highcut=100):
    """Gentler bandpass filter for clinical ECGs"""
    b, a = butter_bandpass(lowcut, highcut, fs, order=2)  # Lower order = less aggressive
    return filtfilt(b, a, sig)

def baseline_wander_removal(sig, window_size=201):
    """Remove baseline wander using median filter"""
    # Ensure window_size is odd
    if window_size % 2 == 0:
        window_size += 1
    baseline = medfilt(sig, kernel_size=window_size)
    return sig - baseline

def notch_filter_gentle(sig, fs, freq=50.0, Q=15):
    """Gentler notch filter"""
    b, a = iirnotch(freq, Q, fs)
    return filtfilt(b, a, sig)

def preprocess_signal_clinical(sig, fs):
    """
    Clinical-grade preprocessing - less aggressive than research preprocessing
    """
    # Step 1: Remove baseline wander first
    window_size = int(fs * 0.6)  # 0.6s window
    if window_size % 2 == 0:  # ← FIX: Ensure odd
        window_size += 1
    sig = baseline_wander_removal(sig, window_size=window_size)
    
    # Step 2: Gentle bandpass filter (wider range)
    sig = bandpass_filter_clinical(sig, fs, lowcut=0.05, highcut=100)
    
    # Step 3: Gentle notch filter (optional - only if needed)
    # sig = notch_filter_gentle(sig, fs, freq=50.0, Q=15)
    
    # Step 4: Resample if needed
    if fs != FS_OUT:
        sig = resample_poly(sig, FS_OUT, fs)
    
    # Step 5: Robust normalization (uses percentiles instead of mean/std)
    p5 = np.percentile(sig, 5)
    p95 = np.percentile(sig, 95)
    sig = (sig - np.median(sig)) / (p95 - p5 + 1e-8)
    
    # Step 6: Clip outliers (optional)
    sig = np.clip(sig, -5, 5)
    
    # Step 7: Fixed length
    if len(sig) < TARGET_LEN:
        sig = np.pad(sig, (0, TARGET_LEN - len(sig)), mode='edge')  # Edge padding instead of zero
    else:
        sig = sig[:TARGET_LEN]
    
    return sig

def decode_base64_waveform(b64_string, num_leads=12):
    """Decode Philips Base64-encoded waveform data"""
    compressed_data = base64.b64decode(b64_string)
    num_samples = len(compressed_data) // 2
    samples = struct.unpack(f'<{num_samples}h', compressed_data)
    samples_per_lead = num_samples // num_leads
    waveform = np.array(samples).reshape(samples_per_lead, num_leads)
    return waveform

print(" Clinical preprocessing functions defined (FIXED)")
print("Changes from original:")
print("  • Wider bandpass: 0.05-100 Hz (was 0.5-45 Hz)")
print("  • Baseline wander removal added (with odd kernel fix)")
print("  • Gentler notch filter (commented out by default)")
print("  • Robust normalization using percentiles")
print("  • Outlier clipping added")
print("  • Edge padding instead of zero padding")
# ========== CELL 6: Extract and Process Waveforms (UPDATED) ==========
import base64
import numpy as np

signals = []
ids = []
metadata = []

ns = "{http://www3.medical.philips.com}"

print(" Searching for Philips waveforms...")

waveform_elem = root.find(f".//{ns}parsedwaveforms")

if waveform_elem is None:
    print(" No parsedwaveforms found in the entire file")
else:
    print(" parsedwaveforms tag FOUND")
    
    raw_text = waveform_elem.text.strip()
    decoded_bytes = base64.b64decode(raw_text)
    data = np.frombuffer(decoded_bytes, dtype=np.int16)
    
    num_leads = int(waveform_elem.attrib.get("numberofleads", 12))
    fs = int(waveform_elem.attrib.get("samplespersecond", 500))
    
    print(f"Total decoded values: {data.shape[0]}")
    
    # Reshape safely
    valid_length = (data.shape[0] // num_leads) * num_leads
    data = data[:valid_length]
    total_samples = valid_length // num_leads
    waveform = data.reshape((total_samples, num_leads))
    
    print(f" Raw ECG shape: {waveform.shape}")
    print(f"Original sampling rate: {fs} Hz")
    print(f"Original duration: {total_samples/fs:.2f} seconds")
    
    #  USE CLINICAL PREPROCESSING
    processed_leads = []
    for lead in range(num_leads):
        sig = waveform[:, lead].astype(float)
        sig = preprocess_signal_clinical(sig, fs)  # ← CHANGED HERE
        processed_leads.append(sig)
    
    processed = np.stack(processed_leads, axis=-1)
    signals.append(processed)
    ids.append("clinical_ecg_0")
    
    metadata.append({
        "id": "clinical_ecg_0",
        "original_samples": total_samples,
        "processed_samples": TARGET_LEN,
        "leads": num_leads,
        "preprocessing": "clinical_grade"
    })
    
    print("\n" + "="*60)
    print(f" ECG signals extracted: {len(signals)}")
    if signals:
        print(f" Final model input shape: {signals[0].shape}")
        print(f" Preprocessing: Clinical-grade (gentler filtering)")
        # ========== CELL 7: Save to HDF5 ==========
if len(signals) == 0:
    print(" ERROR: No signals extracted! Check XML parsing.")
else:
    signals = np.array(signals, dtype=np.float32)
    labels = np.zeros((len(signals), 5), dtype=np.float32)
    
    with h5py.File(OUTPUT_H5, 'w') as f:
        f.create_dataset('signals', data=signals, compression='gzip')
        f.create_dataset('labels', data=labels, compression='gzip')
        f.create_dataset('ids', data=np.array(ids, dtype='S'), compression='gzip')
    
    print("="*60)
    print(" HDF5 FILE SAVED SUCCESSFULLY")
    print("="*60)
    print(f"Location: {OUTPUT_H5}")
    print(f"Signals shape: {signals.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"IDs count: {len(ids)}")
    print(f"\nExpected by model: (N, 5000, 12)")
    print(f"Your data shape:   {signals.shape}")
    
    df_meta = pd.DataFrame(metadata)
    meta_csv_path = os.path.join(OUTPUT_DIR, "clinical_metadata.csv")
    df_meta.to_csv(meta_csv_path, index=False)
    print(f"\n Metadata saved: {meta_csv_path}")
def plot_ecg(signal, fs, leads, n_samples=1000, title="ECG"):
    t = np.arange(min(signal.shape[0], n_samples)) / fs
    n_leads = signal.shape[1]
    n_rows = int(np.ceil(n_leads / 2))
    
    plt.figure(figsize=(14, 3*n_rows))
    for i, lead in enumerate(leads):
        plt.subplot(n_rows, 2, i+1)
        plt.plot(t, signal[:len(t), i], linewidth=0.8)
        plt.title(f"{title} - {lead}")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude (normalized)")
        plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

if len(signals) > 0:
    print("Visualizing first ECG...")
    plot_ecg(signals[0], FS_OUT, LEADS, n_samples=2000, title="Preprocessed Clinical ECG")
else:
    print("No signals to visualize")
    