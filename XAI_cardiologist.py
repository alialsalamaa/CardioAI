#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CARDIOLOGIST DEMONSTRATION TOOL
================================
This script provides an interactive demonstration of XAI results for cardiologists.

Prerequisites:
    1. Run xai_analysis.py first to generate the required data files
    2. Ensure the following files exist:
       - test_data.pt
       - test_labels.pt
       - attributions.npy
       - model_for_demo.pth

Usage:
    python cardiologist_demo.py
    
    To demo different samples, edit the sample_to_demo variable at the bottom.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os

# Create output directory for saving figures
os.makedirs('demo_outputs', exist_ok=True)

# =============================================================================
# MODEL DEFINITION (Same as your xai_analysis.py)
# =============================================================================

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
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        rnn_out, _ = self.gru(x)
        context, weights = self.attn(rnn_out)
        out = self.fc(context)
        
        if return_attention:
            return out, weights
        return out

# =============================================================================
# LOAD SAVED DATA
# =============================================================================

def load_xai_results():
    """Load previously saved XAI analysis results"""
    print("\n" + "="*70)
    print("LOADING XAI ANALYSIS RESULTS")
    print("="*70)
    
    # Check if files exist
    required_files = ['test_data.pt', 'test_labels.pt', 'attributions.npy', 'model_for_demo.pth']
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print("❌ ERROR: Missing required files:")
        for f in missing_files:
            print(f"   - {f}")
        print("\n⚠️  Please run xai_analysis.py first to generate these files.")
        return None, None, None, None, None
    
    # Load device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    print("Loading test data...")
    test_data = torch.load('test_data.pt')
    test_labels = torch.load('test_labels.pt')
    attributions = np.load('attributions.npy')
    
    print("Loading model...")
    model = ECG_CNN_GRU_Attn(n_leads=12, n_classes=5)
    model.load_state_dict(torch.load('model_for_demo.pth', map_location=device))
    model.to(device)
    model.eval()
    
    print(f"\n✅ Successfully loaded:")
    print(f"   - {len(test_data)} test samples")
    print(f"   - Attributions shape: {attributions.shape}")
    print(f"   - Model ready on {device}")
    print("="*70 + "\n")
    
    return test_data, test_labels, attributions, model, device

# =============================================================================
# DEMONSTRATION FUNCTIONS
# =============================================================================

def select_demonstration_sample(test_data, test_labels, model, device):
    """Find interesting samples for demonstration"""
    demo_samples = {
        'high_confidence_correct': None,
        'moderate_confidence_correct': None,
        'incorrect': None
    }
    
    model.eval()
    with torch.no_grad():
        for idx in range(len(test_data)):
            sample = test_data[idx:idx+1].to(device)
            true_label = test_labels[idx].item()
            
            logits, _ = model(sample, return_attention=True)
            pred = torch.sigmoid(logits).cpu().numpy()[0]
            pred_class = np.argmax(pred)
            confidence = pred[pred_class]
            
            if pred_class == true_label:
                if confidence > 0.9 and demo_samples['high_confidence_correct'] is None:
                    demo_samples['high_confidence_correct'] = idx
                elif 0.6 < confidence <= 0.9 and demo_samples['moderate_confidence_correct'] is None:
                    demo_samples['moderate_confidence_correct'] = idx
            else:
                if demo_samples['incorrect'] is None:
                    demo_samples['incorrect'] = idx
            
            if all(v is not None for v in demo_samples.values()):
                break
    
    return demo_samples

def show_standard_12lead_ecg(test_data, sample_idx, test_labels):
    """Display ECG in standard 12-lead format"""
    ecg_signal = test_data[sample_idx].numpy()
    true_label = test_labels[sample_idx].item()
    
    lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 
                  'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    
    fig, axes = plt.subplots(12, 1, figsize=(16, 14))
    fig.suptitle(f'12-Lead ECG - Patient Sample {sample_idx} (True Diagnosis: Class {true_label})', 
                 fontsize=16, fontweight='bold')
    
    for lead_idx in range(12):
        axes[lead_idx].plot(ecg_signal[:, lead_idx], color='black', linewidth=0.8)
        axes[lead_idx].set_ylabel(f"{lead_names[lead_idx]}", fontsize=12, fontweight='bold')
        axes[lead_idx].set_xlim(0, 5000)
        axes[lead_idx].grid(True, alpha=0.3)
        axes[lead_idx].set_yticks([])
        
        if lead_idx < 11:
            axes[lead_idx].set_xticks([])
    
    axes[11].set_xlabel("Time (samples @ 500 Hz)", fontsize=12)
    plt.tight_layout()
    plt.savefig(f'demo_outputs/ecg_sample_{sample_idx}_original.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n" + "="*70)
    print("📋 STEP 1: ORIGINAL ECG PRESENTATION")
    print("="*70)
    print(f"Sample ID: {sample_idx}")
    print(f"True Diagnosis: Class {true_label}")
    print("\n🗣️  SAY TO CARDIOLOGIST:")
    print(f"   'This is a standard 12-lead ECG from our test set.'")
    print(f"   'The true diagnosis is Class {true_label}.'")
    print(f"   'Let me show you what the AI predicted and WHY.'")
    print("="*70)

def show_ai_prediction(test_data, sample_idx, test_labels, model, device):
    """Display AI prediction, confidence, and correctness"""
    sample = test_data[sample_idx:sample_idx+1].to(device)
    true_label = test_labels[sample_idx].item()
    
    with torch.no_grad():
        logits, attn = model(sample, return_attention=True)
        pred = torch.sigmoid(logits).cpu().numpy()[0]
    
    pred_class = np.argmax(pred)
    confidence = pred[pred_class]
    is_correct = pred_class == true_label
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: All class probabilities
    classes = range(len(pred))
    colors = ['green' if i == pred_class else 'gray' for i in classes]
    ax1.bar(classes, pred * 100, color=colors, alpha=0.7)
    ax1.set_xlabel('Class', fontsize=12)
    ax1.set_ylabel('Confidence (%)', fontsize=12)
    ax1.set_title('AI Prediction Confidence per Class', fontsize=14)
    ax1.set_xticks(classes)
    ax1.grid(axis='y', alpha=0.3)
    
    # Right: Prediction summary
    ax2.axis('off')
    status_text = '✓ CORRECT' if is_correct else '✗ INCORRECT'
    
    summary_text = f"""
    AI PREDICTION SUMMARY
    {'='*40}
    
    Predicted Class: {pred_class}
    Confidence: {confidence*100:.2f}%
    
    True Diagnosis: {true_label}
    Status: {status_text}
    
    {'='*40}
    """
    
    ax2.text(0.1, 0.5, summary_text, fontsize=14, family='monospace',
             verticalalignment='center', 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(f'demo_outputs/ecg_sample_{sample_idx}_prediction.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n" + "="*70)
    print("🤖 STEP 2: AI PREDICTION")
    print("="*70)
    print(f"Predicted Class: {pred_class}")
    print(f"Confidence: {confidence*100:.2f}%")
    print(f"True Label: {true_label}")
    print(f"Correctness: {status_text}")
    print("\n🗣️  SAY TO CARDIOLOGIST:")
    print(f"   'The AI predicted Class {pred_class} with {confidence*100:.1f}% confidence.'")
    if is_correct:
        print(f"   'This matches the ground truth diagnosis.'")
    else:
        print(f"   '⚠️  This DISAGREES with the true diagnosis (Class {true_label}).'")
        print(f"   'Let\\'s see what the AI was looking at that led to this decision.'")
    print("="*70)
    
    return pred_class, confidence, is_correct

def show_attribution_explanation(test_data, attributions, sample_idx, 
                                 test_labels, top_n_leads=3):
    """Show XAI explanation focusing on most important leads"""
    lead_importance = np.mean(np.abs(attributions[sample_idx]), axis=0)
    top_lead_indices = np.argsort(lead_importance)[-top_n_leads:][::-1]
    
    lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 
                  'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    
    fig, axes = plt.subplots(top_n_leads, 1, figsize=(16, 4*top_n_leads))
    if top_n_leads == 1:
        axes = [axes]
    
    for plot_idx, lead_idx in enumerate(top_lead_indices):
        signal = test_data[sample_idx].numpy()[:, lead_idx]
        importance = attributions[sample_idx, :, lead_idx]
        
        importance_abs = np.abs(importance)
        if importance_abs.max() > importance_abs.min():
            importance_norm = (importance_abs - importance_abs.min()) / \
                             (importance_abs.max() - importance_abs.min())
        else:
            importance_norm = np.zeros_like(importance_abs)
        
        points = np.arange(len(signal))
        scatter = axes[plot_idx].scatter(points, signal, c=importance_norm, 
                                        cmap='coolwarm', s=10, alpha=0.8, 
                                        vmin=0, vmax=1)
        axes[plot_idx].plot(signal, color='gray', alpha=0.3, linewidth=1)
        axes[plot_idx].set_title(
            f"Lead {lead_names[lead_idx]} (Rank #{plot_idx+1} Most Important) - Importance: {lead_importance[lead_idx]:.6f}",
            fontsize=13, fontweight='bold'
        )
        axes[plot_idx].set_ylabel("Amplitude", fontsize=11)
        axes[plot_idx].grid(alpha=0.3)
        
        cbar = plt.colorbar(scatter, ax=axes[plot_idx])
        cbar.set_label('Attribution Importance', fontsize=10)
    
    axes[-1].set_xlabel("Time (samples @ 500 Hz)", fontsize=12)
    plt.tight_layout()
    plt.savefig(f'demo_outputs/ecg_sample_{sample_idx}_xai_explanation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n" + "="*70)
    print("🔍 STEP 3: XAI EXPLANATION - WHY THIS DIAGNOSIS?")
    print("="*70)
    print(f"Top {top_n_leads} Most Important Leads:")
    for rank, lead_idx in enumerate(top_lead_indices, 1):
        print(f"  {rank}. Lead {lead_names[lead_idx]} (importance: {lead_importance[lead_idx]:.6f})")
    
    print("\n🗣️  SAY TO CARDIOLOGIST:")
    print("   'The colored regions show where the AI focused its analysis.'")
    print("   '🔴 RED areas = High importance for the diagnosis'")
    print("   '🔵 BLUE areas = Less important / background signal'")
    print(f"\n   'The AI primarily relied on Leads {', '.join([lead_names[i] for i in top_lead_indices])}'")
    print("\n   ❓ QUESTION: 'Do these highlighted regions match where YOU would look")
    print("                  to diagnose this condition?'")
    print("="*70)

def show_comprehensive_12lead_attribution(test_data, attributions, sample_idx, test_labels):
    """Show all 12 leads with attribution overlay"""
    fig, axes = plt.subplots(3, 4, figsize=(18, 12))
    axes = axes.flatten()
    
    lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 
                  'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    
    for lead_idx in range(12):
        signal = test_data[sample_idx].numpy()[:, lead_idx]
        importance = attributions[sample_idx, :, lead_idx]
        
        importance_abs = np.abs(importance)
        if importance_abs.max() > importance_abs.min():
            importance_norm = (importance_abs - importance_abs.min()) / \
                             (importance_abs.max() - importance_abs.min())
        else:
            importance_norm = np.zeros_like(importance_abs)
        
        points = np.arange(len(signal))
        axes[lead_idx].scatter(points, signal, c=importance_norm, cmap='coolwarm', 
                              s=2, alpha=0.6, vmin=0, vmax=1)
        axes[lead_idx].plot(signal, color='gray', alpha=0.3, linewidth=0.5)
        axes[lead_idx].set_title(f"{lead_names[lead_idx]}", fontsize=11, fontweight='bold')
        axes[lead_idx].set_xticks([])
        axes[lead_idx].set_yticks([])
    
    plt.suptitle(f"Complete 12-Lead Attribution Analysis - Sample {sample_idx} (Class: {test_labels[sample_idx].item()})", 
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'demo_outputs/ecg_sample_{sample_idx}_all_leads.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n" + "="*70)
    print("📊 STEP 4: COMPREHENSIVE 12-LEAD VIEW")
    print("="*70)
    print("\n🗣️  SAY TO CARDIOLOGIST:")
    print("   'This shows all 12 leads simultaneously with AI attribution.'")
    print("   'Notice which leads have more red coloring - those drove the diagnosis.'")
    print("   'This gives you the complete picture of the AI\\'s reasoning.'")
    print("="*70)

def show_clinical_summary(test_data, attributions, sample_idx, test_labels, 
                         model, device):
    """Generate and display clinical interpretation"""
    sample = test_data[sample_idx:sample_idx+1].to(device)
    with torch.no_grad():
        logit, attn = model(sample, return_attention=True)
        pred = torch.sigmoid(logit).cpu().numpy()[0]
    
    pred_class = np.argmax(pred)
    confidence = pred[pred_class]
    true_label = test_labels[sample_idx].item()
    
    lead_imp = np.mean(np.abs(attributions[sample_idx]), axis=0)
    top_leads = np.argsort(lead_imp)[-3:][::-1]
    lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 
                  'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    
    temporal_imp = np.mean(np.abs(attributions[sample_idx]), axis=1)
    critical_windows = np.where(temporal_imp > np.percentile(temporal_imp, 95))[0]
    
    anatomical_regions = {
        0: 'Lateral', 1: 'Inferior', 2: 'Inferior', 3: 'Lateral',
        4: 'Lateral', 5: 'Inferior', 6: 'Septal', 7: 'Septal',
        8: 'Anterior', 9: 'Anterior', 10: 'Lateral', 11: 'Lateral'
    }
    
    print("\n" + "="*70)
    print("📋 STEP 5: CLINICAL INTERPRETATION SUMMARY")
    print("="*70)
    print(f"\n🏥 DIAGNOSIS:")
    print(f"   • AI Prediction: Class {pred_class} ({confidence*100:.1f}% confidence)")
    print(f"   • True Diagnosis: Class {true_label}")
    print(f"   • Status: {'✓ CORRECT' if pred_class == true_label else '✗ NEEDS REVIEW'}")
    
    print(f"\n🔬 KEY FINDINGS:")
    print(f"   Top 3 Most Important Leads:")
    for rank, lead_idx in enumerate(top_leads, 1):
        region = anatomical_regions[lead_idx]
        print(f"     {rank}. Lead {lead_names[lead_idx]} ({region} wall) - Importance: {lead_imp[lead_idx]:.6f}")
    
    print(f"\n⏱️  TEMPORAL PATTERNS:")
    print(f"   • Critical Time Windows: {len(critical_windows)} regions")
    print(f"   • Key Timesteps: {critical_windows[:10].tolist()}..." if len(critical_windows) > 10 else f"   • Key Timesteps: {critical_windows.tolist()}")
    
    print(f"\n💡 CLINICAL RELEVANCE:")
    involved_regions = list(set([anatomical_regions[i] for i in top_leads]))
    print(f"   The AI detected abnormalities primarily in: {', '.join(involved_regions)} wall(s)")
    
    print("\n🗣️  FINAL QUESTION TO CARDIOLOGIST:")
    print("   'Based on this explanation, would you:")
    print("    A) Trust this AI prediction as a second opinion?")
    print("    B) Want to see more cases before deciding?")
    print("    C) Have concerns about the AI\\'s reasoning?'")
    print("="*70)

# =============================================================================
# MASTER DEMONSTRATION FUNCTION
# =============================================================================

def run_complete_cardiologist_demo(test_data, test_labels, attributions, 
                                  model, device, sample_idx=0):
    """Execute complete step-by-step demonstration for cardiologist"""
    
    print("\n" + "="*70)
    print("🩺 STARTING CARDIOLOGIST DEMONSTRATION")
    print("="*70)
    print(f"Sample Selected: {sample_idx}")
    print("="*70)
    
    # Step 1: Show original ECG
    show_standard_12lead_ecg(test_data, sample_idx, test_labels)
    input("\n⏸️  Press Enter to continue to AI prediction...")
    
    # Step 2: Show AI prediction
    pred_class, confidence, is_correct = show_ai_prediction(
        test_data, sample_idx, test_labels, model, device
    )
    input("\n⏸️  Press Enter to see WHY the AI made this prediction...")
    
    # Step 3: Show XAI explanation
    show_attribution_explanation(test_data, attributions, sample_idx, 
                                 test_labels, top_n_leads=3)
    input("\n⏸️  Press Enter to see all 12 leads...")
    
    # Step 4: Show comprehensive view
    show_comprehensive_12lead_attribution(test_data, attributions, 
                                         sample_idx, test_labels)
    input("\n⏸️  Press Enter for clinical summary...")
    
    # Step 5: Clinical interpretation
    show_clinical_summary(test_data, attributions, sample_idx, 
                         test_labels, model, device)
    
    print("\n" + "="*70)
    print("✅ DEMONSTRATION COMPLETE")
    print("="*70)
    print("\n📁 All visualizations have been saved in demo_outputs/:")
    print(f"  • ecg_sample_{sample_idx}_original.png")
    print(f"  • ecg_sample_{sample_idx}_prediction.png")
    print(f"  • ecg_sample_{sample_idx}_xai_explanation.png")
    print(f"  • ecg_sample_{sample_idx}_all_leads.png")
    print("="*70)

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Load results
    test_data, test_labels, attributions, model, device = load_xai_results()
    
    if test_data is None:
        print("\n❌ Cannot proceed without XAI analysis results.")
        print("Run 'python xai_analysis.py' first, then try again.")
        exit(1)
    
    # =========================================================================
    # CONFIGURE WHICH SAMPLE TO DEMO
    # =========================================================================
    # Change this number to demo different samples (0 to 63)
    sample_to_demo = 0
    
    # Or automatically select interesting cases:
    # demo_samples = select_demonstration_sample(test_data, test_labels, model, device)
    # sample_to_demo = demo_samples['high_confidence_correct']  # or 'moderate_confidence_correct' or 'incorrect'
    
    # =========================================================================
    # RUN THE DEMO
    # =========================================================================
    run_complete_cardiologist_demo(
        test_data=test_data,
        test_labels=test_labels,
        attributions=attributions,
        model=model,
        device=device,
        sample_idx=sample_to_demo
    )
    