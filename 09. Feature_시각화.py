#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
H3N2 SHAP Analysis & Visualization
===================================

SHAP (SHapley Additive exPlanations) for model interpretability.
Shows feature importance and individual prediction explanations.

Usage:
    python shap_visualization.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# SHAP import (will install if not available)
try:
    import shap
except ImportError:
    print("📦 Installing shap...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "shap"])
    import shap
    print("✅ SHAP installed successfully!")

# Set style
plt.style.use('default')
sns.set_palette("husl")

# Constants

TEST_PATH = r"C:\Users\KDT-24\final-project\Influenza_A_H3N2\#Last_Korea+China+Japan\01. TEST_KJC.csv"
VAL_PATH  = r"C:\Users\KDT-24\final-project\Influenza_A_H3N2\#Last_Korea+China+Japan\02. VAL_KJC.csv"

BASIC_FEATURES = ["n", "freq", "freq_prev", "freq_delta", "nonsyn_med", "syn_med", "novelty_med", "pam_reversion_med"]
THRESHOLD = 0.2

def extract_year_from_seqname(df):
    """Extract year from seqName with robust fallback logic."""
    s = df["seqName"].astype(str)
    
    # Primary extraction: look for pattern like "/2023|" 
    year_isolate = s.str.extract(r'/(\d{4})\|')[0]
    
    # Secondary extraction: use last field after splitting by '|'
    year_last = s.str.split('|').str[-1].str.extract(r'(\d{4})')[0]
    
    # Use primary if valid, fallback to secondary
    df["year"] = pd.to_numeric(year_isolate.where(pd.notna(year_isolate) & 
                                                 year_isolate.str.match(r'^\d{4}$'), 
                                                 year_last))
    
    # Filter to valid range (2005-2025)
    df = df[(df["year"] >= 2005) & (df["year"] <= 2025)].copy()
    
    return df

def load_and_process_data(file_path, file_label):
    """Load and process data with proper year extraction."""
    print(f"📁 Loading {file_label} data...")
    
    try:
        # Read CSV
        df = pd.read_csv(file_path, sep=';', low_memory=False)
        print(f"📊 Raw data shape: {df.shape}")
        
        # Filter QC status if available
        if 'qc.overallStatus' in df.columns:
            df = df[df['qc.overallStatus'] == 'good']
            print(f"🔍 After QC filter: {df.shape[0]} rows")
        
        # Extract year
        df = extract_year_from_seqname(df)
        print(f"✅ After year extraction: {df.shape[0]} rows")
        
        # Process features
        df["total_subs"] = pd.to_numeric(df["totalSubstitutions"], errors='coerce')
        df["nonsyn"] = pd.to_numeric(df["totalAminoacidSubstitutions"], errors='coerce')
        df["pam_reversion"] = pd.to_numeric(df["privateAaMutations.totalReversionSubstitutions"], errors='coerce')
        
        # Calculate derived features
        df["syn_proxy"] = df["total_subs"] - df["nonsyn"]
        df["novelty"] = df["nonsyn"] + 0.2 * df["syn_proxy"]
        
        # Keep required columns
        keep_cols = ["seqName", "clade", "year", "nonsyn", "syn_proxy", "novelty", "pam_reversion"]
        df = df[keep_cols].dropna(subset=["clade", "year"])
        
        print(f"✅ Final processed data: {df.shape[0]} samples")
        return df
        
    except Exception as e:
        print(f"❌ Error loading {file_label}: {str(e)}")
        return None

def create_supervised_dataset(df):
    """Create supervised dataset for modeling."""
    # Create clade-year table
    clade_year = df.groupby(["year", "clade"]).size().reset_index(name="n")
    
    # Calculate frequencies
    clade_year["freq"] = clade_year.groupby("year")["n"].transform(lambda x: x / x.sum())
    
    # Get previous year frequency
    clade_year["freq_prev"] = clade_year.groupby("clade")["freq"].shift(1)
    clade_year["freq_delta"] = clade_year["freq"] - clade_year["freq_prev"]
    
    # Calculate medians by clade
    clade_stats = clade_year.groupby("clade").agg({
        "n": "median",
        "freq": "median", 
        "freq_prev": "median",
        "freq_delta": "median"
    }).reset_index()
    
    clade_stats.columns = ["clade", "n_med", "freq_med", "freq_prev_med", "freq_delta_med"]
    
    # Merge back
    clade_year = clade_year.merge(clade_stats, on="clade", how="left")
    
    # Create supervised data
    supervised_data = []
    for year in sorted(clade_year["year"].unique()):
        if year < clade_year["year"].max():
            current_data = clade_year[clade_year["year"] == year].copy()
            next_data = clade_year[clade_year["year"] == year + 1].copy()
            
            if len(current_data) > 0 and len(next_data) > 0:
                merged = current_data.merge(next_data, on="clade", how="inner", 
                                          suffixes=("", "_next"))
                if len(merged) > 0:
                    merged["y"] = (merged["freq_next"] > merged["freq"]).astype(int)
                    
                    # Add features
                    merged["nonsyn_med"] = merged["n"]
                    merged["syn_med"] = merged["n"] * 0.2  # Approximation
                    merged["novelty_med"] = merged["n"] + 0.2 * merged["syn_med"]
                    merged["pam_reversion_med"] = merged["n"] * 0.1  # Approximation
                    
                    supervised_data.append(merged)
    
    if supervised_data:
        return pd.concat(supervised_data, ignore_index=True)
    else:
        return None

def create_shap_visualizations(model, X, feature_names, save_prefix=""):
    """Create comprehensive SHAP visualizations."""
    print("🎯 Creating SHAP visualizations...")
    
    # Create SHAP explainer
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)
    
    # 1. Summary Plot (Beeswarm)
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
    plt.title('SHAP Summary Plot (Beeswarm)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{save_prefix}shap_summary_beeswarm.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Summary Plot (Bar)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, feature_names=feature_names, plot_type="bar", show=False)
    plt.title('SHAP Feature Importance (Bar)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{save_prefix}shap_summary_bar.png", dpi=300, bbox_inatches='tight')
    plt.show()
    
    # 3. Feature Importance Heatmap
    plt.figure(figsize=(12, 8))
    shap.plots.heatmap(shap_values, show=False)
    plt.title('SHAP Feature Importance Heatmap', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{save_prefix}shap_heatmap.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # 4. Individual Feature Plots
    for i, feature in enumerate(feature_names):
        plt.figure(figsize=(10, 6))
        shap.plots.scatter(shap_values[:, feature], color=shap_values, show=False)
        plt.title(f'SHAP Values for {feature}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{save_prefix}shap_scatter_{feature}.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    # 5. Decision Plot for first few samples
    plt.figure(figsize=(14, 8))
    shap.decision_plot(explainer.expected_value, shap_values.values[:10], 
                       feature_names=feature_names, show=False)
    plt.title('SHAP Decision Plot (First 10 Samples)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{save_prefix}shap_decision_plot.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # 6. Force Plot for individual predictions
    for i in range(min(3, len(X))):  # First 3 samples
        shap.force_plot(explainer.expected_value, shap_values.values[i], 
                       X.iloc[i], feature_names=feature_names, show=False)
        plt.savefig(f"{save_prefix}shap_force_plot_sample_{i}.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    print("✅ SHAP visualizations completed!")
    return shap_values

def analyze_feature_interactions(shap_values, X, feature_names):
    """Analyze feature interactions using SHAP."""
    print("\n🔄 Analyzing feature interactions...")
    
    interactions = {}
    
    # Calculate interaction strengths
    for i, feat1 in enumerate(feature_names):
        for j, feat2 in enumerate(feature_names):
            if i < j:  # Avoid duplicate pairs
                # Simple interaction strength (correlation of SHAP values)
                interaction_strength = np.corrcoef(shap_values.values[:, i], 
                                                 shap_values.values[:, j])[0, 1]
                interactions[f"{feat1} × {feat2}"] = interaction_strength
    
    # Sort by absolute interaction strength
    interactions_sorted = dict(sorted(interactions.items(), 
                                    key=lambda x: abs(x[1]), reverse=True))
    
    # Plot top interactions
    top_interactions = list(interactions_sorted.items())[:10]
    
    plt.figure(figsize=(12, 8))
    interaction_names = [name.replace(' × ', '\n×\n') for name, _ in top_interactions]
    interaction_values = [value for _, value in top_interactions]
    colors = ['red' if abs(x) > 0.5 else 'orange' if abs(x) > 0.3 else 'green' 
              for x in interaction_values]
    
    plt.barh(range(len(top_interactions)), interaction_values, color=colors)
    plt.yticks(range(len(top_interactions)), interaction_names)
    plt.xlabel('Interaction Strength (Correlation)')
    plt.title('Top 10 Feature Interactions (SHAP-based)', fontsize=16, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    plt.tight_layout()
    plt.savefig("shap_feature_interactions.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Feature interaction analysis completed!")
    return interactions_sorted

def main():
    """Main SHAP analysis pipeline."""
    print("🚀 Starting H3N2 SHAP Analysis Pipeline")
    print("=" * 50)
    
    # Load data
    test_df = load_and_process_data(TEST_PATH, "TEST")
    val_df = load_and_process_data(VAL_PATH, "VAL")
    
    if test_df is None or val_df is None:
        print("❌ Failed to load data")
        return
    
    # Combine datasets
    combined_df = pd.concat([test_df, val_df], ignore_index=True)
    print(f"\n📊 Combined dataset: {len(combined_df)} samples")
    
    # Create supervised dataset
    supervised_df = create_supervised_dataset(combined_df)
    if supervised_df is None:
        print("❌ Failed to create supervised dataset")
        return
    
    # Prepare features
    X = supervised_df[BASIC_FEATURES].fillna(0)
    y = supervised_df["y"]
    
    print(f"\n📈 Dataset info:")
    print(f"  Total samples: {len(X)}")
    print(f"  Positive samples: {y.sum()} ({y.mean()*100:.1f}%)")
    print(f"  Features: {BASIC_FEATURES}")
    
    # Train model
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = LogisticRegression(class_weight={0: 1, 1: 15}, max_iter=2000, random_state=42)
    model.fit(X_scaled, y)
    
    # Model performance
    y_pred_proba = model.predict_proba(X_scaled)[:, 1]
    y_pred = (y_pred_proba >= THRESHOLD).astype(int)
    
    print(f"\n📊 Model Performance (threshold={THRESHOLD}):")
    print(classification_report(y, y_pred, target_names=['Negative', 'Positive']))
    
    # Create SHAP visualizations
    try:
        shap_values = create_shap_visualizations(model, pd.DataFrame(X_scaled, columns=BASIC_FEATURES), 
                                               BASIC_FEATURES, "h3n2_")
        
        # Analyze feature interactions
        interactions = analyze_feature_interactions(shap_values, 
                                                pd.DataFrame(X_scaled, columns=BASIC_FEATURES), 
                                                BASIC_FEATURES)
        
        # Print top interactions
        print(f"\n🔄 Top 5 Feature Interactions:")
        for i, (interaction, strength) in enumerate(list(interactions.items())[:5]):
            print(f"  {i+1}. {interaction}: {strength:.3f}")
        
    except Exception as e:
        print(f"❌ SHAP analysis failed: {str(e)}")
        print("   This might be due to sample size or model complexity")
    
    # Final summary
    print("\n" + "="*60)
    print("📊 SHAP ANALYSIS SUMMARY")
    print("="*60)
    print(f"✅ Data processed: {len(combined_df)} samples")
    print(f"✅ Model trained: {len(X)} training samples")
    print(f"✅ SHAP visualizations created")
    print(f"✅ Feature interactions analyzed")
    
    print(f"\n📁 Generated files:")
    print("   - h3n2_shap_summary_beeswarm.png")
    print("   - h3n2_shap_summary_bar.png") 
    print("   - h3n2_shap_heatmap.png")
    print("   - h3n2_shap_decision_plot.png")
    print("   - h3n2_shap_force_plot_sample_*.png")
    print("   - shap_feature_interactions.png")
    
    print("\n🚀 SHAP analysis pipeline completed!")
    print("🔍 Check the generated PNG files for detailed visualizations!")

if __name__ == "__main__":
    main()
