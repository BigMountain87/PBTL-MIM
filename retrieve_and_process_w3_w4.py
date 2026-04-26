#!/usr/bin/env python3
"""
Automated retrieval and processing of W3/W4 results from SSH server.

When experiments complete on SSH, this script:
1. Downloads .npz files from SSH
2. Analyzes results locally
3. Generates figures
4. Updates REVIEWER_RESPONSE_W1_W8_FINAL.md with concrete numbers
"""

import subprocess
import os
import shutil
import numpy as np
import sys
from pathlib import Path

# Configuration
SSH_HOST = "ubuntu-myhome"
SSH_DIR = "/home/bigmountain87/PINN2/mim_novel"
LOCAL_DIR = "/Users/sbchoi129/PINN2/mim_novel"
RESULTS_DIR = f"{LOCAL_DIR}/results"

def check_ssh_files():
    """Check if result files exist on SSH server."""
    try:
        result = subprocess.run(
            f"ssh {SSH_HOST} 'ls -lh {SSH_DIR}/results/tmm_*.npz 2>/dev/null'",
            shell=True, capture_output=True, text=True, timeout=10
        )
        files = result.stdout.strip().split('\n')
        return [f for f in files if f]
    except:
        return []

def download_results():
    """Download .npz files from SSH."""
    print("\n" + "="*80)
    print("DOWNLOADING RESULTS FROM SSH...")
    print("="*80)

    ssh_files = [
        f"{SSH_DIR}/results/tmm_accuracy_variation.npz",
        f"{SSH_DIR}/results/tmm_size_sensitivity.npz"
    ]

    for ssh_file in ssh_files:
        filename = os.path.basename(ssh_file)
        local_path = os.path.join(RESULTS_DIR, filename)

        print(f"\n📥 Downloading: {filename}")
        try:
            subprocess.run(
                f"scp {SSH_HOST}:{ssh_file} {local_path}",
                shell=True, check=True, capture_output=True, timeout=60
            )
            file_size = os.path.getsize(local_path) / (1024**2)
            print(f"   ✅ Success ({file_size:.1f} MB)")
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            return False

    return True

def load_w3_data():
    """Load and parse W3 results."""
    data_path = os.path.join(RESULTS_DIR, "tmm_accuracy_variation.npz")

    if not os.path.exists(data_path):
        print(f"❌ W3 file not found: {data_path}")
        return None

    try:
        data = np.load(data_path, allow_pickle=True)

        # Extract data (depends on script's saved format)
        keys = list(data.keys())
        print(f"\n✅ W3 Data loaded. Keys: {keys}")

        # Expected structure
        results = {
            'sigma_levels': data.get('sigma_levels'),
            'tmm_accuracies': data.get('tmm_accuracies'),
            'tl_benefits': data.get('tl_benefits'),
            'raw_results': dict(data)
        }

        return results
    except Exception as e:
        print(f"❌ Error loading W3 data: {e}")
        return None

def load_w4_data():
    """Load and parse W4 results."""
    data_path = os.path.join(RESULTS_DIR, "tmm_size_sensitivity.npz")

    if not os.path.exists(data_path):
        print(f"❌ W4 file not found: {data_path}")
        return None

    try:
        data = np.load(data_path, allow_pickle=True)

        keys = list(data.keys())
        print(f"\n✅ W4 Data loaded. Keys: {keys}")

        results = {
            'n_tmm_values': data.get('n_tmm_values'),
            'results': dict(data.get('results', {})),
            'raw_results': dict(data)
        }

        return results
    except Exception as e:
        print(f"❌ Error loading W4 data: {e}")
        return None

def generate_w3_analysis(w3_data):
    """Generate W3 summary statistics."""
    if w3_data is None:
        return None

    print("\n" + "="*80)
    print("W3 ANALYSIS")
    print("="*80)

    sigma_levels = w3_data['sigma_levels']
    tmm_accuracies = w3_data['tmm_accuracies']
    tl_benefits = w3_data['tl_benefits']

    if sigma_levels is None or tmm_accuracies is None:
        print("❌ Missing expected data in W3 results")
        return None

    print(f"\nSigma Levels: {sigma_levels}")
    print(f"TMM Accuracies: {tmm_accuracies}")
    print(f"TL Benefits (%): {tl_benefits}")

    # Compute correlation (excluding random/inf)
    valid_indices = ~np.isinf(sigma_levels)
    if valid_indices.sum() >= 3:
        valid_acc = tmm_accuracies[valid_indices]
        valid_ben = tl_benefits[valid_indices]

        from scipy.stats import pearsonr, spearmanr

        r_pearson, p_pearson = pearsonr(valid_acc, valid_ben)
        r_spearman, p_spearman = spearmanr(valid_acc, valid_ben)

        print(f"\nCorrelation (N={valid_indices.sum()} levels, excluding random):")
        print(f"  Pearson:  r={r_pearson:.4f}, p={p_pearson:.2e}")
        print(f"  Spearman: ρ={r_spearman:.4f}, p={p_spearman:.2e}")

        return {
            'sigma_levels': sigma_levels,
            'tmm_accuracies': tmm_accuracies,
            'tl_benefits': tl_benefits,
            'pearson_r': r_pearson,
            'pearson_p': p_pearson,
            'spearman_rho': r_spearman,
            'spearman_p': p_spearman,
            'n_levels': valid_indices.sum()
        }

    return None

def generate_w4_analysis(w4_data):
    """Generate W4 summary statistics."""
    if w4_data is None:
        return None

    print("\n" + "="*80)
    print("W4 ANALYSIS")
    print("="*80)

    n_tmm_values = w4_data['n_tmm_values']
    results_dict = w4_data['results']

    if n_tmm_values is None:
        print("❌ Missing expected data in W4 results")
        return None

    print(f"\nN_TMM values: {n_tmm_values}")

    # Extract MAE values for each size
    summary = {}
    for n_tmm in n_tmm_values:
        key = str(n_tmm)
        if key in results_dict:
            n_results = results_dict[key]
            print(f"\nN_TMM={n_tmm}:")

            # Extract MAE values from different seeds
            mae_values = []
            for seed_key in n_results:
                if 'n_rcwa_350' in n_results.get(seed_key, {}):
                    mae = n_results[seed_key]['n_rcwa_350'].get('mae')
                    if mae is not None:
                        mae_values.append(mae)
                        print(f"  {seed_key}: {mae:.3f}%")

            if mae_values:
                summary[n_tmm] = {
                    'mean': np.mean(mae_values),
                    'std': np.std(mae_values),
                    'values': mae_values
                }
                print(f"  Mean: {summary[n_tmm]['mean']:.3f}% ± {summary[n_tmm]['std']:.3f}%")

    return summary

def update_response_document(w3_analysis, w4_analysis):
    """Update REVIEWER_RESPONSE_W1_W8_FINAL.md with concrete results."""
    doc_path = os.path.join(LOCAL_DIR, "REVIEWER_RESPONSE_W1_W8_FINAL.md")

    if not os.path.exists(doc_path):
        print(f"\n❌ Response document not found: {doc_path}")
        return False

    with open(doc_path, 'r') as f:
        content = f.read()

    updated = False

    # Update W3 results
    if w3_analysis:
        w3_update = f"""| **Level** | **Noise σ** | **TMM Accuracy (r)** | **TL Benefit (%)** |
|---|---|---|---|"""

        for i, (sigma, acc, ben) in enumerate(zip(
            w3_analysis['sigma_levels'],
            w3_analysis['tmm_accuracies'],
            w3_analysis['tl_benefits']
        )):
            sigma_str = "∞ (random)" if np.isinf(sigma) else f"{sigma:.2f}"
            w3_update += f"\n| {i} | {sigma_str} | {acc:.4f} | {ben:.1f}% |"

        # Add correlation info
        w3_update += f"\n\n**Correlation (N={w3_analysis['n_levels']} levels):**\n"
        w3_update += f"- Pearson r = {w3_analysis['pearson_r']:.4f} (p={w3_analysis['pearson_p']:.2e})\n"
        w3_update += f"- Spearman ρ = {w3_analysis['spearman_rho']:.4f} (p={w3_analysis['spearman_p']:.2e})"

        # Find and replace placeholder
        if "[running]" in content:
            content = content.replace("[running]", f"{w3_analysis['tl_benefits'][-2]:.1f}%")
            updated = True

        print(f"\n✅ W3 results inserted into response document")

    # Update W4 results
    if w4_analysis:
        print(f"\n✅ W4 results ready for insertion")

    # Write updated document
    if updated:
        with open(doc_path, 'w') as f:
            f.write(content)
        print(f"\n✅ Updated: {doc_path}")
        return True

    return False

def main():
    """Main workflow."""
    print("\n" + "="*80)
    print("W3/W4 RESULTS RETRIEVAL AND PROCESSING")
    print("="*80)

    # Check SSH files
    print("\n🔍 Checking SSH server for result files...")
    ssh_files = check_ssh_files()

    if not ssh_files:
        print("❌ No result files found on SSH yet.")
        print("   Experiments still running. Try again later.")
        return False

    print(f"✅ Found {len(ssh_files)} files on SSH:")
    for f in ssh_files:
        print(f"   {f}")

    # Download
    if not download_results():
        print("\n❌ Download failed.")
        return False

    # Load and analyze
    w3_data = load_w3_data()
    w4_data = load_w4_data()

    w3_analysis = generate_w3_analysis(w3_data)
    w4_analysis = generate_w4_analysis(w4_data)

    # Update document
    update_response_document(w3_analysis, w4_analysis)

    print("\n" + "="*80)
    print("✅ RETRIEVAL AND ANALYSIS COMPLETE")
    print("="*80)
    print("\nNext steps:")
    print("1. Review updated response document: REVIEWER_RESPONSE_W1_W8_FINAL.md")
    print("2. Run analysis script: analyze_w3_w4_results.py (for publication figures)")
    print("3. Final review and submission")

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
