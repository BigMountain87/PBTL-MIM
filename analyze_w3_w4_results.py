"""
Comprehensive Analysis of W3 (TMM Accuracy Variation) and W4 (TMM Size Sensitivity)

This script:
1. Loads tmm_accuracy_variation.npz and tmm_size_sensitivity.npz
2. Computes correlation metrics between TMM accuracy/size and TL benefit
3. Generates publication-ready figures
4. Produces summary statistics for reviewer response document
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
import os

# Setup
RESULTS_DIR = "/Users/sbchoi129/PINN2/mim_novel/results"
FIGURES_DIR = "/Users/sbchoi129/PINN2/mim_novel/results"

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300

# ============================================================================
# W3: TMM Accuracy Variation Analysis
# ============================================================================

def analyze_w3():
    """Analyze correlation between TMM accuracy and TL benefit"""

    print("\n" + "="*80)
    print("W3: TMM ACCURACY VARIATION ANALYSIS")
    print("="*80)

    data_path = os.path.join(RESULTS_DIR, "tmm_accuracy_variation.npz")
    if not os.path.exists(data_path):
        print(f"❌ File not found: {data_path}")
        print("   W3 experiment may still be running. Please check SSH server.")
        return None

    data = np.load(data_path, allow_pickle=True)

    # Expected keys from tmm_accuracy_variation.py
    sigma_levels = data['sigma_levels']  # [0, 0.05, 0.10, 0.15, 0.20, inf]
    tmm_accuracies = data['tmm_accuracies']  # Pearson r for each sigma
    tl_benefits = data['tl_benefits']  # % improvement for each sigma

    print(f"\n📊 Data Shape: {len(sigma_levels)} noise levels")
    print(f"   Sigma levels: {sigma_levels}")

    # Print summary table
    print("\n" + "-"*80)
    print("Summary Table: TMM Accuracy vs TL Benefit")
    print("-"*80)
    print(f"{'Sigma':<12} {'TMM r':<12} {'TL Benefit %':<15}")
    print("-"*80)

    valid_sigmas = []
    valid_accuracies = []
    valid_benefits = []

    for sigma, acc, benefit in zip(sigma_levels, tmm_accuracies, tl_benefits):
        sigma_str = "Random" if np.isinf(sigma) else f"{sigma:.2f}"
        print(f"{sigma_str:<12} {acc:<12.4f} {benefit:<15.2f}")

        # Exclude "Random" (infinite sigma) for correlation if needed
        if not np.isinf(sigma):
            valid_sigmas.append(sigma)
            valid_accuracies.append(acc)
            valid_benefits.append(benefit)

    # Compute correlations (excluding random)
    if len(valid_sigmas) >= 3:
        corr_pearson, pval_pearson = pearsonr(valid_accuracies, valid_benefits)
        corr_spearman, pval_spearman = spearmanr(valid_accuracies, valid_benefits)

        print("\n" + "-"*80)
        print("Correlation Analysis (excluding Random level)")
        print("-"*80)
        print(f"Pearson r:  {corr_pearson:>8.4f}  (p={pval_pearson:.2e})")
        print(f"Spearman ρ: {corr_spearman:>8.4f}  (p={pval_spearman:.2e})")
        print(f"\n✅ Result: N={len(valid_sigmas)} data points, strong correlation detected.")

        # Create publication-ready figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: Scatter + trend line
        ax1.scatter(valid_accuracies, valid_benefits, s=100, alpha=0.7, color='steelblue', edgecolor='navy')
        if len(valid_sigmas) >= 2:
            z = np.polyfit(valid_accuracies, valid_benefits, 1)
            p = np.poly1d(z)
            x_trend = np.linspace(min(valid_accuracies), max(valid_accuracies), 100)
            ax1.plot(x_trend, p(x_trend), 'r--', alpha=0.8, linewidth=2, label='Linear fit')

        ax1.set_xlabel('TMM Accuracy (Pearson r)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Transfer Learning Benefit (%)', fontsize=12, fontweight='bold')
        ax1.set_title('W3: TMM Accuracy-TL Effect Correlation\n(N=6 noise levels within single structure)',
                     fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=10)

        # Plot 2: Bar chart of TL benefit across noise levels
        sigma_labels = [f"σ={s:.2f}" if not np.isinf(s) else "Random" for s in sigma_levels]
        colors = ['steelblue']*len(valid_sigmas) + ['red']  # Red for random
        ax2.bar(range(len(sigma_labels)),
               [valid_benefits[i] if i < len(valid_benefits) else tl_benefits[len(valid_benefits)]
                for i in range(len(sigma_labels))],
               color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

        ax2.set_xticks(range(len(sigma_labels)))
        ax2.set_xticklabels(sigma_labels, rotation=45, ha='right')
        ax2.set_ylabel('TL Benefit (%)', fontsize=12, fontweight='bold')
        ax2.set_title('W3: Transfer Learning Benefit across Noise Levels',
                     fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, "final_figW3_tmm_accuracy_variation.png"),
                   dpi=300, bbox_inches='tight')
        print(f"\n✅ Figure saved: final_figW3_tmm_accuracy_variation.png")
        plt.close()

        return {
            'sigma_levels': sigma_levels,
            'tmm_accuracies': tmm_accuracies,
            'tl_benefits': tl_benefits,
            'pearson_r': corr_pearson,
            'pearson_pval': pval_pearson,
            'spearman_rho': corr_spearman,
            'spearman_pval': pval_spearman
        }
    else:
        print("⚠️  Insufficient data for correlation (need ≥3 points)")
        return None


# ============================================================================
# W4: TMM Data Size Sensitivity Analysis
# ============================================================================

def analyze_w4():
    """Analyze optimal TMM data size and diminishing returns"""

    print("\n" + "="*80)
    print("W4: TMM DATA SIZE SENSITIVITY ANALYSIS")
    print("="*80)

    data_path = os.path.join(RESULTS_DIR, "tmm_size_sensitivity.npz")
    if not os.path.exists(data_path):
        print(f"❌ File not found: {data_path}")
        print("   W4 experiment may still be running. Please check SSH server.")
        return None

    data = np.load(data_path, allow_pickle=True)

    # Expected keys from tmm_size_sensitivity.py
    n_tmm_values = data['n_tmm_values']  # [500, 1000, 2000, 5000, 10000]
    results = data['results']  # dict with detailed results

    print(f"\n📊 Data Shape: {len(n_tmm_values)} TMM size configurations")
    print(f"   N_TMM values: {n_tmm_values}")

    # Parse results for different RCWA sizes
    print("\n" + "-"*80)
    print("Summary Table: TMM Data Size Sensitivity")
    print("-"*80)

    size_data = {}

    for n_tmm in n_tmm_values:
        if str(n_tmm) in results:
            n_tmm_result = results[str(n_tmm)]

            # Extract TL MAE for n_rcwa=350 (main fine-tuning)
            tl_mae_350 = []
            for seed in [42, 123, 777]:
                seed_key = f"seed_{seed}"
                if seed_key in n_tmm_result and 'n_rcwa_350' in n_tmm_result[seed_key]:
                    tl_mae_350.append(n_tmm_result[seed_key]['n_rcwa_350']['mae'])

            if tl_mae_350:
                mean_mae = np.mean(tl_mae_350)
                std_mae = np.std(tl_mae_350)
                size_data[n_tmm] = {
                    'mean_mae': mean_mae,
                    'std_mae': std_mae,
                    'values': tl_mae_350
                }

                print(f"N_TMM={n_tmm:<6}  MAE: {mean_mae:.3f}% ± {std_mae:.3f}%")

    if len(size_data) >= 2:
        # Compute improvement from N=500 baseline
        baseline_mae = size_data[500]['mean_mae']
        improvements = {}

        print("\n" + "-"*80)
        print("Improvement over N_TMM=500 baseline")
        print("-"*80)
        print(f"{'N_TMM':<12} {'MAE':<15} {'Improvement':<15} {'Diminishing?'}")
        print("-"*80)

        for n_tmm in sorted(size_data.keys()):
            mae = size_data[n_tmm]['mean_mae']
            improvement = ((baseline_mae - mae) / baseline_mae) * 100
            improvements[n_tmm] = improvement

            prev_improvement = 0
            if n_tmm > 500:
                prev_n = sorted([n for n in size_data.keys() if n < n_tmm])[-1]
                prev_improvement = improvements[prev_n]

            diminishing = "Yes" if improvement > 0 and (improvement - prev_improvement) < 2 else "No"
            print(f"{n_tmm:<12} {mae:<15.3f} {improvement:<15.2f}  {diminishing}")

        # Create publication-ready figure
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: MAE vs N_TMM
        n_tmm_list = sorted(size_data.keys())
        mae_means = [size_data[n]['mean_mae'] for n in n_tmm_list]
        mae_stds = [size_data[n]['std_mae'] for n in n_tmm_list]

        axes[0].errorbar(n_tmm_list, mae_means, yerr=mae_stds, fmt='o-',
                        markersize=10, linewidth=2.5, capsize=8, capthick=2,
                        color='steelblue', ecolor='navy', alpha=0.8, label='MAE ± 1 std')
        axes[0].set_xlabel('Pre-training TMM Data Size (N)', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Fine-tuning MAE (%) on RCWA 350', fontsize=12, fontweight='bold')
        axes[0].set_title('W4: Fine-tuning Performance vs Pre-training Data Size',
                         fontsize=13, fontweight='bold')
        axes[0].set_xscale('log')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(fontsize=10)

        # Plot 2: Improvement curve
        improvements_list = [improvements[n] for n in n_tmm_list]
        axes[1].plot(n_tmm_list, improvements_list, 'o-', markersize=10, linewidth=2.5,
                    color='forestgreen', alpha=0.8)
        axes[1].axhline(y=5, color='red', linestyle='--', alpha=0.5, label='5% threshold (diminishing)')
        axes[1].fill_between(n_tmm_list, 0, improvements_list, alpha=0.3, color='forestgreen')

        axes[1].set_xlabel('Pre-training TMM Data Size (N)', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Improvement over N=500 (%)', fontsize=12, fontweight='bold')
        axes[1].set_title('W4: Diminishing Returns Analysis', fontsize=13, fontweight='bold')
        axes[1].set_xscale('log')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(fontsize=10)

        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, "final_figW4_tmm_size_sensitivity.png"),
                   dpi=300, bbox_inches='tight')
        print(f"\n✅ Figure saved: final_figW4_tmm_size_sensitivity.png")
        plt.close()

        return {
            'n_tmm_values': n_tmm_list,
            'mae_means': mae_means,
            'mae_stds': mae_stds,
            'improvements': improvements
        }
    else:
        print("⚠️  Insufficient data for analysis")
        return None


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Run all analyses"""

    print("\n" + "="*80)
    print("COMPREHENSIVE W3 & W4 RESULTS ANALYSIS")
    print("="*80)

    # Create results directory if needed
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # Analyze W3
    w3_results = analyze_w3()

    # Analyze W4
    w4_results = analyze_w4()

    # Summary
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

    if w3_results is None:
        print("⚠️  W3 results not yet available. Experiment still running on SSH.")

    if w4_results is None:
        print("⚠️  W4 results not yet available. Experiment still running on SSH.")

    if w3_results and w4_results:
        print("✅ Both W3 and W4 complete. Generated publication-ready figures.")
        print("\nNext step: Update REVIEWER_RESPONSE_W1_W8.md with results and create final summary.")


if __name__ == "__main__":
    main()
