#!/usr/bin/env python3
"""
Analyze LR ablation results to determine if TL benefit persists across learning rates.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Load results
data = np.load("results/lr_ablation_A.npz", allow_pickle=True)
train_sizes = data["train_sizes"]
conditions = data["conditions"]
print(f"Train sizes: {train_sizes}")
print(f"Conditions: {conditions}")

# Extract results
results = {}
for sz in train_sizes:
    results[sz] = {}
    for cond in conditions:
        key = f"{sz}_{cond}"
        results[sz][cond] = np.array(data[key]) * 100  # Convert to %

# Print summary
print("\n" + "=" * 90)
print("LR ABLATION SUMMARY (Test MAE %)")
print("=" * 90)
header = f"{'n':>6}"
for cond in conditions:
    header += f" | {cond:>16s}"
print(header)
print("-" * 90)

for sz in train_sizes:
    row = f"{sz:>6}"
    for cond in conditions:
        v = results[sz][cond]
        row += f" | {v.mean():>7.2f}±{v.std():>5.2f}"
    print(row)

print("\n" + "=" * 90)
print("KEY FINDINGS")
print("=" * 90)

for sz in train_sizes:
    m0_orig = results[sz]["M0_lr1e-3"].mean()
    m0_low = results[sz]["M0_lr3e-4"].mean()
    tl_orig = results[sz]["MTL_lr3e-4"].mean()
    tl_high = results[sz]["MTL_lr1e-3"].mean()

    print(f"\nn={sz}:")
    print(f"  (1) LR effect on M0 (scratch):")
    print(f"      M0 lr=1e-3: {m0_orig:.2f}% vs M0 lr=3e-4: {m0_low:.2f}% → Δ = {m0_orig-m0_low:+.2f}%")
    print(f"  (2) LR effect on M_TL (transfer):")
    print(f"      M_TL lr=3e-4: {tl_orig:.2f}% vs M_TL lr=1e-3: {tl_high:.2f}% → Δ = {tl_orig-tl_high:+.2f}%")
    print(f"  (3) TL benefit when both use lr=1e-3 (same):")
    print(f"      M0: {m0_orig:.2f}% vs M_TL: {tl_high:.2f}% → TL gain = {m0_orig-tl_high:+.2f}%")
    print(f"  (4) TL benefit when both use lr=3e-4 (same):")
    print(f"      M0: {m0_low:.2f}% vs M_TL: {tl_orig:.2f}% → TL gain = {m0_low-tl_orig:+.2f}%")

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle("Learning Rate Ablation: Isolating TL Benefit from LR Effect", fontsize=14, fontweight='bold')

# Plot 1: M0 performance vs LR
ax = axes[0, 0]
m0_lr_high = [results[sz]["M0_lr1e-3"].mean() for sz in train_sizes]
m0_lr_low = [results[sz]["M0_lr3e-4"].mean() for sz in train_sizes]
x = np.arange(len(train_sizes))
width = 0.35
ax.bar(x - width/2, m0_lr_high, width, label="lr=1e-3 (original)", alpha=0.8)
ax.bar(x + width/2, m0_lr_low, width, label="lr=3e-4 (TL's lr)", alpha=0.8)
ax.set_xlabel("Training size (n)")
ax.set_ylabel("Test MAE (%)")
ax.set_title("M0 (Scratch): LR Effect")
ax.set_xticks(x)
ax.set_xticklabels(train_sizes)
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Plot 2: M_TL performance vs LR
ax = axes[0, 1]
tl_lr_low = [results[sz]["MTL_lr3e-4"].mean() for sz in train_sizes]
tl_lr_high = [results[sz]["MTL_lr1e-3"].mean() for sz in train_sizes]
x = np.arange(len(train_sizes))
ax.bar(x - width/2, tl_lr_low, width, label="lr=3e-4 (original)", alpha=0.8)
ax.bar(x + width/2, tl_lr_high, width, label="lr=1e-3 (scratch's lr)", alpha=0.8)
ax.set_xlabel("Training size (n)")
ax.set_ylabel("Test MAE (%)")
ax.set_title("M_TL (Transfer): LR Effect")
ax.set_xticks(x)
ax.set_xticklabels(train_sizes)
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Plot 3: TL benefit at matching LRs
ax = axes[1, 0]
tl_benefit_same_high = [results[sz]["M0_lr1e-3"].mean() - results[sz]["MTL_lr1e-3"].mean() for sz in train_sizes]
tl_benefit_same_low = [results[sz]["M0_lr3e-4"].mean() - results[sz]["MTL_lr3e-4"].mean() for sz in train_sizes]
x = np.arange(len(train_sizes))
ax.bar(x - width/2, tl_benefit_same_high, width, label="When lr=1e-3 (both)", alpha=0.8, color='green')
ax.bar(x + width/2, tl_benefit_same_low, width, label="When lr=3e-4 (both)", alpha=0.8, color='orange')
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax.set_xlabel("Training size (n)")
ax.set_ylabel("TL Benefit (% improvement)")
ax.set_title("TL Benefit with Matched Learning Rates")
ax.set_xticks(x)
ax.set_xticklabels(train_sizes)
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Plot 4: Summary comparison
ax = axes[1, 1]
# Plot all 4 conditions for n=50 as example
n_idx = 0
sz = train_sizes[n_idx]
conds_plot = conditions
means = [results[sz][c].mean() for c in conds_plot]
stds = [results[sz][c].std() for c in conds_plot]
x = np.arange(len(conds_plot))
colors = ['#1f77b4', '#1f77b4', '#ff7f0e', '#ff7f0e']
ax.bar(x, means, yerr=stds, capsize=5, color=colors, alpha=0.7)
ax.set_ylabel("Test MAE (%)")
ax.set_title(f"All Conditions at n={sz}")
ax.set_xticks(x)
ax.set_xticklabels([c.replace('_', '\n') for c in conds_plot], fontsize=9)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig("results/fig_w4_lr_ablation.png", dpi=150, bbox_inches='tight')
print("\n✓ Saved: results/fig_w4_lr_ablation.png")

# Statistical test: Is TL benefit robust to LR?
print("\n" + "=" * 90)
print("STATISTICAL TEST: Is TL benefit robust across learning rates?")
print("=" * 90)

from scipy import stats

for sz in train_sizes:
    m0_orig = results[sz]["M0_lr1e-3"]
    tl_high = results[sz]["MTL_lr1e-3"]
    m0_low = results[sz]["M0_lr3e-4"]
    tl_orig = results[sz]["MTL_lr3e-4"]

    # Test 1: TL benefit at lr=1e-3
    t_stat1, p_val1 = stats.ttest_rel(m0_orig, tl_high)
    # Test 2: TL benefit at lr=3e-4
    t_stat2, p_val2 = stats.ttest_rel(m0_low, tl_orig)

    print(f"\nn={sz}:")
    print(f"  TL @ lr=1e-3: t={t_stat1:.2f}, p={p_val1:.4f} {'***' if p_val1<0.001 else '**' if p_val1<0.01 else '*' if p_val1<0.05 else 'ns'}")
    print(f"  TL @ lr=3e-4: t={t_stat2:.2f}, p={p_val2:.4f} {'***' if p_val2<0.001 else '**' if p_val2<0.01 else '*' if p_val2<0.05 else 'ns'}")

print("\nConclusion: If both p-values < 0.05, TL benefit is robust across learning rates.")
print("            This proves TL benefit is NOT merely due to LR choice.")
