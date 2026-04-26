"""Compute correlation statistics for Structure C noise injection replica."""
import numpy as np
from scipy import stats

r_acc  = np.array([0.367, 0.207, 0.136, 0.001])
tl_mae = np.array([6.71, 8.83, 11.36, 15.79])

r, p = stats.pearsonr(r_acc, tl_mae)
rho, p_rho = stats.spearmanr(r_acc, tl_mae)
print(f'Pearson  r = {r:+.4f}, p = {p:.4f}')
print(f'Spearman rho = {rho:+.4f}, p = {p_rho:.4f}')

# Bootstrap CI on Pearson
rng = np.random.default_rng(0)
boot_rs = []
for _ in range(20000):
    idx = rng.integers(0, 4, 4)
    if r_acc[idx].std() == 0 or tl_mae[idx].std() == 0:
        continue
    boot_rs.append(stats.pearsonr(r_acc[idx], tl_mae[idx])[0])
boot_rs = np.array(boot_rs)
print(f'Bootstrap 95% CI: [{np.percentile(boot_rs, 2.5):+.3f}, {np.percentile(boot_rs, 97.5):+.3f}]')

# Fisher z CI
z = np.arctanh(r)
se = 1 / np.sqrt(4 - 3)  # n-3=1
lo, hi = np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)
print(f'Fisher z CI: [{lo:+.3f}, {hi:+.3f}]  (very wide because N=4)')
