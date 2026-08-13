"""
W3 Response: Statistical tests for all pairwise comparisons
Paired t-test + Wilcoxon signed-rank + Cohen's d
"""
import numpy as np
from scipy import stats

def cohens_d(x, y):
    """Paired Cohen's d"""
    diff = x - y
    return np.mean(diff) / np.std(diff, ddof=1)

def pairwise_test(name_a, vals_a, name_b, vals_b, n_label):
    diff = vals_a - vals_b
    t_stat, t_p = stats.ttest_rel(vals_a, vals_b)
    try:
        w_stat, w_p = stats.wilcoxon(vals_a, vals_b)
    except:
        w_stat, w_p = float('nan'), float('nan')
    d = cohens_d(vals_a, vals_b)
    mean_diff = np.mean(diff) * 100
    sig = "***" if t_p < 0.001 else "**" if t_p < 0.01 else "*" if t_p < 0.05 else "ns"
    print(f'  {name_a} vs {name_b} (n={n_label}): '
          f'delta={mean_diff:+.2f}pp, '
          f't={t_stat:.3f}, p_t={t_p:.4f}, '
          f'p_w={w_p:.4f}, '
          f'd={d:.3f} {sig}')
    return t_p, w_p, d

print("=" * 80)
print("STATISTICAL TESTS FOR ALL KEY PAIRWISE COMPARISONS")
print("=" * 80)

# Structure A
print("\n### Structure A ###")
data_a = np.load('results/pbtl_A_10seed.npz')
for n in [50, 100, 200, 350]:
    m0 = data_a[f'{n}_M0']
    mphys = data_a[f'{n}_M_phys']
    mtl = data_a[f'{n}_M_TL']
    mtlphys = data_a[f'{n}_M_TL+phys']
    print(f'\n  n={n}:')
    pairwise_test('M0', m0, 'M_phys', mphys, n)
    pairwise_test('M0', m0, 'M_TL', mtl, n)
    pairwise_test('M0', m0, 'M_TL+phys', mtlphys, n)
    pairwise_test('M_phys', mphys, 'M_TL', mtl, n)
    pairwise_test('M_phys', mphys, 'M_TL+phys', mtlphys, n)
    pairwise_test('M_TL', mtl, 'M_TL+phys', mtlphys, n)

# Structure B
print("\n\n### Structure B ###")
data_b = np.load('results/pbtl_B_10seed.npz')
for i, n in enumerate([50, 100, 200, 350]):
    m0 = data_b['M0'][i]
    mphys = data_b['M_phys'][i]
    mtl = data_b['M_TL'][i]
    mtlphys = data_b['M_TL_phys'][i]
    print(f'\n  n={n}:')
    pairwise_test('M0', m0, 'M_phys', mphys, n)
    pairwise_test('M0', m0, 'M_TL', mtl, n)
    pairwise_test('M0', m0, 'M_TL+phys', mtlphys, n)
    pairwise_test('M_phys', mphys, 'M_TL', mtl, n)
    pairwise_test('M_phys', mphys, 'M_TL+phys', mtlphys, n)
    pairwise_test('M_TL', mtl, 'M_TL+phys', mtlphys, n)

# Structure C
print("\n\n### Structure C ###")
data_c = np.load('results/pbtl_C_v2_10seed.npz')
for i, n in enumerate([50, 100, 200, 350]):
    m0 = data_c['M0'][i]
    mphys = data_c['M_phys'][i]
    mtl = data_c['M_TL'][i]
    mtlphys = data_c['M_TL_phys'][i]
    print(f'\n  n={n}:')
    pairwise_test('M0', m0, 'M_phys', mphys, n)
    pairwise_test('M0', m0, 'M_TL', mtl, n)
    pairwise_test('M0', m0, 'M_TL+phys', mtlphys, n)
    pairwise_test('M_phys', mphys, 'M_TL', mtl, n)
    pairwise_test('M_phys', mphys, 'M_TL+phys', mtlphys, n)
    pairwise_test('M_TL', mtl, 'M_TL+phys', mtlphys, n)

print('\n\nDone!')
