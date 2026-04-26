"""Summarize all experimental results into one report."""
import numpy as np
import os

R = '/Users/sbchoi129/PINN2/mim_novel/results'

print('=' * 78)
print('  ALL EXPERIMENT RESULTS SUMMARY')
print('=' * 78)

# 1. W3 — TMM accuracy variation (Structure A)
print('\n[1] W3: TMM accuracy variation, Structure A (paper sec:tmm_accuracy)')
d = np.load(f'{R}/tmm_accuracy_variation.npz', allow_pickle=True)
print(f'   keys: {list(d.files)}')

# 2. W4 — TMM data size sensitivity
print('\n[2] W4: TMM pre-training data size sensitivity')
d = np.load(f'{R}/tmm_size_sensitivity.npz', allow_pickle=True)
print(f'   keys: {list(d.files)}')

# 3. Co-Kriging sensitivity (4 kernels)
print('\n[3] Co-Kriging kernel sensitivity (Apr 21)')
d = np.load(f'{R}/cokriging_sensitivity.npz', allow_pickle=True)
print(f'   keys: {list(d.files)[:8]}')

# 4. Deep MF-NN baseline
print('\n[4] Deep composite MF-NN baseline (Apr 21)')
d = np.load(f'{R}/deep_mfnn_baseline.npz', allow_pickle=True)
print(f'   keys: {list(d.files)}')
sizes = d['sizes']; mae = d['mae']
for i, n in enumerate(sizes):
    print(f'   n={int(n)}: MAE = {mae[i].mean()*100:.2f}±{mae[i].std()*100:.2f}%  (3 seeds)')

# 5. Structure C noise injection replica
print('\n[5] Structure C noise injection replica (Apr 24)')
d = np.load(f'{R}/noise_injection_C.npz', allow_pickle=True)
print(f'   keys: {list(d.files)}')
levels = d['level_names']
sigmas = d['sigmas']
rs     = d['r_tmm_vs_rcwa']
mae_per_seed = d['tl_mae_per_seed']
print(f"   {'Level':<14} | {'sigma':>5} | {'r':>8} | {'TL MAE (%)':>14}")
for i in range(len(levels)):
    m = mae_per_seed[i].mean(); s = mae_per_seed[i].std()
    sig = sigmas[i] if sigmas[i] >= 0 else float('inf')
    print(f"   {levels[i]:<14} | {sig:>5.2f} | {rs[i]:>+8.3f} | {m:>5.2f}±{s:.2f}%")

# 6. Sparse Co-Kriging N_TMM sweep
print('\n[6] Sparse Co-Kriging N_TMM sweep (Apr 24)')
d = np.load(f'{R}/cokriging_ntmm_sweep.npz', allow_pickle=True)
sizes = d['sizes']; n_tmms = d['n_tmms']; mae = d['mae']
print(f"   {'n_rcwa':>8} | " + ' | '.join([f'N_TMM={int(t):<5}' for t in n_tmms]))
for i, n in enumerate(sizes):
    row = [f'{int(n):>8}']
    for j, nt in enumerate(n_tmms):
        m = mae[i, j].mean()*100; s = mae[i, j].std()*100
        row.append(f'{m:>5.2f}±{s:.2f}%')
    print('   ' + ' | '.join(row))

# 7. Material generalization B (Au + Structure B)
print('\n[7] Material generalization: Au + Structure B (Apr 25)')
d = np.load(f'{R}/material_generalization_B.npz', allow_pickle=True)
sizes = d['train_sizes']
for n in sizes:
    nt = int(n)
    M0  = d[f'{nt}_M0'].mean();    M0s  = d[f'{nt}_M0'].std()
    Mp  = d[f'{nt}_M_phys'].mean(); Mps = d[f'{nt}_M_phys'].std()
    Mt  = d[f'{nt}_M_TL'].mean();  Mts  = d[f'{nt}_M_TL'].std()
    Mtp = d[f'{nt}_M_TL+phys'].mean(); Mtps = d[f'{nt}_M_TL+phys'].std()
    tl_b = (1 - Mt/M0) * 100
    print(f'   n={nt:>3d}: M0={M0:.2f}±{M0s:.2f}%, M_phys={Mp:.2f}±{Mps:.2f}%, '
          f'M_TL={Mt:.2f}±{Mts:.2f}%, M_TL+phys={Mtp:.2f}±{Mtps:.2f}%  '
          f'[TL benefit {tl_b:+.1f}%]')

# 8. Full-Spectrum baseline
print('\n[8] Full-Spectrum ResNet vs per-wavelength (Apr 25)')
d = np.load(f'{R}/full_spectrum_baseline.npz', allow_pickle=True)
sizes = d['sizes']; m0_fs = d['m0_fs']; mtl_fs = d['m_tl_fs']
for i, n in enumerate(sizes):
    M0  = m0_fs[i].mean()*100;  M0s  = m0_fs[i].std()*100
    Mtl = mtl_fs[i].mean()*100; Mtls = mtl_fs[i].std()*100
    tl_b = (1 - Mtl/M0) * 100
    print(f'   n={int(n):>3d}: M0_FS={M0:.2f}±{M0s:.2f}%, '
          f'M_TL_FS={Mtl:.2f}±{Mtls:.2f}%  [TL benefit {tl_b:+.1f}%]')

# 9. Phase 3: 241-sample test
print('\n[9] Phase 3: Extended 241-sample test set (Apr 26)')
d = np.load(f'{R}/phase3_expanded_test.npz', allow_pickle=True)
print(f'   keys: {list(d.files)}')
m0_per = d['m0_per_seed'].item()  # dict
mtl_per = d['mtl_per_seed'].item()
for n in [100, 350]:
    M0   = m0_per[n].mean()*100;   M0s = m0_per[n].std()*100
    Mtl  = mtl_per[n].mean()*100;  Mtls = mtl_per[n].std()*100
    print(f'   n={n}: M0={M0:.2f}±{M0s:.2f}%, M_TL+phys={Mtl:.2f}±{Mtls:.2f}%  '
          f'[improvement {(1-Mtl/M0)*100:+.1f}%]')

print('\n' + '=' * 78)
print('  Done.')
print('=' * 78)
