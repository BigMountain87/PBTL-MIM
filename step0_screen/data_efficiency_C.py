#!/usr/bin/env python3
"""Data efficiency experiment: M0 vs M_phys for Structure C (Dual-Polarization)."""
import sys, os
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from src.utils.seed_utils import set_global_seed as set_seed
from src.simulation.materials import get_sio2_permittivity, get_metal_permittivity

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}', flush=True)


def compute_physics_features_C(params, wavelengths_nm, metal='Cr'):
    """Physics features for Structure C: Dual-Polarization Rectangular MIM.
    Layers: Air | Patterned Cr (rect Wx x Wy) | SiO2 | Cr mirror | Glass
    Params: P, Wx, Wy, t_Cr, d_SiO2, theta, phi
    """
    N = len(params)
    Nlam = len(wavelengths_nm)
    P      = params[:,0]
    Wx     = params[:,1]
    Wy     = params[:,2]
    t_Cr   = params[:,3]
    d_SiO2 = params[:,4]
    theta  = params[:,5]
    phi    = params[:,6]
    theta_rad = np.deg2rad(theta)
    phi_rad   = np.deg2rad(phi)

    eps_sio2 = get_sio2_permittivity(wavelengths_nm)
    eps_metal = get_metal_permittivity(wavelengths_nm, metal)
    n_sio2 = np.sqrt(np.real(eps_sio2))
    k_metal = np.imag(np.sqrt(eps_metal))
    skin_depth = wavelengths_nm / (4 * np.pi * k_metal)

    feats = []
    # 1-2: Fabry-Perot cavity phase (SiO2)
    sin_ti = np.clip(np.sin(theta_rad[:,None]) / n_sio2[None,:], -1, 1)
    cos_ti = np.sqrt(1 - sin_ti**2)
    phase = 4*np.pi * n_sio2[None,:] * d_SiO2[:,None] * cos_ti / wavelengths_nm[None,:]
    feats.append(np.cos(phase))   # 1
    feats.append(np.sin(phase))   # 2

    # 3: Fill fraction Wx*Wy/P^2
    feats.append(np.tile((Wx*Wy/P**2)[:,None], (1,Nlam)))   # 3

    # 4-5: Sub-wavelength ratios
    feats.append(P[:,None] / wavelengths_nm[None,:])     # 4
    feats.append(Wx[:,None] / wavelengths_nm[None,:])    # 5

    # 6: Wy/lambda (important for polarization dependence!)
    feats.append(Wy[:,None] / wavelengths_nm[None,:])    # 6

    # 7: Skin depth ratio
    feats.append(t_Cr[:,None] / skin_depth[None,:])      # 7

    # 8: Optical path length
    feats.append(n_sio2[None,:] * d_SiO2[:,None] / wavelengths_nm[None,:]) # 8

    # 9: cos(theta)
    feats.append(np.tile(np.cos(theta_rad[:,None]), (1,Nlam)))  # 9

    # 10: cos(phi) - azimuthal angle affects TE/TM coupling
    feats.append(np.tile(np.cos(phi_rad[:,None]), (1,Nlam)))  # 10

    # 11: sin(phi) - azimuthal angle
    feats.append(np.tile(np.sin(phi_rad[:,None]), (1,Nlam)))  # 11

    # 12: Aspect ratio Wy/Wx (polarization asymmetry indicator)
    feats.append(np.tile((Wy/(Wx+1e-10))[:,None], (1,Nlam)))  # 12

    # 13: Metal absorption coefficient
    alpha = 4*np.pi*k_metal / wavelengths_nm
    feats.append(np.tile(alpha[None,:], (N,1)))  # 13

    return np.stack(feats, axis=-1).astype(np.float32)  # (N, Nlam, 13)


class BaseResNet(nn.Module):
    def __init__(self, in_dim, hidden=256, n_blocks=4):
        super().__init__()
        self.fc_in = nn.Linear(in_dim, hidden)
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden,hidden), nn.LayerNorm(hidden), nn.SiLU(),
                          nn.Linear(hidden,hidden), nn.LayerNorm(hidden))
            for _ in range(n_blocks)])
        self.act = nn.SiLU()
    def forward(self, x):
        h = self.act(self.fc_in(x))
        for b in self.blocks: h = h + self.act(b(h))
        return h

class M0(nn.Module):
    def __init__(self, d, out_ch=1):
        super().__init__()
        self.bb = BaseResNet(d)
        self.head = nn.Sequential(nn.Linear(256,128), nn.SiLU(), nn.Linear(128,out_ch), nn.Sigmoid())
    def forward(self, x, **kw):
        out = self.head(self.bb(x))
        R = out[:,:1].squeeze(-1)
        return {'A': 1-R, 'R': R}

class M0_dual(nn.Module):
    """Two-head M0 for TE and TM."""
    def __init__(self, d):
        super().__init__()
        self.bb = BaseResNet(d)
        self.head_te = nn.Sequential(nn.Linear(256,128), nn.SiLU(), nn.Linear(128,1), nn.Sigmoid())
        self.head_tm = nn.Sequential(nn.Linear(256,128), nn.SiLU(), nn.Linear(128,1), nn.Sigmoid())
    def forward(self, x, **kw):
        h = self.bb(x)
        r_te = self.head_te(h).squeeze(-1)
        r_tm = self.head_tm(h).squeeze(-1)
        return {'A_TE': 1-r_te, 'R_TE': r_te, 'A_TM': 1-r_tm, 'R_TM': r_tm}

class MPhys_dual(nn.Module):
    """Two-head M_phys for TE and TM."""
    def __init__(self, gd, pd):
        super().__init__()
        self.bb = BaseResNet(gd+pd)
        self.head_te = nn.Sequential(nn.Linear(256,128), nn.SiLU(), nn.Linear(128,1), nn.Sigmoid())
        self.head_tm = nn.Sequential(nn.Linear(256,128), nn.SiLU(), nn.Linear(128,1), nn.Sigmoid())
    def forward(self, x, p=None, **kw):
        h = self.bb(torch.cat([x,p],-1))
        r_te = self.head_te(h).squeeze(-1)
        r_tm = self.head_tm(h).squeeze(-1)
        return {'A_TE': 1-r_te, 'R_TE': r_te, 'A_TM': 1-r_tm, 'R_TM': r_tm}


def train_eval_dual(model, dl_tr, dl_vl, dl_te, epochs=5000, lr=1e-3, has_phys=False):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    crit = nn.MSELoss()
    best_vl, best_st = float('inf'), None
    for ep in range(epochs):
        model.train()
        for batch in dl_tr:
            if has_phys: x,a_te,r_te,a_tm,r_tm,p = batch; out = model(x, p=p)
            else: x,a_te,r_te,a_tm,r_tm = batch; out = model(x)
            loss = crit(out['A_TE'],a_te) + crit(out['R_TE'],r_te) + crit(out['A_TM'],a_tm) + crit(out['R_TM'],r_tm)
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
        if (ep+1) % 1000 == 0:
            model.eval()
            with torch.no_grad():
                vl=0; vn=0
                for batch in dl_vl:
                    if has_phys: x,a_te,r_te,a_tm,r_tm,p = batch; out = model(x, p=p)
                    else: x,a_te,r_te,a_tm,r_tm = batch; out = model(x)
                    vl += (nn.functional.l1_loss(out['A_TE'],a_te,reduction='sum') +
                           nn.functional.l1_loss(out['A_TM'],a_tm,reduction='sum')).item()
                    vn += len(a_te)*2
                vm = vl/vn
                if vm < best_vl:
                    best_vl = vm; best_st = {k:v.clone() for k,v in model.state_dict().items()}
    if best_st: model.load_state_dict(best_st)
    model.eval()
    with torch.no_grad():
        te_loss=0; te_n=0
        for batch in dl_te:
            if has_phys: x,a_te,r_te,a_tm,r_tm,p = batch; out = model(x, p=p)
            else: x,a_te,r_te,a_tm,r_tm = batch; out = model(x)
            te_loss += (nn.functional.l1_loss(out['A_TE'],a_te,reduction='sum') +
                        nn.functional.l1_loss(out['A_TM'],a_tm,reduction='sum')).item()
            te_n += len(a_te)*2
    return te_loss / te_n


# ========= Load data =========
datapath = 'data/raw/struct_C_500.npz'
print(f'Loading: {datapath}', flush=True)
data = np.load(datapath, allow_pickle=True)
params_all = data['params'].astype(np.float32)
A_TE_all = data['A_TE'].astype(np.float32)
R_TE_all = data['R_TE'].astype(np.float32)
A_TM_all = data['A_TM'].astype(np.float32)
R_TM_all = data['R_TM'].astype(np.float32)
wavelengths = data['wavelengths'].astype(np.float64)
Nlam = len(wavelengths)

# Filter bad samples
good = np.all(A_TE_all >= -0.01, axis=1) & np.all(A_TM_all >= -0.01, axis=1)
gi = np.where(good)[0]
N = len(gi)
params = params_all[gi]
A_TE = np.clip(A_TE_all[gi], 0, 1)
R_TE = np.clip(R_TE_all[gi], 0, 1)
A_TM = np.clip(A_TM_all[gi], 0, 1)
R_TM = np.clip(R_TM_all[gi], 0, 1)
print(f'Data: {N} good samples out of {len(params_all)}, {Nlam} wavelengths', flush=True)

# Normalize params
BOUNDS_C = np.array([[300,800],[50,720],[50,720],[20,80],[50,200],[0,60],[0,45]], dtype=np.float32)
params_norm = (params - BOUNDS_C[:,0]) / (BOUNDS_C[:,1] - BOUNDS_C[:,0])

phys = compute_physics_features_C(params.astype(np.float64), wavelengths, 'Cr')
n_phys = phys.shape[-1]
print(f'Physics features: {n_phys}', flush=True)

wl_norm = (wavelengths.astype(np.float32) - wavelengths.min()) / (wavelengths.max() - wavelengths.min())
geo_dim = 1 + params.shape[1]  # wavelength + 7 params = 8

params_rep = np.repeat(params_norm[:,None,:], Nlam, axis=1)
wl_rep = np.tile(wl_norm[None,:,None], (N,1,1))
X_geo = np.concatenate([wl_rep, params_rep], axis=-1).reshape(-1, geo_dim).astype(np.float32)
X_phys = phys.reshape(-1, n_phys).astype(np.float32)
pm, ps = X_phys.mean(0,keepdims=True), X_phys.std(0,keepdims=True)+1e-8
X_phys_n = ((X_phys - pm) / ps).astype(np.float32)
A_TE_flat = A_TE.reshape(-1).astype(np.float32)
R_TE_flat = R_TE.reshape(-1).astype(np.float32)
A_TM_flat = A_TM.reshape(-1).astype(np.float32)
R_TM_flat = R_TM.reshape(-1).astype(np.float32)

def get_rows(si): return np.concatenate([np.arange(i*Nlam,(i+1)*Nlam) for i in si])

# ========= Experiment =========
TRAIN_SIZES = [50, 100, 200, 350]
N_TEST = 50; N_VAL = 50
SEEDS = [42, 123, 777]
EPOCHS = 5000

rng_split = np.random.default_rng(42)
all_idx = rng_split.permutation(N)
test_idx = all_idx[-N_TEST:]
val_idx  = all_idx[-(N_TEST+N_VAL):-N_TEST]
remaining = all_idx[:-(N_TEST+N_VAL)]
print(f'Test: {len(test_idx)}, Val: {len(val_idx)}, Pool: {len(remaining)}', flush=True)

test_rows = get_rows(test_idx); val_rows = get_rows(val_idx)
X_te=torch.tensor(X_geo[test_rows]).to(device)
Ate_te=torch.tensor(A_TE_flat[test_rows]).to(device)
Rte_te=torch.tensor(R_TE_flat[test_rows]).to(device)
Atm_te=torch.tensor(A_TM_flat[test_rows]).to(device)
Rtm_te=torch.tensor(R_TM_flat[test_rows]).to(device)
P_te=torch.tensor(X_phys_n[test_rows]).to(device)

X_vl=torch.tensor(X_geo[val_rows]).to(device)
Ate_vl=torch.tensor(A_TE_flat[val_rows]).to(device)
Rte_vl=torch.tensor(R_TE_flat[val_rows]).to(device)
Atm_vl=torch.tensor(A_TM_flat[val_rows]).to(device)
Rtm_vl=torch.tensor(R_TM_flat[val_rows]).to(device)
P_vl=torch.tensor(X_phys_n[val_rows]).to(device)

dl_te_m0 = DataLoader(TensorDataset(X_te,Ate_te,Rte_te,Atm_te,Rtm_te), batch_size=2048)
dl_te_ph = DataLoader(TensorDataset(X_te,Ate_te,Rte_te,Atm_te,Rtm_te,P_te), batch_size=2048)
dl_vl_m0 = DataLoader(TensorDataset(X_vl,Ate_vl,Rte_vl,Atm_vl,Rtm_vl), batch_size=2048)
dl_vl_ph = DataLoader(TensorDataset(X_vl,Ate_vl,Rte_vl,Atm_vl,Rtm_vl,P_vl), batch_size=2048)

results = {sz: {'M0':[], 'M_phys':[]} for sz in TRAIN_SIZES}

for n_train in TRAIN_SIZES:
    if n_train > len(remaining): continue
    for seed in SEEDS:
        print(f'\n--- n_train={n_train}, seed={seed} ---', flush=True)
        set_seed(seed)
        rng2 = np.random.default_rng(seed)
        tr_idx = remaining[rng2.permutation(len(remaining))[:n_train]]
        tr_rows = get_rows(tr_idx)
        X_tr=torch.tensor(X_geo[tr_rows]).to(device)
        Ate_tr=torch.tensor(A_TE_flat[tr_rows]).to(device)
        Rte_tr=torch.tensor(R_TE_flat[tr_rows]).to(device)
        Atm_tr=torch.tensor(A_TM_flat[tr_rows]).to(device)
        Rtm_tr=torch.tensor(R_TM_flat[tr_rows]).to(device)
        P_tr=torch.tensor(X_phys_n[tr_rows]).to(device)

        dl_tr_m0 = DataLoader(TensorDataset(X_tr,Ate_tr,Rte_tr,Atm_tr,Rtm_tr), batch_size=512, shuffle=True)
        dl_tr_ph = DataLoader(TensorDataset(X_tr,Ate_tr,Rte_tr,Atm_tr,Rtm_tr,P_tr), batch_size=512, shuffle=True)

        set_seed(seed)
        m0 = M0_dual(geo_dim).to(device)
        mae_m0 = train_eval_dual(m0, dl_tr_m0, dl_vl_m0, dl_te_m0, EPOCHS, has_phys=False)
        results[n_train]['M0'].append(mae_m0)
        print(f'  M0: {mae_m0*100:.3f}%', flush=True)

        set_seed(seed)
        mp = MPhys_dual(geo_dim, n_phys).to(device)
        mae_ph = train_eval_dual(mp, dl_tr_ph, dl_vl_ph, dl_te_ph, EPOCHS, has_phys=True)
        results[n_train]['M_phys'].append(mae_ph)
        print(f'  M_phys: {mae_ph*100:.3f}%', flush=True)

# ========= Summary =========
print('\n' + '='*70, flush=True)
print('DATA EFFICIENCY: M0 vs M_phys (Structure C, Dual-Polarization)', flush=True)
print('='*70, flush=True)
print(f'Good samples: {N}, Test: {N_TEST}, Val: {N_VAL}, Seeds: {SEEDS}', flush=True)
print(f'{"n_train":>8} | {"M0 MAE":>15} | {"M_phys MAE":>15} | {"improvement":>12}', flush=True)
print('-'*70, flush=True)
for sz in TRAIN_SIZES:
    if not results[sz]['M0']: continue
    m0v = np.array(results[sz]['M0'])*100
    mpv = np.array(results[sz]['M_phys'])*100
    m0m,m0s = m0v.mean(),m0v.std()
    mpm,mps = mpv.mean(),mpv.std()
    impr = (1-mpm/m0m)*100
    print(f'{sz:>8} | {m0m:>6.2f} +/- {m0s:>4.2f}% | {mpm:>6.2f} +/- {mps:>4.2f}% | {impr:>10.1f}%', flush=True)

np.savez('results/data_efficiency_C.npz',
         train_sizes=TRAIN_SIZES, seeds=SEEDS, results_m0={sz:results[sz]['M0'] for sz in TRAIN_SIZES},
         results_mphys={sz:results[sz]['M_phys'] for sz in TRAIN_SIZES})
print('\nResults saved!', flush=True)
