#!/usr/bin/env python3
"""PBTL 4-way: Structure C v2 — Anisotropic TMM + Polarization features."""
import sys, os, time
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from copy import deepcopy
from src.utils.seed_utils import set_global_seed as set_seed
from src.simulation.materials import get_sio2_permittivity, get_metal_permittivity
from src.simulation.tmm_struct_c_aniso import compute_tmm_batch  # ANISOTROPIC TMM

# REDESIGN variant: corrected RCWA data (struct_C_500_redesign.npz, jc materials,
# adaptive order, c64). Grid LOADED from data file; MATERIAL_MODEL='jc' explicit;
# sample filter uses disclosed reliable_TE & reliable_TM masks; output tagged _redesign.
import src.simulation.materials as _mat
_mat.MATERIAL_MODEL = "jc"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}", flush=True)

BOUNDS_C = np.array([[300,800],[50,720],[50,720],[20,80],[50,200],[0,60],[0,45]], dtype=np.float32)

def compute_physics_features_C_v2(params, wavelengths_nm, metal="Cr"):
    """Physics features v2: original 13 + 5 polarization-specific = 18 features."""
    N = len(params); Nlam = len(wavelengths_nm)
    P,Wx,Wy,t_Cr,d_SiO2,theta,phi = [params[:,i] for i in range(7)]
    theta_rad = np.deg2rad(theta); phi_rad = np.deg2rad(phi)
    eps_sio2 = get_sio2_permittivity(wavelengths_nm)
    eps_metal = get_metal_permittivity(wavelengths_nm, metal)
    n_sio2 = np.sqrt(np.real(eps_sio2))
    k_metal = np.imag(np.sqrt(eps_metal))
    skin_depth = wavelengths_nm / (4*np.pi*k_metal)
    feats = []

    # --- Original 13 features ---
    sin_ti = np.clip(np.sin(theta_rad[:,None]) / n_sio2[None,:], -1, 1)
    cos_ti = np.sqrt(1 - sin_ti**2)
    phase = 4*np.pi * n_sio2[None,:] * d_SiO2[:,None] * cos_ti / wavelengths_nm[None,:]
    feats.append(np.cos(phase))                                          # 0
    feats.append(np.sin(phase))                                          # 1
    feats.append(np.tile((Wx*Wy/P**2)[:,None], (1,Nlam)))                # 2: isotropic fill
    feats.append(P[:,None] / wavelengths_nm[None,:])                     # 3: P/lam
    feats.append(Wx[:,None] / wavelengths_nm[None,:])                    # 4: Wx/lam
    feats.append(Wy[:,None] / wavelengths_nm[None,:])                    # 5: Wy/lam
    feats.append(t_Cr[:,None] / skin_depth[None,:])                      # 6: t/delta
    feats.append(n_sio2[None,:] * d_SiO2[:,None] / wavelengths_nm[None,:]) # 7: optical path
    feats.append(np.tile(np.cos(theta_rad)[:,None], (1,Nlam)))           # 8
    feats.append(np.tile(np.cos(phi_rad)[:,None], (1,Nlam)))             # 9
    feats.append(np.tile(np.sin(phi_rad)[:,None], (1,Nlam)))             # 10
    feats.append(np.tile((Wy/(Wx+1e-10))[:,None], (1,Nlam)))            # 11: AR
    alpha = 4*np.pi*k_metal / wavelengths_nm
    feats.append(np.tile(alpha[None,:], (N,1)))                          # 12: absorption coeff

    # --- NEW: 5 polarization-specific features ---
    f_x = Wx / P   # directional fill fraction (TM-relevant)
    f_y = Wy / P   # directional fill fraction (TE-relevant)
    feats.append(np.tile(f_x[:,None], (1,Nlam)))                         # 13: f_x = Wx/P
    feats.append(np.tile(f_y[:,None], (1,Nlam)))                         # 14: f_y = Wy/P
    aniso = np.abs(f_x - f_y) / (f_x + f_y + 1e-10)
    feats.append(np.tile(aniso[:,None], (1,Nlam)))                       # 15: normalized anisotropy
    # Resonance parameter per polarization direction
    feats.append(Wx[:,None]**2 / (P[:,None] * wavelengths_nm[None,:]))   # 16: Wx^2/(P*lam) ~ TM resonance
    feats.append(Wy[:,None]**2 / (P[:,None] * wavelengths_nm[None,:]))   # 17: Wy^2/(P*lam) ~ TE resonance

    return np.stack(feats, axis=-1).astype(np.float32)

class BaseResNet(nn.Module):
    def __init__(self, in_dim, hidden=256, n_blocks=4):
        super().__init__()
        self.fc_in = nn.Linear(in_dim, hidden)
        self.blocks = nn.ModuleList([nn.Sequential(nn.Linear(hidden,hidden),nn.LayerNorm(hidden),nn.SiLU(),nn.Linear(hidden,hidden),nn.LayerNorm(hidden)) for _ in range(n_blocks)])
        self.act = nn.SiLU()
    def forward(self, x):
        h = self.act(self.fc_in(x))
        for b in self.blocks: h = h + self.act(b(h))
        return h

class M0_dual(nn.Module):
    def __init__(self, d):
        super().__init__(); self.bb = BaseResNet(d)
        self.head_te = nn.Sequential(nn.Linear(256,128),nn.SiLU(),nn.Linear(128,1),nn.Sigmoid())
        self.head_tm = nn.Sequential(nn.Linear(256,128),nn.SiLU(),nn.Linear(128,1),nn.Sigmoid())
    def forward(self, x, **kw):
        h = self.bb(x)
        return {"A_TE": 1-self.head_te(h).squeeze(-1), "R_TE": self.head_te(h).squeeze(-1),
                "A_TM": 1-self.head_tm(h).squeeze(-1), "R_TM": self.head_tm(h).squeeze(-1)}

class MPhys_dual(nn.Module):
    def __init__(self, gd, pd):
        super().__init__(); self.bb = BaseResNet(gd+pd)
        self.head_te = nn.Sequential(nn.Linear(256,128),nn.SiLU(),nn.Linear(128,1),nn.Sigmoid())
        self.head_tm = nn.Sequential(nn.Linear(256,128),nn.SiLU(),nn.Linear(128,1),nn.Sigmoid())
    def forward(self, x, p=None, **kw):
        h = self.bb(torch.cat([x, p], -1))
        return {"A_TE": 1-self.head_te(h).squeeze(-1), "R_TE": self.head_te(h).squeeze(-1),
                "A_TM": 1-self.head_tm(h).squeeze(-1), "R_TM": self.head_tm(h).squeeze(-1)}

def train_model(model, dl_tr, dl_vl, epochs, lr, has_phys=False, dual=True):
    # OPTIMIZED: chunk-over-GPU-resident-tensors with per-epoch torch.randperm
    # reshuffle (mirrors ablation_C_redesign / noise_injection_C_redesign). The
    # DataLoaders already hold GPU-resident TensorDatasets, so we iterate their
    # .dataset.tensors directly in same-batch-size chunks. Mathematically
    # equivalent to the DataLoader loop (same batch size, full-pass eval,
    # per-epoch shuffle); the flattened TMM pretrain stays multi-step.
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    crit = nn.MSELoss(); best_vl = float("inf"); best_st = None
    _ts = dl_tr.dataset.tensors; _N = _ts[0].shape[0]; bs = dl_tr.batch_size
    _vts = dl_vl.dataset.tensors; _vN = _vts[0].shape[0]; _vbs = dl_vl.batch_size
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(_N, device=device)
        for i in range(0, _N, bs):
            idx = perm[i:i+bs]
            if has_phys: x,ate,rte,atm,rtm,p = [t[idx] for t in _ts]; out = model(x, p=p)
            else: x,ate,rte,atm,rtm = [t[idx] for t in _ts]; out = model(x)
            loss = crit(out["A_TE"],ate)+crit(out["R_TE"],rte)+crit(out["A_TM"],atm)+crit(out["R_TM"],rtm)
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
        if (ep+1)%100==0:
            model.eval()
            with torch.no_grad():
                vl=0; vn=0
                for i in range(0, _vN, _vbs):
                    if has_phys: x,ate,rte,atm,rtm,p = [t[i:i+_vbs] for t in _vts]; out = model(x, p=p)
                    else: x,ate,rte,atm,rtm = [t[i:i+_vbs] for t in _vts]; out = model(x)
                    vl += (nn.functional.l1_loss(out["A_TE"],ate,reduction="sum")+nn.functional.l1_loss(out["A_TM"],atm,reduction="sum")).item()
                    vn += len(ate)*2
                vm = vl/vn
                if vm < best_vl: best_vl=vm; best_st={k:v.clone() for k,v in model.state_dict().items()}
    if best_st: model.load_state_dict(best_st)
    return model

def eval_model(model, dl_te, has_phys=False):
    model.eval()
    with torch.no_grad():
        te=0; tn=0; _ts = dl_te.dataset.tensors; _N = _ts[0].shape[0]; bs = dl_te.batch_size
        for i in range(0, _N, bs):
            if has_phys: x,ate,rte,atm,rtm,p = [t[i:i+bs] for t in _ts]; out = model(x, p=p)
            else: x,ate,rte,atm,rtm = [t[i:i+bs] for t in _ts]; out = model(x)
            te += (nn.functional.l1_loss(out["A_TE"],ate,reduction="sum")+nn.functional.l1_loss(out["A_TM"],atm,reduction="sum")).item()
            tn += len(ate)*2
    return te/tn

# ===== Step 1: TMM data (ANISOTROPIC) =====
print("\n=== Step 1: TMM data generation (ANISOTROPIC) ===", flush=True)
N_TMM=5000
RCWA_PATH = "data/raw/struct_C_500_redesign.npz"
wavelengths=np.load(RCWA_PATH, allow_pickle=True)["wavelengths"].astype(np.float32); Nlam=len(wavelengths)
print(f"Grid (from data): {wavelengths.min():.0f}-{wavelengths.max():.0f}nm, {Nlam}pts", flush=True)
rng=np.random.default_rng(99)
params_tmm=rng.uniform(BOUNDS_C[:,0],BOUNDS_C[:,1],(N_TMM,7)).astype(np.float32)
t0=time.time()
tmm_out=compute_tmm_batch(params_tmm, wavelengths.astype(np.float64), "Cr")
print(f"TMM done: {N_TMM} samples in {time.time()-t0:.1f}s", flush=True)

# Check TE/TM difference
te_mean = np.mean(np.abs(tmm_out["A_tmm_te"]))
tm_mean = np.mean(np.abs(tmm_out["A_tmm_tm"]))
diff = np.mean(np.abs(tmm_out["A_tmm_te"] - tmm_out["A_tmm_tm"]))
print(f"Aniso TMM: mean|A_TE|={te_mean:.4f}, mean|A_TM|={tm_mean:.4f}, mean|TE-TM|={diff:.4f}", flush=True)

A_tmm_te=np.clip(tmm_out["A_tmm_te"],0,1).astype(np.float32)
R_tmm_te=np.clip(tmm_out["R_tmm_te"],0,1).astype(np.float32)
A_tmm_tm=np.clip(tmm_out["A_tmm_tm"],0,1).astype(np.float32)
R_tmm_tm=np.clip(tmm_out["R_tmm_tm"],0,1).astype(np.float32)

phys_tmm=compute_physics_features_C_v2(params_tmm.astype(np.float64), wavelengths.astype(np.float64), "Cr")
n_phys=phys_tmm.shape[-1]
print(f"Physics features: {n_phys} (13 original + 5 polarization)", flush=True)

params_tmm_norm=(params_tmm-BOUNDS_C[:,0])/(BOUNDS_C[:,1]-BOUNDS_C[:,0])
wl_norm=((wavelengths-wavelengths.min())/(wavelengths.max()-wavelengths.min())).astype(np.float32)
geo_dim=1+7

params_rep=np.repeat(params_tmm_norm[:,None,:],Nlam,axis=1)
wl_rep=np.tile(wl_norm[None,:,None],(N_TMM,1,1))
X_geo_tmm=np.concatenate([wl_rep,params_rep],axis=-1).reshape(-1,geo_dim).astype(np.float32)
X_phys_tmm=phys_tmm.reshape(-1,n_phys).astype(np.float32)
pm=X_phys_tmm.mean(0,keepdims=True); ps=X_phys_tmm.std(0,keepdims=True)+1e-8
X_phys_tmm_n=((X_phys_tmm-pm)/ps).astype(np.float32)
Ate_f=A_tmm_te.reshape(-1); Rte_f=R_tmm_te.reshape(-1)
Atm_f=A_tmm_tm.reshape(-1); Rtm_f=R_tmm_tm.reshape(-1)

n_tr_tmm=int(N_TMM*0.9); tmm_idx=rng.permutation(N_TMM)
tmm_tr_rows=np.concatenate([np.arange(i*Nlam,(i+1)*Nlam) for i in tmm_idx[:n_tr_tmm]])
tmm_vl_rows=np.concatenate([np.arange(i*Nlam,(i+1)*Nlam) for i in tmm_idx[n_tr_tmm:]])

def to_dl_tmm(rows, has_phys, bs=2048, shuffle=False):
    xg=torch.tensor(X_geo_tmm[rows]).to(device)
    ate=torch.tensor(Ate_f[rows]).to(device); rte=torch.tensor(Rte_f[rows]).to(device)
    atm=torch.tensor(Atm_f[rows]).to(device); rtm=torch.tensor(Rtm_f[rows]).to(device)
    if has_phys:
        p=torch.tensor(X_phys_tmm_n[rows]).to(device)
        return DataLoader(TensorDataset(xg,ate,rte,atm,rtm,p),bs,shuffle=shuffle)
    return DataLoader(TensorDataset(xg,ate,rte,atm,rtm),bs,shuffle=shuffle)

# ===== Step 2: Pre-train =====
print("\n=== Step 2: Pre-train on ANISOTROPIC TMM ===", flush=True)
set_seed(42); pt_m0=M0_dual(geo_dim).to(device)
pt_m0=train_model(pt_m0,to_dl_tmm(tmm_tr_rows,False,shuffle=True),to_dl_tmm(tmm_vl_rows,False),500,1e-3,False)
print(f"Pre-trained M0_dual: TMM val MAE={eval_model(pt_m0,to_dl_tmm(tmm_vl_rows,False),False)*100:.2f}%", flush=True)

set_seed(42); pt_mp=MPhys_dual(geo_dim,n_phys).to(device)
pt_mp=train_model(pt_mp,to_dl_tmm(tmm_tr_rows,True,shuffle=True),to_dl_tmm(tmm_vl_rows,True),500,1e-3,True)
print(f"Pre-trained MPhys_dual: TMM val MAE={eval_model(pt_mp,to_dl_tmm(tmm_vl_rows,True),True)*100:.2f}%", flush=True)

# ===== Step 3: RCWA data =====
print("\n=== Step 3: Load RCWA data ===", flush=True)
data=np.load(RCWA_PATH, allow_pickle=True)
assert np.allclose(data["wavelengths"].astype(np.float32), wavelengths), "grid mismatch!"
params_r=data["params"].astype(np.float32)
A_TE_r=data["A_TE"].astype(np.float32); R_TE_r=data["R_TE"].astype(np.float32)
A_TM_r=data["A_TM"].astype(np.float32); R_TM_r=data["R_TM"].astype(np.float32)
# Disclosed filtering protocol: keep samples reliable at ALL wavelengths in BOTH pols.
if "reliable_TE" in data.files and "reliable_TM" in data.files:
    good = data["reliable_TE"].all(axis=1) & data["reliable_TM"].all(axis=1)
else:
    good=np.all(A_TE_r>=-0.01,axis=1)&np.all(A_TM_r>=-0.01,axis=1)
gi=np.where(good)[0]
params_r=params_r[gi]; A_TE_r=np.clip(A_TE_r[gi],0,1); R_TE_r=np.clip(R_TE_r[gi],0,1)
A_TM_r=np.clip(A_TM_r[gi],0,1); R_TM_r=np.clip(R_TM_r[gi],0,1); N_r=len(gi)
print(f"RCWA: {N_r} reliable samples (of {len(good)})", flush=True)

phys_r=compute_physics_features_C_v2(params_r.astype(np.float64), wavelengths.astype(np.float64), "Cr")
params_r_norm=(params_r-BOUNDS_C[:,0])/(BOUNDS_C[:,1]-BOUNDS_C[:,0])
params_rep_r=np.repeat(params_r_norm[:,None,:],Nlam,axis=1)
wl_rep_r=np.tile(wl_norm[None,:,None],(N_r,1,1))
X_geo_r=np.concatenate([wl_rep_r,params_rep_r],axis=-1).reshape(-1,geo_dim).astype(np.float32)
X_phys_r=phys_r.reshape(-1,n_phys).astype(np.float32)
X_phys_r_n=((X_phys_r-pm)/ps).astype(np.float32)
Ate_r=A_TE_r.reshape(-1); Rte_r=R_TE_r.reshape(-1)
Atm_r=A_TM_r.reshape(-1); Rtm_r=R_TM_r.reshape(-1)

def get_rows(si): return np.concatenate([np.arange(i*Nlam,(i+1)*Nlam) for i in si])
rng_sp=np.random.default_rng(42); all_idx=rng_sp.permutation(N_r)
N_TEST=50; N_VAL=50
test_idx=all_idx[-N_TEST:]; val_idx=all_idx[-(N_TEST+N_VAL):-N_TEST]; remaining=all_idx[:-(N_TEST+N_VAL)]
test_rows=get_rows(test_idx); val_rows=get_rows(val_idx)

def make_dl(rows, has_phys, bs=2048, shuffle=False):
    xg=torch.tensor(X_geo_r[rows]).to(device)
    ate=torch.tensor(Ate_r[rows]).to(device); rte=torch.tensor(Rte_r[rows]).to(device)
    atm=torch.tensor(Atm_r[rows]).to(device); rtm=torch.tensor(Rtm_r[rows]).to(device)
    if has_phys:
        p=torch.tensor(X_phys_r_n[rows]).to(device)
        return DataLoader(TensorDataset(xg,ate,rte,atm,rtm,p),bs,shuffle=shuffle)
    return DataLoader(TensorDataset(xg,ate,rte,atm,rtm),bs,shuffle=shuffle)

dl_te_m0=make_dl(test_rows,False); dl_te_ph=make_dl(test_rows,True)
dl_vl_m0=make_dl(val_rows,False); dl_vl_ph=make_dl(val_rows,True)

# ===== Step 4: 4-way comparison =====
print("\n=== Step 4: 4-way comparison (Anisotropic TMM + Pol features) ===", flush=True)
TRAIN_SIZES=[50,100,200,350]; SEEDS=[42,123,777,321,456,654,999,111,222,333]
results={sz:{"M0":[],"M_phys":[],"M_TL":[],"M_TL+phys":[]} for sz in TRAIN_SIZES}

# Note (REDESIGN 2026-06): with the corrected reliable_TE & reliable_TM filter the
# Structure C pool is 487 reliable samples -> 387 remaining after the 50+50 test/val
# reservation, so n_train=350 NOW FITS (no truncation). All four sizes [50,100,200,350]
# train on genuine n_train samples. The effective_n guard below is retained for safety
# but never fires for the redesign data -> report the C n=350 row as a TRUE n=350
# (the old "effective n=300" caveat applied only to the legacy ~400-sample data).
N_REMAINING_C = len(remaining)
if N_REMAINING_C < max(TRAIN_SIZES):
    print(f"WARNING: training pool ({N_REMAINING_C}) is smaller than max TRAIN_SIZES "
          f"({max(TRAIN_SIZES)}). Labels above {N_REMAINING_C} will train on the full "
          f"pool of {N_REMAINING_C} samples.", flush=True)

for n_train in TRAIN_SIZES:
    for seed in SEEDS:
        print(f"\n--- n_train={n_train}, seed={seed} ---", flush=True)
        effective_n = min(n_train, N_REMAINING_C)
        if effective_n != n_train:
            print(f"  [effective n={effective_n}, label n={n_train}]", flush=True)
        rng2=np.random.default_rng(seed); tr_idx=remaining[rng2.permutation(len(remaining))[:n_train]]
        tr_rows=get_rows(tr_idx)
        dl_tr_m0=make_dl(tr_rows,False,bs=512,shuffle=True)
        dl_tr_ph=make_dl(tr_rows,True,bs=512,shuffle=True)

        set_seed(seed); m0=M0_dual(geo_dim).to(device)
        m0=train_model(m0,dl_tr_m0,dl_vl_m0,1000,1e-3,False)
        results[n_train]["M0"].append(eval_model(m0,dl_te_m0,False))
        v=results[n_train]['M0'][-1]*100; print(f'  M0: {v:.3f}%', flush=True)

        set_seed(seed); mp=MPhys_dual(geo_dim,n_phys).to(device)
        mp=train_model(mp,dl_tr_ph,dl_vl_ph,1000,1e-3,True)
        results[n_train]["M_phys"].append(eval_model(mp,dl_te_ph,True))
        v=results[n_train]['M_phys'][-1]*100; print(f'  M_phys: {v:.3f}%', flush=True)

        set_seed(seed); m_tl=M0_dual(geo_dim).to(device); m_tl.load_state_dict(deepcopy(pt_m0.state_dict()))
        m_tl=train_model(m_tl,dl_tr_m0,dl_vl_m0,1000,3e-4,False)
        results[n_train]["M_TL"].append(eval_model(m_tl,dl_te_m0,False))
        v=results[n_train]['M_TL'][-1]*100; print(f'  M_TL: {v:.3f}%', flush=True)

        set_seed(seed); m_tlp=MPhys_dual(geo_dim,n_phys).to(device); m_tlp.load_state_dict(deepcopy(pt_mp.state_dict()))
        m_tlp=train_model(m_tlp,dl_tr_ph,dl_vl_ph,1000,3e-4,True)
        results[n_train]["M_TL+phys"].append(eval_model(m_tlp,dl_te_ph,True))
        v=results[n_train]['M_TL+phys'][-1]*100; print(f'  M_TL+phys: {v:.3f}%', flush=True)

print("\n"+"="*70, flush=True)
print("PBTL 4-way: Structure C v2 (Aniso TMM + Pol features)", flush=True)
print("="*70, flush=True)
hdr = '     n |         M0 |     M_phys |       M_TL |    M_TL+phys'
print(hdr, flush=True); print("-"*70, flush=True)
for sz in TRAIN_SIZES:
    r = results[sz]
    if not r["M0"]: continue
    a,b,c,d=np.mean(r['M0'])*100,np.mean(r['M_phys'])*100,np.mean(r['M_TL'])*100,np.mean(r['M_TL+phys'])*100
    print(f'{sz:>6} | {a:>8.2f}% | {b:>8.2f}% | {c:>8.2f}% | {d:>10.2f}%', flush=True)

os.makedirs("results", exist_ok=True)
np.savez("results/pbtl_C_v2_redesign_10seed.npz", train_sizes=TRAIN_SIZES, seeds=SEEDS,
         M0=[results[sz]["M0"] for sz in TRAIN_SIZES], M_phys=[results[sz]["M_phys"] for sz in TRAIN_SIZES],
         M_TL=[results[sz]["M_TL"] for sz in TRAIN_SIZES], M_TL_phys=[results[sz]["M_TL+phys"] for sz in TRAIN_SIZES])
print("Results saved: pbtl_C_v2_redesign_10seed.npz", flush=True)
