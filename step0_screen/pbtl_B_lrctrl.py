#!/usr/bin/env python3
"""PBTL 4-way experiment for Structure B (Ring-Disk Fano, 400-1800nm)."""
import sys, os, time
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from copy import deepcopy
from src.utils.seed_utils import set_global_seed as set_seed
from src.simulation.materials import get_sio2_permittivity, get_metal_permittivity
from src.simulation.tmm_struct_b import compute_tmm_batch

# REDESIGN variant: corrected RCWA data (struct_B_500_redesign.npz, jc materials,
# adaptive order, c64). Grid LOADED from data file; MATERIAL_MODEL='jc' explicit;
# sample filter uses the disclosed `reliable` mask; output -> *_redesign_10seed.npz.
import src.simulation.materials as _mat
_mat.MATERIAL_MODEL = "jc"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}", flush=True)

BOUNDS_B = np.array([[300,800],[80,350],[30,300],[10,100],[20,80],[50,200],[0,60],[0,45]], dtype=np.float32)

def compute_physics_features_B(params, wavelengths_nm, metal='Cr'):
    N = len(params); Nlam = len(wavelengths_nm)
    P,R_out,R_in,R_disk,t_Cr,d_SiO2,theta,phi = [params[:,i] for i in range(8)]
    theta_rad = np.deg2rad(theta); phi_rad = np.deg2rad(phi)
    eps_sio2 = get_sio2_permittivity(wavelengths_nm)
    eps_metal = get_metal_permittivity(wavelengths_nm, metal)
    n_sio2 = np.sqrt(np.real(eps_sio2))
    k_metal = np.imag(np.sqrt(eps_metal))
    skin_depth = wavelengths_nm / (4*np.pi*k_metal)
    feats = []
    sin_ti = np.clip(np.sin(theta_rad[:,None])/n_sio2[None,:], -1, 1)
    cos_ti = np.sqrt(1-sin_ti**2)
    phase = 4*np.pi*n_sio2[None,:]*d_SiO2[:,None]*cos_ti/wavelengths_nm[None,:]
    feats.append(np.cos(phase)); feats.append(np.sin(phase))
    ring_area = np.pi*(R_out**2-R_in**2); disk_area = np.pi*R_disk**2
    feats.append(np.tile((ring_area/P**2)[:,None],(1,Nlam)))
    feats.append(np.tile((disk_area/P**2)[:,None],(1,Nlam)))
    feats.append(np.tile(((ring_area+disk_area)/P**2)[:,None],(1,Nlam)))
    feats.append(P[:,None]/wavelengths_nm[None,:]); feats.append(R_out[:,None]/wavelengths_nm[None,:])
    feats.append(t_Cr[:,None]/skin_depth[None,:])
    feats.append(n_sio2[None,:]*d_SiO2[:,None]/wavelengths_nm[None,:])
    feats.append(np.tile(np.cos(theta_rad)[:,None],(1,Nlam)))
    gap_ratio=(R_out-R_in)/(R_out+1e-10); feats.append(np.tile(gap_ratio[:,None],(1,Nlam)))
    disk_ring=R_disk/(R_in+1e-10); feats.append(np.tile(disk_ring[:,None],(1,Nlam)))
    alpha=4*np.pi*k_metal/wavelengths_nm; feats.append(np.tile(alpha[None,:],(N,1)))
    return np.stack(feats,axis=-1).astype(np.float32)

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

class M0(nn.Module):
    def __init__(self, d):
        super().__init__(); self.bb = BaseResNet(d)
        self.head = nn.Sequential(nn.Linear(256,128),nn.SiLU(),nn.Linear(128,1),nn.Sigmoid())
    def forward(self, x, **kw): R=self.head(self.bb(x)).squeeze(-1); return {'A':1-R,'R':R}

class MPhys(nn.Module):
    def __init__(self, gd, pd):
        super().__init__(); self.bb = BaseResNet(gd+pd)
        self.head = nn.Sequential(nn.Linear(256,128),nn.SiLU(),nn.Linear(128,1),nn.Sigmoid())
    def forward(self, x, p=None, **kw): h=self.bb(torch.cat([x,p],-1)); R=self.head(h).squeeze(-1); return {'A':1-R,'R':R}

def train_model(model, dl_tr, dl_vl, epochs, lr, has_phys=False):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    crit = nn.MSELoss(); best_vl, best_st = float('inf'), None
    for ep in range(epochs):
        model.train()
        for batch in dl_tr:
            if has_phys: x,a,r,p=batch; out=model(x,p=p)
            else: x,a,r=batch; out=model(x)
            loss=crit(out['A'],a)+crit(out['R'],r)
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
        if (ep+1)%100==0:
            model.eval()
            with torch.no_grad():
                vl=0; vn=0
                for batch in dl_vl:
                    if has_phys: x,a,r,p=batch; out=model(x,p=p)
                    else: x,a,r=batch; out=model(x)
                    vl+=(nn.functional.l1_loss(out['A'],a,reduction='sum')+nn.functional.l1_loss(out['R'],r,reduction='sum')).item(); vn+=len(a)*2
                vm=vl/vn
                if vm<best_vl: best_vl=vm; best_st={k:v.clone() for k,v in model.state_dict().items()}
    if best_st: model.load_state_dict(best_st)
    return model

def eval_model(model, dl_te, has_phys=False):
    model.eval()
    with torch.no_grad():
        te=0; tn=0
        for batch in dl_te:
            if has_phys: x,a,r,p=batch; out=model(x,p=p)
            else: x,a,r=batch; out=model(x)
            te+=nn.functional.l1_loss(out['A'],a,reduction='sum').item(); tn+=len(a)
    return te/tn

# ===== Step 1: TMM data =====
print("\n=== Step 1: TMM data generation ===", flush=True)
N_TMM = 5000
RCWA_PATH = "data/raw/struct_B_500_redesign.npz"
wavelengths = np.load(RCWA_PATH, allow_pickle=True)["wavelengths"].astype(np.float32)
Nlam = len(wavelengths)
print(f"Grid (from data): {wavelengths.min():.0f}-{wavelengths.max():.0f}nm, {Nlam}pts", flush=True)
rng = np.random.default_rng(99)
params_tmm = rng.uniform(BOUNDS_B[:,0], BOUNDS_B[:,1], (N_TMM,8)).astype(np.float32)
t0 = time.time()
tmm_out = compute_tmm_batch(params_tmm, wavelengths.astype(np.float64), 'Cr')
print(f"TMM done: {N_TMM} samples in {time.time()-t0:.1f}s", flush=True)
A_tmm = np.clip(tmm_out['A_tmm'], 0, 1).astype(np.float32)
R_tmm = np.clip(tmm_out['R_tmm'], 0, 1).astype(np.float32)

phys_tmm = compute_physics_features_B(params_tmm.astype(np.float64), wavelengths.astype(np.float64), 'Cr')
n_phys = phys_tmm.shape[-1]
params_tmm_norm = (params_tmm - BOUNDS_B[:,0]) / (BOUNDS_B[:,1]-BOUNDS_B[:,0])
wl_norm = ((wavelengths - wavelengths.min())/(wavelengths.max()-wavelengths.min())).astype(np.float32)
geo_dim = 1 + 8

params_rep = np.repeat(params_tmm_norm[:,None,:], Nlam, axis=1)
wl_rep = np.tile(wl_norm[None,:,None], (N_TMM,1,1))
X_geo_tmm = np.concatenate([wl_rep,params_rep],axis=-1).reshape(-1,geo_dim).astype(np.float32)
X_phys_tmm = phys_tmm.reshape(-1,n_phys).astype(np.float32)
pm = X_phys_tmm.mean(0,keepdims=True); ps = X_phys_tmm.std(0,keepdims=True)+1e-8
X_phys_tmm_n = ((X_phys_tmm-pm)/ps).astype(np.float32)
A_tmm_flat = A_tmm.reshape(-1); R_tmm_flat = R_tmm.reshape(-1)

n_tr_tmm = int(N_TMM*0.9)
tmm_idx = rng.permutation(N_TMM)
tmm_tr_rows = np.concatenate([np.arange(i*Nlam,(i+1)*Nlam) for i in tmm_idx[:n_tr_tmm]])
tmm_vl_rows = np.concatenate([np.arange(i*Nlam,(i+1)*Nlam) for i in tmm_idx[n_tr_tmm:]])

def to_dl_tmm(rows, has_phys, bs=2048, shuffle=False):
    xg=torch.tensor(X_geo_tmm[rows]).to(device); a=torch.tensor(A_tmm_flat[rows]).to(device); r=torch.tensor(R_tmm_flat[rows]).to(device)
    if has_phys: p=torch.tensor(X_phys_tmm_n[rows]).to(device); return DataLoader(TensorDataset(xg,a,r,p),bs,shuffle=shuffle)
    return DataLoader(TensorDataset(xg,a,r),bs,shuffle=shuffle)

# ===== Step 2: SKIPPED (lr-control: M0 from scratch only, no pretrain) =====
# Pretrain + pretrained_*_B.pt save removed; control trains M0 from scratch at lr=3e-4.

# ===== Step 3: RCWA data =====
print("\n=== Step 3: Load RCWA data ===", flush=True)
data = np.load(RCWA_PATH, allow_pickle=True)
assert np.allclose(data['wavelengths'].astype(np.float32), wavelengths), "grid mismatch!"
params_r = data['params'].astype(np.float32); A_r = data['A'].astype(np.float32); R_r = data['R'].astype(np.float32)
# Disclosed filtering protocol: keep samples reliable at ALL wavelengths.
if 'reliable' in data.files:
    good = data['reliable'].all(axis=1)
else:
    good = np.all(A_r>=-0.01,axis=1)
gi=np.where(good)[0]
params_r=params_r[gi]; A_r=np.clip(A_r[gi],0,1); R_r=np.clip(R_r[gi],0,1)
N_r=len(gi); print(f"RCWA: {N_r} reliable samples (of {len(good)})", flush=True)

phys_r = compute_physics_features_B(params_r.astype(np.float64), wavelengths.astype(np.float64), 'Cr')
params_r_norm = (params_r-BOUNDS_B[:,0])/(BOUNDS_B[:,1]-BOUNDS_B[:,0])
params_rep_r = np.repeat(params_r_norm[:,None,:],Nlam,axis=1)
wl_rep_r = np.tile(wl_norm[None,:,None],(N_r,1,1))
X_geo_r = np.concatenate([wl_rep_r,params_rep_r],axis=-1).reshape(-1,geo_dim).astype(np.float32)
X_phys_r = phys_r.reshape(-1,n_phys).astype(np.float32)
X_phys_r_n = ((X_phys_r-pm)/ps).astype(np.float32)
A_r_flat=A_r.reshape(-1).astype(np.float32); R_r_flat=R_r.reshape(-1).astype(np.float32)

def get_rows(si): return np.concatenate([np.arange(i*Nlam,(i+1)*Nlam) for i in si])

rng_sp=np.random.default_rng(42); all_idx=rng_sp.permutation(N_r)
N_TEST=50; N_VAL=50
test_idx=all_idx[-N_TEST:]; val_idx=all_idx[-(N_TEST+N_VAL):-N_TEST]; remaining=all_idx[:-(N_TEST+N_VAL)]
test_rows=get_rows(test_idx); val_rows=get_rows(val_idx)

def make_dl(rows, has_phys, bs=2048, shuffle=False):
    xg=torch.tensor(X_geo_r[rows]).to(device); a=torch.tensor(A_r_flat[rows]).to(device); r=torch.tensor(R_r_flat[rows]).to(device)
    if has_phys: p=torch.tensor(X_phys_r_n[rows]).to(device); return DataLoader(TensorDataset(xg,a,r,p),bs,shuffle=shuffle)
    return DataLoader(TensorDataset(xg,a,r),bs,shuffle=shuffle)

dl_te_m0=make_dl(test_rows,False); dl_te_ph=make_dl(test_rows,True)
dl_vl_m0=make_dl(val_rows,False); dl_vl_ph=make_dl(val_rows,True)

# ===== Step 4: 4-way comparison =====
print("\n=== Step 4: 4-way comparison ===", flush=True)
TRAIN_SIZES=[50,100,200,350]; SEEDS=[42,123,777,321,456,654,999,111,222,333]
results={sz:{'M0_lr3e4':[]} for sz in TRAIN_SIZES}

for n_train in TRAIN_SIZES:
    for seed in SEEDS:
        print(f"\n--- n_train={n_train}, seed={seed} ---", flush=True)
        rng2=np.random.default_rng(seed); tr_idx=remaining[rng2.permutation(len(remaining))[:n_train]]
        tr_rows=get_rows(tr_idx)
        dl_tr_m0=make_dl(tr_rows,False,bs=512,shuffle=True)
        # M0 from scratch at lr=3e-4 (MATCHED to M_TL) — lr-control
        set_seed(seed); m0=M0(geo_dim).to(device)
        m0=train_model(m0,dl_tr_m0,dl_vl_m0,1000,3e-4,False)
        results[n_train]['M0_lr3e4'].append(eval_model(m0,dl_te_m0,False))
        print(f"  M0_lr3e4: {results[n_train]['M0_lr3e4'][-1]*100:.3f}%", flush=True)

print('\n'+'='*70, flush=True)
print('lr-CONTROL: M_TL(3e-4) vs M0(3e-4) — Structure B', flush=True)
print('='*70, flush=True)
mtl=m0orig=None
try:
    sv=np.load('results/pbtl_B_redesign_10seed.npz',allow_pickle=True)
    mtl=[np.array(sv['M_TL'][i]) for i in range(len(TRAIN_SIZES))]
    m0orig=[np.array(sv['M0'][i]) for i in range(len(TRAIN_SIZES))]
except Exception as e:
    print(f"(saved M_TL load failed: {e})", flush=True)
print(f'{"n":>6} | {"M0@1e-3":>9} | {"M0@3e-4":>9} | {"M_TL@3e-4":>10} | {"lr-matched benefit":>19}', flush=True)
print('-'*70, flush=True)
for i,sz in enumerate(TRAIN_SIZES):
    m0c=np.mean(results[sz]['M0_lr3e4'])*100
    if mtl is not None:
        m0o=np.mean(m0orig[i])*100; mt=np.mean(mtl[i])*100; ben=(1-mt/m0c)*100
        print(f"{sz:>6} | {m0o:>8.2f}% | {m0c:>8.2f}% | {mt:>9.2f}% | {ben:>+17.1f}%", flush=True)
    else:
        print(f"{sz:>6} | {'?':>9} | {m0c:>8.2f}% | {'?':>10} |", flush=True)

os.makedirs('results', exist_ok=True)
np.savez('results/pbtl_B_lrctrl.npz', train_sizes=TRAIN_SIZES, seeds=SEEDS,
         M0_lr3e4=[results[sz]['M0_lr3e4'] for sz in TRAIN_SIZES])
print("Saved: pbtl_B_lrctrl.npz", flush=True)
