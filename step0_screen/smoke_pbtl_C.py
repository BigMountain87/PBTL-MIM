#!/usr/bin/env python3
"""SMOKE TEST for the optimized pbtl_C_v2_redesign train loop.

Replicates the pbtl_C data pipeline at REDUCED scale (small N_TMM, few epochs,
1 seed, sizes [50,100]) and trains an M0_dual + M_TL with BOTH:
  (A) the NEW optimized chunk-over-GPU-resident-tensors loop, and
  (B) a reference DataLoader loop (the pre-optimization iteration),
on the SAME fixed seed and split, to confirm:
  * MAEs are on a sane absorption scale (a few %),
  * the two implementations agree closely (per-epoch shuffle differs, so not
    bit-identical, but same procedure -> close MAE),
  * TL gives benefit over M0 (sanity of pretrain path).
Trains small; not a results run.
"""
import sys, os, time
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from copy import deepcopy
from src.utils.seed_utils import set_global_seed as set_seed
import src.simulation.materials as _mat
_mat.MATERIAL_MODEL = "jc"
from src.simulation.tmm_struct_c_aniso import compute_tmm_batch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}", flush=True)

# import model + feature builders from the (optimized) pbtl_C module
import importlib.util
spec = importlib.util.spec_from_file_location("pbtlc", "step0_screen/pbtl_C_v2_redesign.py")
# We cannot exec the whole module (it runs the full experiment). Instead, copy the
# minimal pieces here mirroring the module's definitions.

BOUNDS_C = np.array([[300,800],[50,720],[50,720],[20,80],[50,200],[0,60],[0,45]], dtype=np.float32)

from src.simulation.materials import get_sio2_permittivity, get_metal_permittivity

def compute_physics_features_C_v2(params, wavelengths_nm, metal="Cr"):
    N = len(params); Nlam = len(wavelengths_nm)
    P,Wx,Wy,t_Cr,d_SiO2,theta,phi = [params[:,i] for i in range(7)]
    theta_rad = np.deg2rad(theta); phi_rad = np.deg2rad(phi)
    eps_sio2 = get_sio2_permittivity(wavelengths_nm); eps_metal = get_metal_permittivity(wavelengths_nm, metal)
    n_sio2 = np.sqrt(np.real(eps_sio2)); k_metal = np.imag(np.sqrt(eps_metal))
    skin_depth = wavelengths_nm / (4*np.pi*k_metal); feats = []
    sin_ti = np.clip(np.sin(theta_rad[:,None]) / n_sio2[None,:], -1, 1); cos_ti = np.sqrt(1 - sin_ti**2)
    phase = 4*np.pi * n_sio2[None,:] * d_SiO2[:,None] * cos_ti / wavelengths_nm[None,:]
    feats.append(np.cos(phase)); feats.append(np.sin(phase))
    feats.append(np.tile((Wx*Wy/P**2)[:,None], (1,Nlam))); feats.append(P[:,None] / wavelengths_nm[None,:])
    feats.append(Wx[:,None] / wavelengths_nm[None,:]); feats.append(Wy[:,None] / wavelengths_nm[None,:])
    feats.append(t_Cr[:,None] / skin_depth[None,:]); feats.append(n_sio2[None,:] * d_SiO2[:,None] / wavelengths_nm[None,:])
    feats.append(np.tile(np.cos(theta_rad)[:,None], (1,Nlam))); feats.append(np.tile(np.cos(phi_rad)[:,None], (1,Nlam)))
    feats.append(np.tile(np.sin(phi_rad)[:,None], (1,Nlam))); feats.append(np.tile((Wy/(Wx+1e-10))[:,None], (1,Nlam)))
    alpha = 4*np.pi*k_metal / wavelengths_nm; feats.append(np.tile(alpha[None,:], (N,1)))
    f_x = Wx / P; f_y = Wy / P
    feats.append(np.tile(f_x[:,None], (1,Nlam))); feats.append(np.tile(f_y[:,None], (1,Nlam)))
    aniso = np.abs(f_x - f_y) / (f_x + f_y + 1e-10); feats.append(np.tile(aniso[:,None], (1,Nlam)))
    feats.append(Wx[:,None]**2 / (P[:,None] * wavelengths_nm[None,:]))
    feats.append(Wy[:,None]**2 / (P[:,None] * wavelengths_nm[None,:]))
    return np.stack(feats, axis=-1).astype(np.float32)

class BaseResNet(nn.Module):
    def __init__(self, in_dim, hidden=256, n_blocks=4):
        super().__init__(); self.fc_in = nn.Linear(in_dim, hidden)
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

# --- NEW optimized loop (mirror of pbtl_C optimized) ---
def train_opt(model, dl_tr, dl_vl, epochs, lr):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    crit = nn.MSELoss(); best_vl = float("inf"); best_st = None
    _ts = dl_tr.dataset.tensors; _N = _ts[0].shape[0]; bs = dl_tr.batch_size
    _vts = dl_vl.dataset.tensors; _vN = _vts[0].shape[0]; _vbs = dl_vl.batch_size
    for ep in range(epochs):
        model.train(); perm = torch.randperm(_N, device=device)
        for i in range(0, _N, bs):
            idx = perm[i:i+bs]; x,ate,rte,atm,rtm = [t[idx] for t in _ts]; out = model(x)
            loss = crit(out["A_TE"],ate)+crit(out["R_TE"],rte)+crit(out["A_TM"],atm)+crit(out["R_TM"],rtm)
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
        if (ep+1)%50==0:
            model.eval()
            with torch.no_grad():
                vl=0; vn=0
                for i in range(0, _vN, _vbs):
                    x,ate,rte,atm,rtm = [t[i:i+_vbs] for t in _vts]; out = model(x)
                    vl += (nn.functional.l1_loss(out["A_TE"],ate,reduction="sum")+nn.functional.l1_loss(out["A_TM"],atm,reduction="sum")).item(); vn += len(ate)*2
                vm = vl/vn
                if vm < best_vl: best_vl=vm; best_st={k:v.clone() for k,v in model.state_dict().items()}
    if best_st: model.load_state_dict(best_st)
    return model

# --- reference DataLoader loop (pre-optimization) ---
def train_dl(model, dl_tr, dl_vl, epochs, lr):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    crit = nn.MSELoss(); best_vl = float("inf"); best_st = None
    for ep in range(epochs):
        model.train()
        for batch in dl_tr:
            x,ate,rte,atm,rtm = batch; out = model(x)
            loss = crit(out["A_TE"],ate)+crit(out["R_TE"],rte)+crit(out["A_TM"],atm)+crit(out["R_TM"],rtm)
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
        if (ep+1)%50==0:
            model.eval()
            with torch.no_grad():
                vl=0; vn=0
                for batch in dl_vl:
                    x,ate,rte,atm,rtm = batch; out = model(x)
                    vl += (nn.functional.l1_loss(out["A_TE"],ate,reduction="sum")+nn.functional.l1_loss(out["A_TM"],atm,reduction="sum")).item(); vn += len(ate)*2
                vm = vl/vn
                if vm < best_vl: best_vl=vm; best_st={k:v.clone() for k,v in model.state_dict().items()}
    if best_st: model.load_state_dict(best_st)
    return model

def eval_model(model, dl_te):
    model.eval()
    with torch.no_grad():
        te=0; tn=0; _ts = dl_te.dataset.tensors; _N=_ts[0].shape[0]; bs=dl_te.batch_size
        for i in range(0,_N,bs):
            x,ate,rte,atm,rtm = [t[i:i+bs] for t in _ts]; out=model(x)
            te += (nn.functional.l1_loss(out["A_TE"],ate,reduction="sum")+nn.functional.l1_loss(out["A_TM"],atm,reduction="sum")).item(); tn+=len(ate)*2
    return te/tn

# ===== reduced data pipeline =====
N_TMM = 800; EPOCHS_PT = 100; EPOCHS_FT = 200
RCWA_PATH = "data/raw/struct_C_500_redesign.npz"
wavelengths=np.load(RCWA_PATH, allow_pickle=True)["wavelengths"].astype(np.float32); Nlam=len(wavelengths)
rng=np.random.default_rng(99)
params_tmm=rng.uniform(BOUNDS_C[:,0],BOUNDS_C[:,1],(N_TMM,7)).astype(np.float32)
t0=time.time(); tmm_out=compute_tmm_batch(params_tmm, wavelengths.astype(np.float64), "Cr")
print(f"TMM {N_TMM} in {time.time()-t0:.1f}s", flush=True)
A_tmm_te=np.clip(tmm_out["A_tmm_te"],0,1).astype(np.float32); R_tmm_te=np.clip(tmm_out["R_tmm_te"],0,1).astype(np.float32)
A_tmm_tm=np.clip(tmm_out["A_tmm_tm"],0,1).astype(np.float32); R_tmm_tm=np.clip(tmm_out["R_tmm_tm"],0,1).astype(np.float32)
params_tmm_norm=(params_tmm-BOUNDS_C[:,0])/(BOUNDS_C[:,1]-BOUNDS_C[:,0])
wl_norm=((wavelengths-wavelengths.min())/(wavelengths.max()-wavelengths.min())).astype(np.float32)
geo_dim=1+7
params_rep=np.repeat(params_tmm_norm[:,None,:],Nlam,axis=1); wl_rep=np.tile(wl_norm[None,:,None],(N_TMM,1,1))
X_geo_tmm=np.concatenate([wl_rep,params_rep],axis=-1).reshape(-1,geo_dim).astype(np.float32)
Ate_f=A_tmm_te.reshape(-1); Rte_f=R_tmm_te.reshape(-1); Atm_f=A_tmm_tm.reshape(-1); Rtm_f=R_tmm_tm.reshape(-1)
n_tr_tmm=int(N_TMM*0.9); tmm_idx=rng.permutation(N_TMM)
tmm_tr_rows=np.concatenate([np.arange(i*Nlam,(i+1)*Nlam) for i in tmm_idx[:n_tr_tmm]])
tmm_vl_rows=np.concatenate([np.arange(i*Nlam,(i+1)*Nlam) for i in tmm_idx[n_tr_tmm:]])
def to_dl_tmm(rows, bs=2048, shuffle=False):
    xg=torch.tensor(X_geo_tmm[rows]).to(device)
    ate=torch.tensor(Ate_f[rows]).to(device); rte=torch.tensor(Rte_f[rows]).to(device)
    atm=torch.tensor(Atm_f[rows]).to(device); rtm=torch.tensor(Rtm_f[rows]).to(device)
    return DataLoader(TensorDataset(xg,ate,rte,atm,rtm),bs,shuffle=shuffle)

# pretrain via OPTIMIZED loop (this is what production uses)
set_seed(42); pt_m0=M0_dual(geo_dim).to(device)
pt_m0=train_opt(pt_m0,to_dl_tmm(tmm_tr_rows,shuffle=True),to_dl_tmm(tmm_vl_rows),EPOCHS_PT,1e-3)
print(f"Pretrain TMM val MAE={eval_model(pt_m0,to_dl_tmm(tmm_vl_rows))*100:.2f}%", flush=True)

# RCWA
data=np.load(RCWA_PATH, allow_pickle=True)
params_r=data["params"].astype(np.float32)
A_TE_r=data["A_TE"].astype(np.float32); R_TE_r=data["R_TE"].astype(np.float32)
A_TM_r=data["A_TM"].astype(np.float32); R_TM_r=data["R_TM"].astype(np.float32)
good = data["reliable_TE"].all(axis=1) & data["reliable_TM"].all(axis=1); gi=np.where(good)[0]
params_r=params_r[gi]; A_TE_r=np.clip(A_TE_r[gi],0,1); R_TE_r=np.clip(R_TE_r[gi],0,1)
A_TM_r=np.clip(A_TM_r[gi],0,1); R_TM_r=np.clip(R_TM_r[gi],0,1); N_r=len(gi)
print(f"RCWA {N_r} reliable", flush=True)
params_r_norm=(params_r-BOUNDS_C[:,0])/(BOUNDS_C[:,1]-BOUNDS_C[:,0])
params_rep_r=np.repeat(params_r_norm[:,None,:],Nlam,axis=1); wl_rep_r=np.tile(wl_norm[None,:,None],(N_r,1,1))
X_geo_r=np.concatenate([wl_rep_r,params_rep_r],axis=-1).reshape(-1,geo_dim).astype(np.float32)
Ate_r=A_TE_r.reshape(-1); Rte_r=R_TE_r.reshape(-1); Atm_r=A_TM_r.reshape(-1); Rtm_r=R_TM_r.reshape(-1)
def get_rows(si): return np.concatenate([np.arange(i*Nlam,(i+1)*Nlam) for i in si])
rng_sp=np.random.default_rng(42); all_idx=rng_sp.permutation(N_r)
test_idx=all_idx[-50:]; val_idx=all_idx[-100:-50]; remaining=all_idx[:-100]
test_rows=get_rows(test_idx); val_rows=get_rows(val_idx)
def make_dl(rows, bs=2048, shuffle=False):
    xg=torch.tensor(X_geo_r[rows]).to(device)
    ate=torch.tensor(Ate_r[rows]).to(device); rte=torch.tensor(Rte_r[rows]).to(device)
    atm=torch.tensor(Atm_r[rows]).to(device); rtm=torch.tensor(Rtm_r[rows]).to(device)
    return DataLoader(TensorDataset(xg,ate,rte,atm,rtm),bs,shuffle=shuffle)
dl_te=make_dl(test_rows); dl_vl=make_dl(val_rows)

print("\n=== OPT vs DL reference (seed=42) ===", flush=True)
for n_train in [50,100]:
    rng2=np.random.default_rng(42); tr_idx=remaining[rng2.permutation(len(remaining))[:n_train]]
    dl_tr=make_dl(get_rows(tr_idx),bs=512,shuffle=True)
    # M0 optimized
    set_seed(42); m0o=M0_dual(geo_dim).to(device); m0o=train_opt(m0o,dl_tr,dl_vl,EPOCHS_FT,1e-3); mae_m0o=eval_model(m0o,dl_te)
    # M0 dataloader ref
    set_seed(42); m0d=M0_dual(geo_dim).to(device); m0d=train_dl(m0d,dl_tr,dl_vl,EPOCHS_FT,1e-3); mae_m0d=eval_model(m0d,dl_te)
    # M_TL optimized
    set_seed(42); mtl=M0_dual(geo_dim).to(device); mtl.load_state_dict(deepcopy(pt_m0.state_dict())); mtl=train_opt(mtl,dl_tr,dl_vl,EPOCHS_FT,3e-4); mae_tl=eval_model(mtl,dl_te)
    ben=(1-mae_tl/mae_m0o)*100
    print(f"n={n_train}: M0_opt={mae_m0o*100:.3f}%  M0_dl={mae_m0d*100:.3f}%  (diff={abs(mae_m0o-mae_m0d)*100:.3f}pp)  M_TL_opt={mae_tl*100:.3f}%  benefit={ben:+.1f}%", flush=True)
print("\nSMOKE DONE", flush=True)
