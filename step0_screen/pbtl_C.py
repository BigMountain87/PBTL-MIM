#!/usr/bin/env python3
"""PBTL 4-way experiment for Structure C (Dual-Polarization, 400-1800nm)."""
import sys, os, time
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from copy import deepcopy
from src.utils.seed_utils import set_global_seed as set_seed
from src.simulation.materials import get_sio2_permittivity, get_metal_permittivity
from src.simulation.tmm_struct_c import compute_tmm_batch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}", flush=True)

BOUNDS_C = np.array([[300,800],[50,400],[50,400],[20,80],[50,200],[0,60],[0,45]], dtype=np.float32)

def compute_physics_features_C(params, wavelengths_nm, metal='Cr'):
    N=len(params); Nlam=len(wavelengths_nm)
    P,Wx,Wy,t_Cr,d_SiO2,theta,phi=[params[:,i] for i in range(7)]
    theta_rad=np.deg2rad(theta); phi_rad=np.deg2rad(phi)
    eps_sio2=get_sio2_permittivity(wavelengths_nm); eps_metal=get_metal_permittivity(wavelengths_nm,metal)
    n_sio2=np.sqrt(np.real(eps_sio2)); k_metal=np.imag(np.sqrt(eps_metal))
    skin_depth=wavelengths_nm/(4*np.pi*k_metal); feats=[]
    sin_ti=np.clip(np.sin(theta_rad[:,None])/n_sio2[None,:],-1,1); cos_ti=np.sqrt(1-sin_ti**2)
    phase=4*np.pi*n_sio2[None,:]*d_SiO2[:,None]*cos_ti/wavelengths_nm[None,:]
    feats.append(np.cos(phase)); feats.append(np.sin(phase))
    feats.append(np.tile((Wx*Wy/P**2)[:,None],(1,Nlam)))
    feats.append(P[:,None]/wavelengths_nm[None,:]); feats.append(Wx[:,None]/wavelengths_nm[None,:])
    feats.append(Wy[:,None]/wavelengths_nm[None,:])
    feats.append(t_Cr[:,None]/skin_depth[None,:])
    feats.append(n_sio2[None,:]*d_SiO2[:,None]/wavelengths_nm[None,:])
    feats.append(np.tile(np.cos(theta_rad)[:,None],(1,Nlam)))
    feats.append(np.tile(np.cos(phi_rad)[:,None],(1,Nlam)))
    feats.append(np.tile(np.sin(phi_rad)[:,None],(1,Nlam)))
    feats.append(np.tile((Wy/(Wx+1e-10))[:,None],(1,Nlam)))
    alpha=4*np.pi*k_metal/wavelengths_nm; feats.append(np.tile(alpha[None,:],(N,1)))
    return np.stack(feats,axis=-1).astype(np.float32)

class BaseResNet(nn.Module):
    def __init__(self, in_dim, hidden=256, n_blocks=4):
        super().__init__()
        self.fc_in=nn.Linear(in_dim,hidden)
        self.blocks=nn.ModuleList([nn.Sequential(nn.Linear(hidden,hidden),nn.LayerNorm(hidden),nn.SiLU(),nn.Linear(hidden,hidden),nn.LayerNorm(hidden)) for _ in range(n_blocks)])
        self.act=nn.SiLU()
    def forward(self,x):
        h=self.act(self.fc_in(x))
        for b in self.blocks: h=h+self.act(b(h))
        return h

class M0_dual(nn.Module):
    def __init__(self,d):
        super().__init__(); self.bb=BaseResNet(d)
        self.head_te=nn.Sequential(nn.Linear(256,128),nn.SiLU(),nn.Linear(128,1),nn.Sigmoid())
        self.head_tm=nn.Sequential(nn.Linear(256,128),nn.SiLU(),nn.Linear(128,1),nn.Sigmoid())
    def forward(self,x,**kw):
        h=self.bb(x); Rte=self.head_te(h).squeeze(-1); Rtm=self.head_tm(h).squeeze(-1)
        return {'A_TE':1-Rte,'R_TE':Rte,'A_TM':1-Rtm,'R_TM':Rtm}

class MPhys_dual(nn.Module):
    def __init__(self,gd,pd):
        super().__init__(); self.bb=BaseResNet(gd+pd)
        self.head_te=nn.Sequential(nn.Linear(256,128),nn.SiLU(),nn.Linear(128,1),nn.Sigmoid())
        self.head_tm=nn.Sequential(nn.Linear(256,128),nn.SiLU(),nn.Linear(128,1),nn.Sigmoid())
    def forward(self,x,p=None,**kw):
        h=self.bb(torch.cat([x,p],-1)); Rte=self.head_te(h).squeeze(-1); Rtm=self.head_tm(h).squeeze(-1)
        return {'A_TE':1-Rte,'R_TE':Rte,'A_TM':1-Rtm,'R_TM':Rtm}

def train_model(model,dl_tr,dl_vl,epochs,lr,has_phys=False,dual=False):
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,epochs)
    crit=nn.MSELoss(); best_vl,best_st=float('inf'),None
    for ep in range(epochs):
        model.train()
        for batch in dl_tr:
            if dual and has_phys: x,ate,rte,atm,rtm,p=batch; out=model(x,p=p)
            elif dual: x,ate,rte,atm,rtm=batch; out=model(x)
            elif has_phys: x,a,r,p=batch; out=model(x,p=p)
            else: x,a,r=batch; out=model(x)
            if dual: loss=crit(out['A_TE'],ate)+crit(out['R_TE'],rte)+crit(out['A_TM'],atm)+crit(out['R_TM'],rtm)
            else: loss=crit(out['A'],a)+crit(out['R'],r)
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
        if (ep+1)%100==0:
            model.eval()
            with torch.no_grad():
                vl=0; vn=0
                for batch in dl_vl:
                    if dual and has_phys: x,ate,rte,atm,rtm,p=batch; out=model(x,p=p)
                    elif dual: x,ate,rte,atm,rtm=batch; out=model(x)
                    elif has_phys: x,a,r,p=batch; out=model(x,p=p)
                    else: x,a,r=batch; out=model(x)
                    if dual: vl+=(nn.functional.l1_loss(out['A_TE'],ate,reduction='sum')+nn.functional.l1_loss(out['A_TM'],atm,reduction='sum')).item(); vn+=len(ate)*2
                    else: vl+=(nn.functional.l1_loss(out['A'],a,reduction='sum')+nn.functional.l1_loss(out['R'],r,reduction='sum')).item(); vn+=len(a)*2
                vm=vl/vn
                if vm<best_vl: best_vl=vm; best_st={k:v.clone() for k,v in model.state_dict().items()}
    if best_st: model.load_state_dict(best_st)
    return model

def eval_model(model,dl_te,has_phys=False,dual=False):
    model.eval()
    with torch.no_grad():
        te=0; tn=0
        for batch in dl_te:
            if dual and has_phys: x,ate,rte,atm,rtm,p=batch; out=model(x,p=p)
            elif dual: x,ate,rte,atm,rtm=batch; out=model(x)
            elif has_phys: x,a,r,p=batch; out=model(x,p=p)
            else: x,a,r=batch; out=model(x)
            if dual: te+=(nn.functional.l1_loss(out['A_TE'],ate,reduction='sum')+nn.functional.l1_loss(out['A_TM'],atm,reduction='sum')).item(); tn+=len(ate)*2
            else: te+=nn.functional.l1_loss(out['A'],a,reduction='sum').item(); tn+=len(a)
    return te/tn

# ===== Step 1: TMM data =====
print("\n=== Step 1: TMM data generation ===", flush=True)
N_TMM=5000; wavelengths=np.linspace(400,1800,100).astype(np.float32); Nlam=len(wavelengths)
rng=np.random.default_rng(99)
params_tmm=rng.uniform(BOUNDS_C[:,0],BOUNDS_C[:,1],(N_TMM,7)).astype(np.float32)
t0=time.time()
tmm_out=compute_tmm_batch(params_tmm,wavelengths.astype(np.float64),'Cr')
print(f"TMM done: {N_TMM} samples in {time.time()-t0:.1f}s", flush=True)
# TMM TE=TM for EMA, use average
A_tmm_te=np.clip(tmm_out['A_tmm_te'],0,1).astype(np.float32); R_tmm_te=np.clip(tmm_out['R_tmm_te'],0,1).astype(np.float32)
A_tmm_tm=np.clip(tmm_out['A_tmm_tm'],0,1).astype(np.float32); R_tmm_tm=np.clip(tmm_out['R_tmm_tm'],0,1).astype(np.float32)

phys_tmm=compute_physics_features_C(params_tmm.astype(np.float64),wavelengths.astype(np.float64),'Cr')
n_phys=phys_tmm.shape[-1]
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

def to_dl_tmm(rows,has_phys,bs=2048,shuffle=False):
    xg=torch.tensor(X_geo_tmm[rows]).to(device)
    ate=torch.tensor(Ate_f[rows]).to(device); rte=torch.tensor(Rte_f[rows]).to(device)
    atm=torch.tensor(Atm_f[rows]).to(device); rtm=torch.tensor(Rtm_f[rows]).to(device)
    if has_phys: p=torch.tensor(X_phys_tmm_n[rows]).to(device); return DataLoader(TensorDataset(xg,ate,rte,atm,rtm,p),bs,shuffle=shuffle)
    return DataLoader(TensorDataset(xg,ate,rte,atm,rtm),bs,shuffle=shuffle)

# ===== Step 2: Pre-train =====
print("\n=== Step 2: Pre-train on TMM ===", flush=True)
set_seed(42); pt_m0=M0_dual(geo_dim).to(device)
pt_m0=train_model(pt_m0,to_dl_tmm(tmm_tr_rows,False,shuffle=True),to_dl_tmm(tmm_vl_rows,False),500,1e-3,False,True)
print(f"Pre-trained M0_dual: TMM val MAE={eval_model(pt_m0,to_dl_tmm(tmm_vl_rows,False),False,True)*100:.2f}%", flush=True)

set_seed(42); pt_mp=MPhys_dual(geo_dim,n_phys).to(device)
pt_mp=train_model(pt_mp,to_dl_tmm(tmm_tr_rows,True,shuffle=True),to_dl_tmm(tmm_vl_rows,True),500,1e-3,True,True)
print(f"Pre-trained MPhys_dual: TMM val MAE={eval_model(pt_mp,to_dl_tmm(tmm_vl_rows,True),True,True)*100:.2f}%", flush=True)

torch.save(pt_m0.state_dict(),"results/pretrained_m0_tmm_C.pt")
torch.save(pt_mp.state_dict(),"results/pretrained_mphys_tmm_C.pt")

# ===== Step 3: RCWA data =====
print("\n=== Step 3: Load RCWA data ===", flush=True)
data=np.load('data/raw/struct_C_500.npz',allow_pickle=True)
params_r=data['params'].astype(np.float32)
A_TE_r=data['A_TE'].astype(np.float32); R_TE_r=data['R_TE'].astype(np.float32)
A_TM_r=data['A_TM'].astype(np.float32); R_TM_r=data['R_TM'].astype(np.float32)
good=np.all(A_TE_r>=-0.01,axis=1)&np.all(A_TM_r>=-0.01,axis=1); gi=np.where(good)[0]
params_r=params_r[gi]; A_TE_r=np.clip(A_TE_r[gi],0,1); R_TE_r=np.clip(R_TE_r[gi],0,1)
A_TM_r=np.clip(A_TM_r[gi],0,1); R_TM_r=np.clip(R_TM_r[gi],0,1); N_r=len(gi)
print(f"RCWA: {N_r} good samples", flush=True)

phys_r=compute_physics_features_C(params_r.astype(np.float64),wavelengths.astype(np.float64),'Cr')
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

def make_dl(rows,has_phys,bs=2048,shuffle=False):
    xg=torch.tensor(X_geo_r[rows]).to(device)
    ate=torch.tensor(Ate_r[rows]).to(device); rte=torch.tensor(Rte_r[rows]).to(device)
    atm=torch.tensor(Atm_r[rows]).to(device); rtm=torch.tensor(Rtm_r[rows]).to(device)
    if has_phys: p=torch.tensor(X_phys_r_n[rows]).to(device); return DataLoader(TensorDataset(xg,ate,rte,atm,rtm,p),bs,shuffle=shuffle)
    return DataLoader(TensorDataset(xg,ate,rte,atm,rtm),bs,shuffle=shuffle)

dl_te_m0=make_dl(test_rows,False); dl_te_ph=make_dl(test_rows,True)
dl_vl_m0=make_dl(val_rows,False); dl_vl_ph=make_dl(val_rows,True)

# ===== Step 4: 4-way =====
print("\n=== Step 4: 4-way comparison ===", flush=True)
TRAIN_SIZES=[50,100,200,350]; SEEDS=[42,123,777]
results={sz:{'M0':[],'M_phys':[],'M_TL':[],'M_TL+phys':[]} for sz in TRAIN_SIZES}

for n_train in TRAIN_SIZES:
    for seed in SEEDS:
        print(f"\n--- n_train={n_train}, seed={seed} ---", flush=True)
        rng2=np.random.default_rng(seed); tr_idx=remaining[rng2.permutation(len(remaining))[:n_train]]
        tr_rows=get_rows(tr_idx)
        dl_tr_m0=make_dl(tr_rows,False,bs=512,shuffle=True); dl_tr_ph=make_dl(tr_rows,True,bs=512,shuffle=True)

        set_seed(seed); m0=M0_dual(geo_dim).to(device)
        m0=train_model(m0,dl_tr_m0,dl_vl_m0,1000,1e-3,False,True)
        results[n_train]['M0'].append(eval_model(m0,dl_te_m0,False,True))
        print(f"  M0: {results[n_train]['M0'][-1]*100:.3f}%", flush=True)

        set_seed(seed); mp=MPhys_dual(geo_dim,n_phys).to(device)
        mp=train_model(mp,dl_tr_ph,dl_vl_ph,1000,1e-3,True,True)
        results[n_train]['M_phys'].append(eval_model(mp,dl_te_ph,True,True))
        print(f"  M_phys: {results[n_train]['M_phys'][-1]*100:.3f}%", flush=True)

        set_seed(seed); m_tl=M0_dual(geo_dim).to(device); m_tl.load_state_dict(deepcopy(pt_m0.state_dict()))
        m_tl=train_model(m_tl,dl_tr_m0,dl_vl_m0,1000,3e-4,False,True)
        results[n_train]['M_TL'].append(eval_model(m_tl,dl_te_m0,False,True))
        print(f"  M_TL: {results[n_train]['M_TL'][-1]*100:.3f}%", flush=True)

        set_seed(seed); m_tlp=MPhys_dual(geo_dim,n_phys).to(device); m_tlp.load_state_dict(deepcopy(pt_mp.state_dict()))
        m_tlp=train_model(m_tlp,dl_tr_ph,dl_vl_ph,1000,3e-4,True,True)
        results[n_train]['M_TL+phys'].append(eval_model(m_tlp,dl_te_ph,True,True))
        print(f"  M_TL+phys: {results[n_train]['M_TL+phys'][-1]*100:.3f}%", flush=True)

print('\n'+'='*70, flush=True)
print('PBTL 4-way: Structure C (Dual-Polarization)', flush=True)
print('='*70, flush=True)
print(f'{"n":>6} | {"M0":>10} | {"M_phys":>10} | {"M_TL":>10} | {"M_TL+phys":>12}', flush=True)
print('-'*70, flush=True)
for sz in TRAIN_SIZES:
    r=results[sz]
    if not r['M0']: continue
    print(f"{sz:>6} | {np.mean(r['M0'])*100:>8.2f}% | {np.mean(r['M_phys'])*100:>8.2f}% | {np.mean(r['M_TL'])*100:>8.2f}% | {np.mean(r['M_TL+phys'])*100:>10.2f}%", flush=True)

os.makedirs('results', exist_ok=True)
np.savez('results/pbtl_C.npz', train_sizes=TRAIN_SIZES, seeds=SEEDS,
         M0=[results[sz]['M0'] for sz in TRAIN_SIZES], M_phys=[results[sz]['M_phys'] for sz in TRAIN_SIZES],
         M_TL=[results[sz]['M_TL'] for sz in TRAIN_SIZES], M_TL_phys=[results[sz]['M_TL+phys'] for sz in TRAIN_SIZES])
print("Results saved: pbtl_C.npz", flush=True)
