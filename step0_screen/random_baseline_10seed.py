#!/usr/bin/env python3
"""
Random baseline experiment for Structure A.
Tests whether TMM pre-training provides genuine benefit vs random pre-training.

5-way comparison:
  M0:         no pre-train, no physics features
  M_phys:     no pre-train, physics features
  M_TL:       TMM pre-train, no physics features
  M_TL+phys:  TMM pre-train, physics features
  M_rand:     RANDOM pre-train, no physics features  ← new baseline
  M_rand+phys:RANDOM pre-train, physics features     ← new baseline

If M_rand ≈ M_TL → TMM pre-train has no real value (just more data)
If M_rand >> M_TL → TMM pre-train genuinely helps (physics knowledge transfer)
"""
import sys, os, time
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from copy import deepcopy
from src.utils.seed_utils import set_global_seed as set_seed
from src.utils.data_utils import normalize_params, get_bounds
from src.utils.physics_features import compute_physics_features_A

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}", flush=True)

# ===== Model definitions (same as pbtl_experiment.py) =====
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
    def __init__(self, d):
        super().__init__()
        self.bb = BaseResNet(d)
        self.head = nn.Sequential(nn.Linear(256,128), nn.SiLU(), nn.Linear(128,1), nn.Sigmoid())
    def forward(self, x, **kw):
        R = self.head(self.bb(x)).squeeze(-1); return {'A': 1-R, 'R': R}

class MPhys(nn.Module):
    def __init__(self, gd, pd):
        super().__init__()
        self.bb = BaseResNet(gd+pd)
        self.head = nn.Sequential(nn.Linear(256,128), nn.SiLU(), nn.Linear(128,1), nn.Sigmoid())
    def forward(self, x, p=None, **kw):
        h = self.bb(torch.cat([x,p],-1)); R = self.head(h).squeeze(-1); return {'A': 1-R, 'R': R}

def train_model(model, dl_tr, dl_vl, epochs, lr, has_phys=False):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    crit = nn.MSELoss(); best_vl, best_st = float('inf'), None
    for ep in range(epochs):
        model.train()
        for batch in dl_tr:
            if has_phys: x,a,r,p=batch; out=model(x,p=p)
            else: x,a,r=batch; out=model(x)
            loss = crit(out['A'],a)+crit(out['R'],r)
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
        if (ep+1)%100==0:
            model.eval()
            with torch.no_grad():
                vl=0; vn=0
                for batch in dl_vl:
                    if has_phys: x,a,r,p=batch; out=model(x,p=p)
                    else: x,a,r=batch; out=model(x)
                    vl+=(nn.functional.l1_loss(out['A'],a,reduction='sum')+
                         nn.functional.l1_loss(out['R'],r,reduction='sum')).item(); vn+=len(a)*2
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

# ===== Step 1: Prepare random pre-train data =====
print("\n=== Step 1: Generate RANDOM pre-train data ===", flush=True)
N_RAND = 5000
wavelengths = np.linspace(380, 780, 100).astype(np.float32)
Nlam = len(wavelengths)
_, bounds_min, bounds_max = get_bounds("A")
rng = np.random.default_rng(99)

# Random params (same distribution as TMM)
params_rand = rng.uniform(bounds_min, bounds_max, (N_RAND, 10)).astype(np.float32)

# RANDOM spectra: uniform [0,1], no physics at all
A_rand = rng.uniform(0, 1, (N_RAND, Nlam)).astype(np.float32)
R_rand = rng.uniform(0, 1-A_rand, (N_RAND, Nlam)).astype(np.float32)  # R <= 1-A
print(f"Random data: A range [{A_rand.min():.3f}, {A_rand.max():.3f}]", flush=True)

# Physics features for random params
phys_rand = compute_physics_features_A(params_rand, wavelengths, "Cr")
n_phys = phys_rand.shape[-1]

params_rand_norm = normalize_params(params_rand, "A")
wl_norm = (wavelengths - wavelengths.min()) / (wavelengths.max() - wavelengths.min())
geo_dim = 1 + 10

params_rep = np.repeat(params_rand_norm[:,None,:], Nlam, axis=1)
wl_rep = np.tile(wl_norm[None,:,None], (N_RAND,1,1))
X_geo_rand = np.concatenate([wl_rep, params_rep], axis=-1).reshape(-1, geo_dim).astype(np.float32)
X_phys_rand = phys_rand.reshape(-1, n_phys).astype(np.float32)
pm = X_phys_rand.mean(0, keepdims=True); ps = X_phys_rand.std(0, keepdims=True)+1e-8
X_phys_rand_n = ((X_phys_rand-pm)/ps).astype(np.float32)
A_rand_flat = A_rand.reshape(-1); R_rand_flat = R_rand.reshape(-1)

n_tr = int(N_RAND*0.9); rand_idx = rng.permutation(N_RAND)
rand_tr_rows = np.concatenate([np.arange(i*Nlam,(i+1)*Nlam) for i in rand_idx[:n_tr]])
rand_vl_rows = np.concatenate([np.arange(i*Nlam,(i+1)*Nlam) for i in rand_idx[n_tr:]])

def to_dl_rand(rows, has_phys, bs=2048, shuffle=False):
    xg=torch.tensor(X_geo_rand[rows]).to(device)
    a=torch.tensor(A_rand_flat[rows]).to(device); r=torch.tensor(R_rand_flat[rows]).to(device)
    if has_phys: p=torch.tensor(X_phys_rand_n[rows]).to(device); return DataLoader(TensorDataset(xg,a,r,p),bs,shuffle=shuffle)
    return DataLoader(TensorDataset(xg,a,r),bs,shuffle=shuffle)

# ===== Step 2: Pre-train on random data =====
print("\n=== Step 2: Pre-train on RANDOM data ===", flush=True)
set_seed(42)
pt_rand_m0 = M0(geo_dim).to(device)
pt_rand_m0 = train_model(pt_rand_m0, to_dl_rand(rand_tr_rows,False,shuffle=True),
                         to_dl_rand(rand_vl_rows,False), 500, 1e-3, False)
print(f"Pre-trained M_rand M0: val MAE={eval_model(pt_rand_m0, to_dl_rand(rand_vl_rows,False))*100:.2f}%", flush=True)

set_seed(42)
pt_rand_mp = MPhys(geo_dim, n_phys).to(device)
pt_rand_mp = train_model(pt_rand_mp, to_dl_rand(rand_tr_rows,True,shuffle=True),
                         to_dl_rand(rand_vl_rows,True), 500, 1e-3, True)
print(f"Pre-trained M_rand MPhys: val MAE={eval_model(pt_rand_mp, to_dl_rand(rand_vl_rows,True), True)*100:.2f}%", flush=True)

# ===== Step 3: Load RCWA data =====
print("\n=== Step 3: Load RCWA data ===", flush=True)
data = np.load("data/raw/struct_A_vis_500.npz", allow_pickle=True)
params_rcwa = data["params"].astype(np.float32)
A_rcwa = data["A"].astype(np.float32); R_rcwa = data["R"].astype(np.float32)
good = np.all((A_rcwa>=0)&(A_rcwa<=1)&(R_rcwa>=0)&(R_rcwa<=1), axis=1)
gi = np.where(good)[0]
params_rcwa=params_rcwa[gi]; A_rcwa=A_rcwa[gi]; R_rcwa=R_rcwa[gi]; N_r=len(gi)
print(f"RCWA: {N_r} good samples", flush=True)

phys_rcwa = compute_physics_features_A(params_rcwa, wavelengths, "Cr")
params_rcwa_norm = normalize_params(params_rcwa, "A")
params_rep_r = np.repeat(params_rcwa_norm[:,None,:], Nlam, axis=1)
wl_rep_r = np.tile(wl_norm[None,:,None], (N_r,1,1))
X_geo_r = np.concatenate([wl_rep_r, params_rep_r], axis=-1).reshape(-1, geo_dim).astype(np.float32)
X_phys_r = phys_rcwa.reshape(-1, n_phys).astype(np.float32)
X_phys_r_n = ((X_phys_r-pm)/ps).astype(np.float32)
A_r_flat=A_rcwa.reshape(-1); R_r_flat=R_rcwa.reshape(-1)

def get_rows(si): return np.concatenate([np.arange(i*Nlam,(i+1)*Nlam) for i in si])
rng_sp=np.random.default_rng(42); all_idx=rng_sp.permutation(N_r)
N_TEST=50; N_VAL=50
test_idx=all_idx[-N_TEST:]; val_idx=all_idx[-(N_TEST+N_VAL):-N_TEST]; remaining=all_idx[:-(N_TEST+N_VAL)]
test_rows=get_rows(test_idx); val_rows=get_rows(val_idx)

# Load pre-trained TMM weights (from pbtl_experiment.py)
pt_tmm_m0 = M0(geo_dim).to(device)
pt_tmm_mp = MPhys(geo_dim, n_phys).to(device)
pt_tmm_m0.load_state_dict(torch.load("results/pretrained_m0_tmm.pt", map_location=device))
pt_tmm_mp.load_state_dict(torch.load("results/pretrained_mphys_tmm.pt", map_location=device))
print("Loaded pre-trained TMM weights", flush=True)

def make_dl(rows, has_phys, bs=2048, shuffle=False):
    xg=torch.tensor(X_geo_r[rows]).to(device)
    a=torch.tensor(A_r_flat[rows]).to(device); r=torch.tensor(R_r_flat[rows]).to(device)
    if has_phys: p=torch.tensor(X_phys_r_n[rows]).to(device); return DataLoader(TensorDataset(xg,a,r,p),bs,shuffle=shuffle)
    return DataLoader(TensorDataset(xg,a,r),bs,shuffle=shuffle)

dl_te_m0=make_dl(test_rows,False); dl_te_ph=make_dl(test_rows,True)
dl_vl_m0=make_dl(val_rows,False); dl_vl_ph=make_dl(val_rows,True)

# ===== Step 4: 6-way comparison =====
print("\n=== Step 4: 6-way comparison ===", flush=True)
TRAIN_SIZES=[50,100,200,350]; SEEDS=[42,123,777,321,456,654,999,111,222,333]
results={sz:{'M0':[],'M_phys':[],'M_TL':[],'M_TL+phys':[],'M_rand':[],'M_rand+phys':[]} for sz in TRAIN_SIZES}

for n_train in TRAIN_SIZES:
    for seed in SEEDS:
        print(f"\n--- n_train={n_train}, seed={seed} ---", flush=True)
        rng2=np.random.default_rng(seed); tr_idx=remaining[rng2.permutation(len(remaining))[:n_train]]
        tr_rows=get_rows(tr_idx)
        dl_tr_m0=make_dl(tr_rows,False,bs=512,shuffle=True)
        dl_tr_ph=make_dl(tr_rows,True,bs=512,shuffle=True)

        # M0: scratch
        set_seed(seed); m0=M0(geo_dim).to(device)
        m0=train_model(m0, dl_tr_m0, dl_vl_m0, 1000, 1e-3, False)
        results[n_train]['M0'].append(eval_model(m0, dl_te_m0, False))
        print(f"  M0:          {results[n_train]['M0'][-1]*100:.3f}%", flush=True)

        # M_phys: scratch + physics features
        set_seed(seed); mp=MPhys(geo_dim, n_phys).to(device)
        mp=train_model(mp, dl_tr_ph, dl_vl_ph, 1000, 1e-3, True)
        results[n_train]['M_phys'].append(eval_model(mp, dl_te_ph, True))
        print(f"  M_phys:      {results[n_train]['M_phys'][-1]*100:.3f}%", flush=True)

        # M_TL: TMM pre-train
        set_seed(seed); m_tl=M0(geo_dim).to(device); m_tl.load_state_dict(deepcopy(pt_tmm_m0.state_dict()))
        m_tl=train_model(m_tl, dl_tr_m0, dl_vl_m0, 1000, 3e-4, False)
        results[n_train]['M_TL'].append(eval_model(m_tl, dl_te_m0, False))
        print(f"  M_TL:        {results[n_train]['M_TL'][-1]*100:.3f}%", flush=True)

        # M_TL+phys: TMM pre-train + physics features
        set_seed(seed); m_tlp=MPhys(geo_dim, n_phys).to(device); m_tlp.load_state_dict(deepcopy(pt_tmm_mp.state_dict()))
        m_tlp=train_model(m_tlp, dl_tr_ph, dl_vl_ph, 1000, 3e-4, True)
        results[n_train]['M_TL+phys'].append(eval_model(m_tlp, dl_te_ph, True))
        print(f"  M_TL+phys:   {results[n_train]['M_TL+phys'][-1]*100:.3f}%", flush=True)

        # M_rand: RANDOM pre-train (key baseline!)
        set_seed(seed); m_rand=M0(geo_dim).to(device); m_rand.load_state_dict(deepcopy(pt_rand_m0.state_dict()))
        m_rand=train_model(m_rand, dl_tr_m0, dl_vl_m0, 1000, 3e-4, False)
        results[n_train]['M_rand'].append(eval_model(m_rand, dl_te_m0, False))
        print(f"  M_rand:      {results[n_train]['M_rand'][-1]*100:.3f}%", flush=True)

        # M_rand+phys: RANDOM pre-train + physics features
        set_seed(seed); m_randp=MPhys(geo_dim, n_phys).to(device); m_randp.load_state_dict(deepcopy(pt_rand_mp.state_dict()))
        m_randp=train_model(m_randp, dl_tr_ph, dl_vl_ph, 1000, 3e-4, True)
        results[n_train]['M_rand+phys'].append(eval_model(m_randp, dl_te_ph, True))
        print(f"  M_rand+phys: {results[n_train]['M_rand+phys'][-1]*100:.3f}%", flush=True)

# ===== Summary =====
print('\n'+'='*80, flush=True)
print('RANDOM BASELINE: Structure A', flush=True)
print('='*80, flush=True)
print(f'{"n":>6} | {"M0":>8} | {"M_phys":>8} | {"M_TL":>8} | {"M_TL+p":>8} | {"M_rand":>8} | {"M_rand+p":>9}', flush=True)
print('-'*80, flush=True)
for sz in TRAIN_SIZES:
    r=results[sz]
    if not r['M0']: continue
    print(f"{sz:>6} | {np.mean(r['M0'])*100:>6.2f}% | {np.mean(r['M_phys'])*100:>6.2f}% | "
          f"{np.mean(r['M_TL'])*100:>6.2f}% | {np.mean(r['M_TL+phys'])*100:>6.2f}% | "
          f"{np.mean(r['M_rand'])*100:>6.2f}% | {np.mean(r['M_rand+phys'])*100:>7.2f}%", flush=True)

print('\nKey question: Is M_TL << M_rand? (TMM provides genuine physics transfer)', flush=True)
for sz in TRAIN_SIZES:
    r=results[sz]
    if not r['M_TL']: continue
    tl=np.mean(r['M_TL'])*100; rand=np.mean(r['M_rand'])*100
    gap=rand-tl
    print(f"  n={sz:3d}: M_TL={tl:.2f}% vs M_rand={rand:.2f}% → gap={gap:+.2f}% {'← TMM genuinely helps!' if gap>0.5 else '← marginal'}", flush=True)

os.makedirs('results', exist_ok=True)
np.savez('results/random_baseline_10seed.npz',
         train_sizes=TRAIN_SIZES, seeds=SEEDS,
         M0=[results[sz]['M0'] for sz in TRAIN_SIZES],
         M_phys=[results[sz]['M_phys'] for sz in TRAIN_SIZES],
         M_TL=[results[sz]['M_TL'] for sz in TRAIN_SIZES],
         M_TL_phys=[results[sz]['M_TL+phys'] for sz in TRAIN_SIZES],
         M_rand=[results[sz]['M_rand'] for sz in TRAIN_SIZES],
         M_rand_phys=[results[sz]['M_rand+phys'] for sz in TRAIN_SIZES])
print("Saved: random_baseline_10seed.npz", flush=True)
