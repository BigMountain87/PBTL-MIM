"""R2 Q3: Structure-B spectral evolution vs ring-disk gap (coupling evidence)."""
import numpy as np, matplotlib.pyplot as plt
plt.rcParams.update({'font.size':9,'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],
 'mathtext.fontset':'dejavusans','axes.linewidth':0.8,'axes.labelsize':10,'pdf.fonttype':42,'ps.fonttype':42})
d=np.load('results/gap_sweep.npz',allow_pickle=True)
A=d['A']; gaps=d['gaps']; wl=d['wl']
order=np.argsort(gaps)
fig,ax=plt.subplots(figsize=(5.6,3.2))
cmap=plt.cm.viridis(np.linspace(0,0.9,len(gaps)))
for k,i in enumerate(order):
    if np.all(np.isfinite(A[i])):
        ax.plot(wl,A[i],color=cmap[k],lw=1.8,label=f'gap = {gaps[i]:.0f} nm')
ax.set_xlabel('Wavelength (nm)'); ax.set_ylabel('Absorptance'); ax.set_xlim(400,1800)
ax.set_title('Structure B: spectral evolution vs ring–disk gap',fontsize=10,fontweight='bold',loc='left')
ax.legend(fontsize=7.5,ncol=2,framealpha=0.9); ax.grid(True,alpha=0.3,lw=0.5)
plt.tight_layout(); plt.savefig('figures/Figure_S3.pdf',bbox_inches='tight',dpi=300); plt.savefig('figures/Figure_S3.png',bbox_inches='tight',dpi=250)
print('saved Figure_S3; gaps:',sorted([round(float(g),1) for g in gaps]),'A_max:',[round(float(np.nanmax(A[i])),3) for i in order])
