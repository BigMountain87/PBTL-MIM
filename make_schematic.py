"""Generate clean schematic figures for three MIM absorber structures.

Design changes vs v1:
  - Each cross-section shows ONE unit cell (period bracket aligned)
  - Layer thicknesses labelled with offset tick marks (no box-overlap)
  - Top-view dimension arrows placed outside the unit cell box
  - Larger fonts (TXT 10, LBL 11, TITLE 12)
  - Compact subplot spacing for manuscript embedding
"""
from __future__ import annotations
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch

# Publication-grade rcParams — UNIFIED across Figures 1–5
# (Optics Communications guide: TrueType / Type 42)
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans'],
    'mathtext.fontset': 'dejavusans',
    'mathtext.default': 'regular',
    'axes.linewidth': 0.9,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'pdf.fonttype': 42,   # TrueType (avoids Type 3 reader-compat issues)
    'ps.fonttype': 42,
})

# ─── Colors ───
C_CR      = '#708090'
C_SIO2    = '#87CEEB'
C_TIO2    = '#DDA0DD'
C_GLASS   = '#F5F5DC'
C_AIR     = '#FFFFFF'
C_CR_DARK = '#4A5568'
C_DIM     = '#1f4e79'   # dimension arrow / label
C_RED     = '#C00000'   # incident light
C_TE      = '#1a73e8'   # TE polarization
C_TM      = '#d93025'   # TM polarization

TXT_FS   = 10
LBL_FS   = 11
TITLE_FS = 12
LEG_FS   = 9

# Cross-section panel dimensions (single unit cell)
W_CELL = 5.0           # logical unit-cell width
LX0, LX1 = -0.9, 6.9   # x-axis bounds (room for thickness labels)

# Top-view panel dimensions (range matched to ylim 7.35 → square panel)
TX0, TX1 = -0.9, 6.45

fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.5),
                         gridspec_kw={'height_ratios': [9.4, 6.7],
                                       'hspace': -0.30, 'wspace': 0.30})

LBL_MIRROR = 'Cr mirror'     # short to fit inside the unit-cell box


def thickness_tick(ax, x_box_right, y0, y1, label, lbl_offset=0.55):
    """Tick mark + thickness label outside the box, on the right."""
    x_tick = x_box_right + 0.45
    ax.plot([x_box_right, x_tick], [y0, y0], color=C_DIM, lw=0.8)
    ax.plot([x_box_right, x_tick], [y1, y1], color=C_DIM, lw=0.8)
    ax.annotate('', xy=(x_tick, y0), xytext=(x_tick, y1),
                arrowprops=dict(arrowstyle='<->', color=C_DIM, lw=1.1,
                                mutation_scale=12))
    ax.text(x_tick + lbl_offset, (y0 + y1) / 2, label,
            fontsize=LBL_FS, ha='left', va='center', color=C_DIM,
            fontweight='bold')


def period_bracket(ax, x0, x1, y, label='$P$'):
    """Period bracket below the cross-section (aligned with one unit cell)."""
    ax.annotate('', xy=(x0, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle='<->', color='black', lw=1.1))
    ax.plot([x0, x0], [y - 0.12, y + 0.12], color='black', lw=0.9)
    ax.plot([x1, x1], [y - 0.12, y + 0.12], color='black', lw=0.9)
    ax.text((x0 + x1) / 2, y - 0.30, label, ha='center', va='top',
            fontsize=LBL_FS + 1)


def incident_light(ax, x_tip, y_tip, label=r'$\theta$', color=C_RED, dy=0.95,
                   dx=-0.42, lbl_dx=-0.22, lbl_dy=-0.02):
    """Slanted incidence arrow + angle label."""
    ax.annotate('', xy=(x_tip, y_tip), xytext=(x_tip + dx, y_tip + dy),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.6))
    ax.text(x_tip + dx + lbl_dx, y_tip + dy + lbl_dy, label,
            fontsize=LBL_FS, color=color, fontweight='bold')


def panel_title(ax, title, x_data=0.0, ha='left', y=0.93):
    """Panel title anchored to a chosen data-position above the panel."""
    x0, x1 = ax.get_xlim()
    x_axes = (x_data - x0) / (x1 - x0)
    ax.text(x_axes, y, title, transform=ax.transAxes,
            fontsize=TITLE_FS, fontweight='bold', ha=ha, va='bottom',
            clip_on=False)


# ═══════════════════════════════════════════════════════════════════
# (a) Structure A — Asymmetric Dual-Dielectric Dual-Cavity
# ═══════════════════════════════════════════════════════════════════
ax = axes[0, 0]
ax.set_xlim(LX0, LX1); ax.set_ylim(-1.7, 6.6); ax.set_aspect('equal', anchor='N'); ax.axis('off')
panel_title(ax, '(a) Structure A\nAsymmetric Dual-Dielectric Dual-Cavity', x_data=W_CELL / 2, ha='center')
yA = 0.0

# Glass
ax.add_patch(Rectangle((0, yA + 0), W_CELL, 0.68, fc=C_GLASS, ec='k', lw=0.8))
ax.text(W_CELL / 2, yA + 0.34, 'Glass', ha='center', va='center',
        fontsize=TXT_FS, style='italic')

# Cr mirror (100 nm)
y0 = yA + 0.68; h = 0.58
ax.add_patch(Rectangle((0, y0), W_CELL, h, fc=C_CR_DARK, ec='k', lw=0.8))
ax.text(W_CELL / 2, y0 + h / 2, LBL_MIRROR, ha='center', va='center',
        fontsize=TXT_FS, color='white')

# Cr patterned (square, t2)
y_t2 = y0 + h; h_t2 = 0.38
ax.add_patch(Rectangle((1.30, y_t2), 2.4, h_t2, fc=C_CR, ec='k', lw=0.8))
ax.add_patch(Rectangle((0, y_t2), 1.30, h_t2, fc=C_AIR, ec='gray', lw=0.5, ls='--'))
ax.add_patch(Rectangle((3.70, y_t2), 1.30, h_t2, fc=C_AIR, ec='gray', lw=0.5, ls='--'))
thickness_tick(ax, W_CELL, y_t2, y_t2 + h_t2, '$t_2$')

# TiO2 (d2)
y_tio2 = y_t2 + h_t2; h_tio2 = 0.70
ax.add_patch(Rectangle((0, y_tio2), W_CELL, h_tio2, fc=C_TIO2, ec='k', lw=0.8))
ax.text(W_CELL / 2, y_tio2 + h_tio2 / 2, r'TiO$_2$ ($d_2$)', ha='center', va='center',
        fontsize=TXT_FS)

# Cr middle (t_mid)
y_mid = y_tio2 + h_tio2; h_mid = 0.38
ax.add_patch(Rectangle((0, y_mid), W_CELL, h_mid, fc=C_CR, ec='k', lw=0.8))
ax.text(W_CELL / 2, y_mid + h_mid / 2, 'Cr',
        ha='center', va='center', fontsize=TXT_FS, color='white')
# Optional t_mid thickness label outside the box
thickness_tick(ax, W_CELL, y_mid, y_mid + h_mid, r'$t_{\mathrm{mid}}$',
               lbl_offset=0.55)

# SiO2 (d1)
y_sio2 = y_mid + h_mid; h_sio2 = 0.70
ax.add_patch(Rectangle((0, y_sio2), W_CELL, h_sio2, fc=C_SIO2, ec='k', lw=0.8))
ax.text(W_CELL / 2, y_sio2 + h_sio2 / 2, r'SiO$_2$ ($d_1$)', ha='center', va='center',
        fontsize=TXT_FS)

# Cr patterned (rect, t1)
y_t1 = y_sio2 + h_sio2; h_t1 = 0.38
ax.add_patch(Rectangle((0.5, y_t1), 4.0, h_t1, fc=C_CR, ec='k', lw=0.8))
ax.add_patch(Rectangle((0, y_t1), 0.5, h_t1, fc=C_AIR, ec='gray', lw=0.5, ls='--'))
ax.add_patch(Rectangle((4.5, y_t1), 0.5, h_t1, fc=C_AIR, ec='gray', lw=0.5, ls='--'))
thickness_tick(ax, W_CELL, y_t1, y_t1 + h_t1, '$t_1$')

# Air label
ax.text(W_CELL / 2, y_t1 + h_t1 + 0.55, 'Air', ha='center', va='center',
        fontsize=TXT_FS, style='italic')

# Incident light
incident_light(ax, x_tip=1.35, y_tip=y_t1 + h_t1 + 0.04, label=r'$\theta$',
               dy=0.72, dx=-0.34, lbl_dx=-0.14, lbl_dy=-0.06)

# Period bracket (single unit cell)
period_bracket(ax, 0, W_CELL, yA - 0.45, '$P$')
ax.text(W_CELL / 2, -1.30, 'cross-section', ha='center', va='top',
        fontsize=TITLE_FS, fontweight='bold')

# ─── Top view (a) ───
ax = axes[1, 0]
ax.set_xlim(LX0, LX1); ax.set_ylim(-1.7, 6.55); ax.set_aspect('equal', anchor='N'); ax.axis('off')
# Unit cell
ax.add_patch(Rectangle((0, 0), W_CELL, W_CELL, fc='#f7f7f7', ec='k', lw=1.0, ls='--'))
# Cr rect (top patterned)
ax.add_patch(Rectangle((0.7, 0.9), 3.6, 3.2, fc=C_CR, ec='k', lw=1.0))
ax.text(2.5, 2.5, 'Cr (rect)', ha='center', va='center',
        fontsize=TXT_FS, color='white', fontweight='bold')

# Wx (above the box, with end ticks)
yx = 5.6
ax.annotate('', xy=(0.7, yx), xytext=(4.3, yx),
            arrowprops=dict(arrowstyle='<->', color=C_DIM, lw=1.0))
ax.plot([0.7, 0.7], [4.1, yx + 0.1], color=C_DIM, lw=0.6)
ax.plot([4.3, 4.3], [4.1, yx + 0.1], color=C_DIM, lw=0.6)
ax.text(2.5, yx + 0.30, '$W_x$', ha='center', fontsize=LBL_FS, color=C_DIM)

# Wy (right of the box)
xy_x = 5.6
ax.annotate('', xy=(xy_x, 0.9), xytext=(xy_x, 4.1),
            arrowprops=dict(arrowstyle='<->', color=C_DIM, lw=1.0))
ax.plot([4.3, xy_x + 0.1], [0.9, 0.9], color=C_DIM, lw=0.6)
ax.plot([4.3, xy_x + 0.1], [4.1, 4.1], color=C_DIM, lw=0.6)
ax.text(xy_x + 0.30, 2.5, '$W_y$', ha='left', va='center', fontsize=LBL_FS, color=C_DIM)

# Period (below)
period_bracket(ax, 0, W_CELL, -0.6, '$P$')
ax.text(W_CELL / 2, -1.40, 'top view', ha='center', va='top',
        fontsize=TITLE_FS, fontweight='bold')


# ═══════════════════════════════════════════════════════════════════
# (b) Structure B — Ring–Disk Fano Resonance
# ═══════════════════════════════════════════════════════════════════
ax = axes[0, 1]
ax.set_xlim(LX0, LX1); ax.set_ylim(-1.7, 6.6); ax.set_aspect('equal', anchor='N'); ax.axis('off')
panel_title(ax, '(b) Structure B\nRing-Disk Fano Resonance', x_data=W_CELL / 2, ha='center')

# Glass
ax.add_patch(Rectangle((0, 0), W_CELL, 1.0, fc=C_GLASS, ec='k', lw=0.8))
ax.text(W_CELL / 2, 0.5, 'Glass', ha='center', va='center',
        fontsize=TXT_FS, style='italic')

# Cr mirror
y0 = 1.0; h = 0.8
ax.add_patch(Rectangle((0, y0), W_CELL, h, fc=C_CR_DARK, ec='k', lw=0.8))
ax.text(W_CELL / 2, y0 + h / 2, LBL_MIRROR, ha='center', va='center',
        fontsize=TXT_FS, color='white')

# SiO2 spacer
y_s = y0 + h; h_s = 1.4
ax.add_patch(Rectangle((0, y_s), W_CELL, h_s, fc=C_SIO2, ec='k', lw=0.8))
ax.text(W_CELL / 2, y_s + h_s / 2, r'SiO$_2$ ($d_{\mathrm{SiO_2}}$)',
        ha='center', va='center', fontsize=TXT_FS)

# Ring + disk patterned Cr layer
y_cr = y_s + h_s; h_cr = 0.6
# Ring outer (two thin bars)
ax.add_patch(Rectangle((0.50, y_cr), 0.70, h_cr, fc=C_CR, ec='k', lw=0.8))
ax.add_patch(Rectangle((3.80, y_cr), 0.70, h_cr, fc=C_CR, ec='k', lw=0.8))
# Disk (centred bar with hatch)
ax.add_patch(Rectangle((1.75, y_cr), 1.50, h_cr, fc=C_CR, ec='k', lw=0.8, hatch='///'))
# Air gaps
ax.add_patch(Rectangle((0, y_cr),     0.50, h_cr, fc=C_AIR, ec='gray', lw=0.5, ls='--'))
ax.add_patch(Rectangle((1.20, y_cr),  0.55, h_cr, fc=C_AIR, ec='gray', lw=0.5, ls='--'))
ax.add_patch(Rectangle((3.25, y_cr),  0.55, h_cr, fc=C_AIR, ec='gray', lw=0.5, ls='--'))
ax.add_patch(Rectangle((4.50, y_cr),  0.50, h_cr, fc=C_AIR, ec='gray', lw=0.5, ls='--'))
# Subtle role labels (above thin bars + disk)
ax.text(0.85, y_cr + h_cr + 0.15, 'ring', ha='center', fontsize=8, color='gray', style='italic')
ax.text(2.50, y_cr + h_cr + 0.15, 'disk', ha='center', fontsize=8, color='gray', style='italic')
ax.text(4.15, y_cr + h_cr + 0.15, 'ring', ha='center', fontsize=8, color='gray', style='italic')

thickness_tick(ax, W_CELL, y_cr, y_cr + h_cr, r'$t_{\mathrm{Cr}}$')

# Air
ax.text(W_CELL / 2, y_cr + h_cr + 0.65, 'Air', ha='center', va='center',
        fontsize=TXT_FS, style='italic')

# Incident light
incident_light(ax, x_tip=1.4, y_tip=y_cr + h_cr + 0.10, label=r'$\theta,\phi$')

period_bracket(ax, 0, W_CELL, -0.55, '$P$')
ax.text(W_CELL / 2, -1.30, 'cross-section', ha='center', va='top',
        fontsize=TITLE_FS, fontweight='bold')


# ─── Top view (b) ───
ax = axes[1, 1]
ax.set_xlim(LX0, LX1); ax.set_ylim(-1.7, 6.55); ax.set_aspect('equal', anchor='N'); ax.axis('off')
ax.add_patch(Rectangle((0, 0), W_CELL, W_CELL, fc='#f7f7f7', ec='k', lw=1.0, ls='--'))

# Outer ring
ax.add_patch(Circle((2.5, 2.5), 2.0, fc=C_CR, ec='k', lw=1.0))
# Inner cutout
ax.add_patch(Circle((2.5, 2.5), 1.4, fc='#f7f7f7', ec='k', lw=0.8))
# Disk
ax.add_patch(Circle((2.5, 2.5), 0.85, fc=C_CR, ec='k', lw=1.0, hatch='///'))
ax.text(2.5, 2.5, 'disk', ha='center', va='center', fontsize=8,
        color='white', fontweight='bold')

# R_out (top-right radial)
ang = np.pi / 4
ax.annotate('', xy=(2.5 + 2.0 * np.cos(ang), 2.5 + 2.0 * np.sin(ang)),
            xytext=(2.5, 2.5),
            arrowprops=dict(arrowstyle='->', color=C_DIM, lw=1.1))
ax.text(2.5 + 2.2 * np.cos(ang) + 0.15, 2.5 + 2.2 * np.sin(ang) + 0.15,
        r'$R_{\mathrm{out}}$', fontsize=LBL_FS, color=C_DIM)

# R_in (bottom-right radial)
ang = -np.pi / 4
ax.annotate('', xy=(2.5 + 1.4 * np.cos(ang), 2.5 + 1.4 * np.sin(ang)),
            xytext=(2.5, 2.5),
            arrowprops=dict(arrowstyle='->', color='#2e7d32', lw=1.1))
ax.text(2.5 + 1.5 * np.cos(ang) + 0.10, 2.5 + 1.5 * np.sin(ang) - 0.30,
        r'$R_{\mathrm{in}}$', fontsize=LBL_FS, color='#2e7d32')

# R_disk (left radial; label placed clearly outside the disk)
ax.annotate('', xy=(2.5 - 0.85, 2.5), xytext=(2.5, 2.5),
            arrowprops=dict(arrowstyle='->', color='#8e0000', lw=1.1))
ax.text(0.55, 2.5, r'$R_{\mathrm{disk}}$',
        fontsize=LBL_FS, color='#8e0000', ha='right', va='center')

period_bracket(ax, 0, W_CELL, -0.6, '$P$')
ax.text(W_CELL / 2, -1.40, 'top view', ha='center', va='top',
        fontsize=TITLE_FS, fontweight='bold')


# ═══════════════════════════════════════════════════════════════════
# (c) Structure C — Dual-Polarization Rectangular
# ═══════════════════════════════════════════════════════════════════
ax = axes[0, 2]
ax.set_xlim(LX0, LX1); ax.set_ylim(-1.7, 6.6); ax.set_aspect('equal', anchor='N'); ax.axis('off')
panel_title(ax, '(c) Structure C\nDual-Polarization Rectangular', x_data=W_CELL / 2, ha='center')

# Glass
ax.add_patch(Rectangle((0, 0), W_CELL, 1.0, fc=C_GLASS, ec='k', lw=0.8))
ax.text(W_CELL / 2, 0.5, 'Glass', ha='center', va='center',
        fontsize=TXT_FS, style='italic')

# Cr mirror
y0 = 1.0; h = 0.8
ax.add_patch(Rectangle((0, y0), W_CELL, h, fc=C_CR_DARK, ec='k', lw=0.8))
ax.text(W_CELL / 2, y0 + h / 2, LBL_MIRROR, ha='center', va='center',
        fontsize=TXT_FS, color='white')

# SiO2 spacer
y_s = y0 + h; h_s = 1.4
ax.add_patch(Rectangle((0, y_s), W_CELL, h_s, fc=C_SIO2, ec='k', lw=0.8))
ax.text(W_CELL / 2, y_s + h_s / 2, r'SiO$_2$ ($d_{\mathrm{SiO_2}}$)',
        ha='center', va='center', fontsize=TXT_FS)

# Cr rect patch
y_cr = y_s + h_s; h_cr = 0.6
ax.add_patch(Rectangle((0.5, y_cr), 4.0, h_cr, fc=C_CR, ec='k', lw=0.8))
ax.add_patch(Rectangle((0, y_cr),   0.5, h_cr, fc=C_AIR, ec='gray', lw=0.5, ls='--'))
ax.add_patch(Rectangle((4.5, y_cr), 0.5, h_cr, fc=C_AIR, ec='gray', lw=0.5, ls='--'))
thickness_tick(ax, W_CELL, y_cr, y_cr + h_cr, r'$t_{\mathrm{Cr}}$')

ax.text(W_CELL / 2, y_cr + h_cr + 0.65, 'Air', ha='center', va='center',
        fontsize=TXT_FS, style='italic')

# Incident light
incident_light(ax, x_tip=1.4, y_tip=y_cr + h_cr + 0.10, label=r'$\theta,\phi$')

# TE / TM polarization indicators (cross-section)
ax.text(3.25, y_cr + h_cr + 1.30, 'TE', fontsize=LBL_FS, color=C_TE, fontweight='bold')
ax.text(4.0,  y_cr + h_cr + 1.30, '/',  fontsize=LBL_FS, color='gray')
ax.text(4.25, y_cr + h_cr + 1.30, 'TM', fontsize=LBL_FS, color=C_TM, fontweight='bold')

period_bracket(ax, 0, W_CELL, -0.55, '$P$')
ax.text(W_CELL / 2, -1.30, 'cross-section', ha='center', va='top',
        fontsize=TITLE_FS, fontweight='bold')


# ─── Top view (c) ───
ax = axes[1, 2]
ax.set_xlim(LX0, LX1 + 0.6); ax.set_ylim(-1.7, 6.55); ax.set_aspect('equal', anchor='N'); ax.axis('off')
ax.add_patch(Rectangle((0, 0), W_CELL, W_CELL, fc='#f7f7f7', ec='k', lw=1.0, ls='--'))

# Cr rect
ax.add_patch(Rectangle((0.6, 0.9), 3.8, 3.2, fc=C_CR, ec='k', lw=1.0))
ax.text(2.5, 2.5, 'Cr (rect)', ha='center', va='center',
        fontsize=TXT_FS, color='white', fontweight='bold')

# Wx (above the box)
yx = 5.6
ax.annotate('', xy=(0.6, yx), xytext=(4.4, yx),
            arrowprops=dict(arrowstyle='<->', color=C_DIM, lw=1.0))
ax.plot([0.6, 0.6], [4.1, yx + 0.1], color=C_DIM, lw=0.6)
ax.plot([4.4, 4.4], [4.1, yx + 0.1], color=C_DIM, lw=0.6)
ax.text(2.5, yx + 0.30, '$W_x$', ha='center', fontsize=LBL_FS, color=C_DIM)

# Wy (right of the box)
xy_x = 5.6
ax.annotate('', xy=(xy_x, 0.9), xytext=(xy_x, 4.1),
            arrowprops=dict(arrowstyle='<->', color=C_DIM, lw=1.0))
ax.plot([4.4, xy_x + 0.1], [0.9, 0.9], color=C_DIM, lw=0.6)
ax.plot([4.4, xy_x + 0.1], [4.1, 4.1], color=C_DIM, lw=0.6)
ax.text(xy_x + 0.30, 2.5, '$W_y$', ha='left', va='center',
        fontsize=LBL_FS, color=C_DIM)

# Period
period_bracket(ax, 0, W_CELL, -0.6, '$P$')
ax.text(W_CELL / 2, -1.40, 'top view', ha='center', va='top',
        fontsize=TITLE_FS, fontweight='bold')

# TE / TM polarization arrows placed compactly to the right of the unit cell.
ax.annotate('', xy=(xy_x + 1.30, 4.6), xytext=(xy_x + 0.06, 4.6),
            arrowprops=dict(arrowstyle='<->', color=C_TE, lw=1.4))
ax.text(xy_x + 0.68, 4.88, 'TE', ha='center', fontsize=LBL_FS - 1,
        color=C_TE, fontweight='bold')
ax.annotate('', xy=(xy_x + 1.62, 5.22), xytext=(xy_x + 1.62, 3.98),
            arrowprops=dict(arrowstyle='<->', color=C_TM, lw=1.4))
ax.text(xy_x + 1.78, 4.6, 'TM', ha='left', va='center',
        fontsize=LBL_FS - 1, color=C_TM, fontweight='bold')


# ═══════════════════════════════════════════════════════════════════
# Legend
# ═══════════════════════════════════════════════════════════════════
legend_elements = [
    mpatches.Patch(facecolor=C_CR,      edgecolor='k', label='Cr (patterned)'),
    mpatches.Patch(facecolor=C_CR_DARK, edgecolor='k', label='Cr mirror'),
    mpatches.Patch(facecolor=C_SIO2,    edgecolor='k', label=r'SiO$_2$'),
    mpatches.Patch(facecolor=C_TIO2,    edgecolor='k', label=r'TiO$_2$'),
    mpatches.Patch(facecolor=C_GLASS,   edgecolor='k', label='Glass'),
]
fig.legend(handles=legend_elements, loc='lower center', ncol=5,
           fontsize=LBL_FS - 1, frameon=True, fancybox=True, shadow=False,
           bbox_to_anchor=(0.5, 0.07))

plt.savefig('figures/fig0_structure_schematic.pdf',
            bbox_inches='tight', dpi=300)
plt.savefig('figures/fig0_structure_schematic.png',
            bbox_inches='tight', dpi=250)
plt.savefig('figures/Figure_1.pdf',
            bbox_inches='tight', dpi=300)
plt.savefig('figures/Figure_1.png',
            bbox_inches='tight', dpi=250)
print('Schematic saved (Type 42 fonts, PNG 250 DPI).')
