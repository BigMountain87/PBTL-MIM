# src/simulation/rcwa_utils.py
"""
Shared RCWA utility functions for TORCWA.
Optimized R/T computation using batch S_parameters.
"""

import torch
import numpy as np


def compute_RT_batch(sim, order):
    """
    Compute total R and T from solved TORCWA simulation using batch S_parameters.
    ~8x faster than per-order loop.
    
    For TE input (Ey, amplitude=[0,1]):
      R = sum |S_refl_yy|^2 + |S_refl_xy|^2
      T = sum |S_trans_yy|^2 + |S_trans_xy|^2
    
    Args:
        sim: solved torcwa.rcwa object
        order: [order_x, order_y]
    
    Returns:
        R_total, T_total (floats)
    """
    all_orders = [[ox, oy] for ox in range(-order[0], order[0]+1)
                           for oy in range(-order[1], order[1]+1)]
    
    R_total = 0.0
    T_total = 0.0
    for pol in ['yy', 'xy']:  # TE input components
        r_all = sim.S_parameters(orders=all_orders, direction='forward',
                                 port='reflection', polarization=pol, power_norm=True)
        t_all = sim.S_parameters(orders=all_orders, direction='forward',
                                 port='transmission', polarization=pol, power_norm=True)
        R_total += torch.sum(torch.abs(r_all)**2).item()
        T_total += torch.sum(torch.abs(t_all)**2).item()
    
    return R_total, T_total


def compute_RT_batch_TM(sim, order):
    """
    Same as compute_RT_batch but for TM input (Ex, amplitude=[1,0]).
    
    R = sum |S_refl_xx|^2 + |S_refl_yx|^2
    T = sum |S_trans_xx|^2 + |S_trans_yx|^2
    """
    all_orders = [[ox, oy] for ox in range(-order[0], order[0]+1)
                           for oy in range(-order[1], order[1]+1)]
    
    R_total = 0.0
    T_total = 0.0
    for pol in ['xx', 'yx']:  # TM input components
        r_all = sim.S_parameters(orders=all_orders, direction='forward',
                                 port='reflection', polarization=pol, power_norm=True)
        t_all = sim.S_parameters(orders=all_orders, direction='forward',
                                 port='transmission', polarization=pol, power_norm=True)
        R_total += torch.sum(torch.abs(r_all)**2).item()
        T_total += torch.sum(torch.abs(t_all)**2).item()
    
    return R_total, T_total
