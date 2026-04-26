import numpy as np
import torch
import random

def set_global_seed(seed: int):
    """모든 랜덤 시드를 동일하게 고정."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[Seed] Global seed set to {seed}")
