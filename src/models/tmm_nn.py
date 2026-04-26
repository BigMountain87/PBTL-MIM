# src/models/tmm_nn.py
"""
TMM+NN surrogate models for Step 0 screening.
Simplified to M0 (Baseline ANN) and M7 (TMM+NN+sequential logit) for quick comparison.
"""

import torch
import torch.nn as nn

EPS_CLAMP = 0.01


class BaseResNet(nn.Module):
    """Shared feedforward network backbone."""

    def __init__(self, in_dim: int, hidden: list, out_dim: int,
                 activation=nn.Tanh, use_skip: bool = False):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), activation()]
            prev = h
        self.body = nn.Sequential(*layers)
        self.out = nn.Linear(prev, out_dim)
        self.skip = nn.Linear(in_dim, out_dim) if use_skip else None

    def forward(self, x):
        out = self.out(self.body(x))
        if self.skip is not None:
            out = out + self.skip(x)
        return out


def logit_safe(p, eps=EPS_CLAMP):
    p_c = torch.clamp(p, eps, 1 - eps)
    return torch.log(p_c / (1 - p_c))


def sigmoid_safe(x):
    return torch.sigmoid(x)


class SequentialLogitOutput(nn.Module):
    """Stick-breaking energy conservation output layer."""

    def forward(self, logits_a, logits_r, A_tmm, R_tmm):
        A = sigmoid_safe(logit_safe(A_tmm) + logits_a)
        cond = torch.clamp(R_tmm / (1 - A_tmm + 1e-8), EPS_CLAMP, 1 - EPS_CLAMP)
        R = sigmoid_safe(logit_safe(cond) + logits_r) * (1 - A)
        T = 1.0 - A - R
        return A, R, T


class MIMSurrogate(nn.Module):
    """
    Unified model supporting different structures and configurations.
    
    Config keys:
        geo_dim: int - geometric input dimension (1+n_params)
        use_tmm_backbone: bool - use TMM features as input
        conservation: str - "none", "penalty", "softmax", "logit"
        hidden: list - hidden layer sizes
        use_skip: bool - skip connection
        n_channels: int - 3 (A/R/T) or 6 (dual-pol)
    """

    def __init__(self, config: dict):
        super().__init__()
        self.cfg = config
        use_tmm = config.get("use_tmm_backbone", True)
        use_skip = config.get("use_skip", False)
        hidden = config.get("hidden", [128, 256, 256, 128])
        conservation = config.get("conservation", "logit")
        geo_dim = config["geo_dim"]
        n_channels = config.get("n_channels", 3)  # 3 or 6

        # Input dimension
        if use_tmm:
            in_dim = geo_dim + n_channels  # geo + TMM predictions
        else:
            in_dim = geo_dim  # geo only

        # Output dimension
        if conservation == "softmax":
            if n_channels == 6:
                out_dim = 6  # 3 for TE softmax + 3 for TM softmax
            else:
                out_dim = 3
        else:
            if n_channels == 6:
                out_dim = 4  # 2 for TE logit + 2 for TM logit
            else:
                out_dim = 2

        self.net = BaseResNet(in_dim, hidden, out_dim, use_skip=use_skip)
        self.conservation = conservation
        self.n_channels = n_channels

        if conservation == "logit":
            self.output_layer = SequentialLogitOutput()

    def forward(self, x_geo, A_tmm=None, R_tmm=None, T_tmm=None,
                A_tmm_tm=None, R_tmm_tm=None, T_tmm_tm=None):
        """
        x_geo: [B, geo_dim]
        For 3-channel: A_tmm, R_tmm, T_tmm: [B]
        For 6-channel: additionally A_tmm_tm, R_tmm_tm, T_tmm_tm: [B]
        """
        if self.cfg.get("use_tmm_backbone", True):
            if self.n_channels == 6:
                x = torch.cat([x_geo,
                               A_tmm.unsqueeze(1), R_tmm.unsqueeze(1), T_tmm.unsqueeze(1),
                               A_tmm_tm.unsqueeze(1), R_tmm_tm.unsqueeze(1), T_tmm_tm.unsqueeze(1)],
                              dim=1)
            else:
                x = torch.cat([x_geo,
                               A_tmm.unsqueeze(1), R_tmm.unsqueeze(1), T_tmm.unsqueeze(1)],
                              dim=1)
        else:
            x = x_geo

        raw = self.net(x)

        c = self.conservation
        if self.n_channels == 6:
            return self._output_6ch(raw, c, A_tmm, R_tmm, A_tmm_tm, R_tmm_tm)
        else:
            return self._output_3ch(raw, c, A_tmm, R_tmm)

    def _output_3ch(self, raw, c, A_tmm, R_tmm):
        if c == "none" or c == "penalty":
            A = torch.sigmoid(raw[:, 0])
            R = torch.sigmoid(raw[:, 1])
            T = 1 - A - R
        elif c == "softmax":
            probs = torch.softmax(raw, dim=-1)
            A, R, T = probs[:, 0], probs[:, 1], probs[:, 2]
        elif c == "logit":
            A, R, T = self.output_layer(raw[:, 0], raw[:, 1], A_tmm, R_tmm)
        return A, R, T

    def _output_6ch(self, raw, c, A_tmm_te, R_tmm_te, A_tmm_tm, R_tmm_tm):
        if c == "none" or c == "penalty":
            A_te = torch.sigmoid(raw[:, 0])
            R_te = torch.sigmoid(raw[:, 1])
            T_te = 1 - A_te - R_te
            A_tm = torch.sigmoid(raw[:, 2])
            R_tm = torch.sigmoid(raw[:, 3])
            T_tm = 1 - A_tm - R_tm
        elif c == "logit":
            A_te, R_te, T_te = self.output_layer(raw[:, 0], raw[:, 1],
                                                  A_tmm_te, R_tmm_te)
            A_tm, R_tm, T_tm = self.output_layer(raw[:, 2], raw[:, 3],
                                                  A_tmm_tm, R_tmm_tm)
        elif c == "softmax":
            probs_te = torch.softmax(raw[:, :3], dim=-1)
            probs_tm = torch.softmax(raw[:, 3:], dim=-1)
            A_te, R_te, T_te = probs_te[:, 0], probs_te[:, 1], probs_te[:, 2]
            A_tm, R_tm, T_tm = probs_tm[:, 0], probs_tm[:, 1], probs_tm[:, 2]
        return A_te, R_te, T_te, A_tm, R_tm, T_tm


# Model factory configs for Step 0 screening
MODEL_CONFIGS = {
    "M0": {
        "use_tmm_backbone": False,
        "conservation": "none",
        "hidden": [128, 256, 256, 128],
        "use_skip": False,
    },
    "M7": {
        "use_tmm_backbone": True,
        "conservation": "logit",
        "hidden": [128, 256, 256, 128],
        "use_skip": False,
    },
}


def create_model(model_name: str, geo_dim: int, n_channels: int = 3) -> MIMSurrogate:
    """Create model by name (M0 or M7) with appropriate dimensions."""
    cfg = dict(MODEL_CONFIGS[model_name])
    cfg["geo_dim"] = geo_dim
    cfg["n_channels"] = n_channels
    return MIMSurrogate(cfg)
