"""Fixed stage-1 network (spec 04 §4.1)."""
import torch
import torch.nn as nn


class GenoEncoder(nn.Module):
    def __init__(self, p: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(p, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256, 64))

    def forward(self, x):
        return self.net(x)


def env_mlp(q: int, out: int = 64) -> nn.Module:
    return nn.Sequential(nn.Linear(q, 64), nn.GELU(), nn.Linear(64, out))


class DartStage1(nn.Module):
    """two_head=True: ranking head s_ge = MLP([h_g, h_e, h_g ⊙ h_e]) plus an independent environment
    mean head μ̂_e = MLP(EC). two_head=False (`mse`): the same trunk outputs ŷ directly."""

    def __init__(self, p: int, q: int, two_head: bool = True, use_ec: bool = True, linear_g: bool = False):
        super().__init__()
        self.use_ec, self.linear_g = use_ec, linear_g
        self.lin = nn.Linear(p, 1) if linear_g else None
        self.g = GenoEncoder(p)
        self.e = env_mlp(q)
        self.head = nn.Sequential(nn.Linear(192, 64), nn.GELU(), nn.Linear(64, 1))
        self.two_head = two_head
        self.mu = env_mlp(q, 1) if two_head else None

    def forward(self, xg, xe):
        """xg (K, L, p), xe (K, q) -> s (K, L), mu (K,) or None."""
        if self.linear_g:  # ablation N2: s = wᵀx
            s = self.lin(xg).squeeze(-1)
        else:
            hg = self.g(xg)
            he = self.e(xe)[:, None, :].expand_as(hg)
            if not self.use_ec:  # ablation N1/N3: no environment input to the ranking head
                he = torch.zeros_like(hg)
            s = self.head(torch.cat([hg, he, hg * he], -1)).squeeze(-1)
        mu = self.mu(xe).squeeze(-1) if self.two_head else None
        return s, mu
