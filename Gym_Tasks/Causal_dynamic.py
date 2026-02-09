
import numpy as np
import torch
import torch.nn as nn




class EpsNet(nn.Module):
    def __init__(self, a_dim, s_dim, time_emb_dim=64, hidden=[256,256]):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_emb_dim), nn.ReLU(),
            nn.Linear(time_emb_dim, time_emb_dim), nn.ReLU()
        )
        self.net = nn.Sequential(
            nn.Linear(a_dim + s_dim + time_emb_dim, hidden[0]), nn.ReLU(),
            nn.Linear(hidden[0], hidden[1]), nn.ReLU(),
            nn.Linear(hidden[1], a_dim)
        )

    def forward(self, a_k, s, k):
        k = k.float().unsqueeze(-1) if k.dim() == 1 else k.float()
        t_emb = self.time_mlp(k)
        x = torch.cat([a_k, s, t_emb], dim=-1)
        return self.net(x)


class TransitionModel(nn.Module):

    def __init__(self, s_dim, a_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(s_dim + a_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, s_dim)
        )
    def forward(self, s, a):
        if isinstance(s, np.ndarray):
            s = torch.from_numpy(s).float()
        if isinstance(a, np.ndarray):
            a = torch.from_numpy(a).float()
        return self.net(torch.cat([s, a], dim=-1))


class RewardModel(nn.Module):
    def __init__(self, s_dim, a_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(s_dim + a_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )
    def forward(self, s_next, a):
        return self.net(torch.cat([s_next, a], dim=-1)).squeeze(-1)
