# learn_dynamic_causal_mask_ant.py
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import os

# -----------------------
# Config
# -----------------------

SEED = 42
LR = 1e-3
L1_LAMBDA = 1e-2
L2_LAMBDA = 1e-4
BATCH_SIZE = 512
EPOCHS_S = 400
EPOCHS_R = 200
PERCENTILE = 90
WINDOW_SIZE = 10
N_UPDATE_INTERVAL = 5000
MAX_STEPS = 30000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(SEED)
torch.manual_seed(SEED)

# -----------------------
# Linear Models
# -----------------------
class LinearMapMultiOutput(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W = nn.Parameter(torch.randn(out_dim, in_dim) * 0.01)
        self.b = nn.Parameter(torch.zeros(out_dim))
    def forward(self, x):
        return x @ self.W.t() + self.b

class LinearMapVector(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.w = nn.Parameter(torch.randn(in_dim) * 0.01)
        self.b = nn.Parameter(torch.zeros(1))
    def forward(self, x):
        return x @ self.w.unsqueeze(1) + self.b


def standardize(x):
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True) + 1e-8
    x_norm = (x - mean) / std
    return x_norm, mean, std

def train_sparse_linear(X, Y, epochs, L1_LAMBDA, L2_LAMBDA):

    model = LinearMapMultiOutput(X.shape[1], Y.shape[1]).to(device)

    opt = optim.Adam(model.parameters(), lr=LR)
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    Y_t = torch.tensor(Y, dtype=torch.float32, device=device)

    for ep in range(epochs):
        perm = np.random.permutation(X.shape[0])
        for i in range(0, X.shape[0], BATCH_SIZE):
            idx = perm[i:i+BATCH_SIZE]
            xb, yb = X_t[idx], Y_t[idx]
            pred = model(xb)
            mse = ((pred - yb)**2).mean()
            l1 = torch.norm(model.W, p=1)
            l2 = 0.5 * torch.sum(model.W**2)
            loss = mse + L1_LAMBDA * l1 + L2_LAMBDA * l2
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model

def extract_mask(W, percentile):
    absW = np.abs(W)
    thr = np.percentile(absW, 100.0 - percentile)
    eps = 1e-12

    mask = 1+np.clip(absW / (thr + eps), 0.0, 1.0).astype(np.float32)
    return mask/1.2, thr

def compute_and_save_masks(states, actions, rewards, next_states):
    """Compute causal masks based on sliding window data."""


    Z = np.concatenate([states, actions], axis=1)
    d_s = states.shape[1]
    d_a = actions.shape[1]
    d_in = d_s + d_a

    # normalize
    Z_norm, Z_mean, Z_std = standardize(Z)
    Y_s_norm, Y_s_mean, Y_s_std = standardize(next_states)
    Y_r_norm, Y_r_mean, Y_r_std = standardize(rewards.reshape(-1,1))


    model_s = train_sparse_linear(Z_norm, Y_s_norm, EPOCHS_S, L1_LAMBDA, L2_LAMBDA)
    with torch.no_grad():
        W_norm = model_s.W.detach().cpu().numpy()

    W_orig = (Y_s_std.T) * W_norm * (1.0 / Z_std)
    mask_W, thr_s = extract_mask(W_orig, PERCENTILE)
    M_s_to_s = mask_W[:, :d_s]
    M_a_to_s = mask_W[:, d_s:]

    # ---- Train reward ----
    model_r = LinearMapVector(d_in).to(device)
    opt_r = optim.Adam(model_r.parameters(), lr=LR)
    X_t = torch.tensor(Z_norm, dtype=torch.float32, device=device)
    Y_t = torch.tensor(Y_r_norm, dtype=torch.float32, device=device)
    for ep in range(EPOCHS_R):
        perm = np.random.permutation(X_t.shape[0])
        for i in range(0, X_t.shape[0], BATCH_SIZE):
            idx = perm[i:i+BATCH_SIZE]
            xb, yb = X_t[idx], Y_t[idx]
            pred = model_r(xb)
            mse = ((pred - yb)**2).mean()
            l1 = torch.norm(model_r.w, p=1)
            l2 = 0.5 * torch.sum(model_r.w**2)
            loss = mse + L1_LAMBDA*l1 + L2_LAMBDA*l2
            opt_r.zero_grad(); loss.backward(); opt_r.step()

    with torch.no_grad():
        w_norm = model_r.w.detach().cpu().numpy()
    w_orig_r = (Y_r_std.reshape(1,) * w_norm) * (1.0 / Z_std.reshape(-1,))
    mask_w_r, thr_r = extract_mask(w_orig_r, PERCENTILE)
    M_state_to_r = mask_w_r[:d_s]
    M_action_to_r = mask_w_r[d_s:]

    return M_s_to_s, M_a_to_s, M_state_to_r, M_action_to_r
