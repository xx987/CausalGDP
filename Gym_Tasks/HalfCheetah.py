import math
import random
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import  gymnasium as gym
from Causal_mask_temporal_windows_updating import compute_and_save_masks as compute_mask


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = random.randint(0, 100)
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

env = gym.make("HalfCheetah-v4")
try:
    env.reset(seed=SEED)
except TypeError:
    pass

env.action_space.seed(SEED)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_low = env.action_space.low
action_high = env.action_space.high

K = 10
betas = torch.linspace(1e-4, 2e-2, K).to(DEVICE)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)


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


class QNetwork(nn.Module):
    def __init__(self, s_dim, a_dim, hidden=256):
        super().__init__()
        # MLP with LayerNorm after hidden layers
        self.fc1 = nn.Linear(s_dim + a_dim, hidden)
        self.ln1 = nn.LayerNorm(hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, 1)

    def forward(self, s, a):
        x = torch.cat([s, a], dim=-1)
        x = F.relu(self.ln1(self.fc1(x)))

        return self.out(x).squeeze(-1)


class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buf = []
        self.pos = 0
        self.capacity = capacity

    @staticmethod
    def to_2d_tensor(x):
        if isinstance(x, torch.Tensor):
            return x.detach().float()
        else:
            return torch.tensor(np.asarray(x, dtype=np.float32), dtype=torch.float32)

    def push(self, s, a, r, s_next, done):
        entry = (self.to_2d_tensor(s), self.to_2d_tensor(a), float(r),
                 self.to_2d_tensor(s_next), float(done))
        if len(self.buf) < self.capacity:
            self.buf.append(entry)
        else:
            self.buf[self.pos] = entry
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        idx = random.sample(range(len(self.buf)), batch_size)
        batch = [self.buf[i] for i in idx]
        s = torch.stack([b[0] for b in batch]).to(DEVICE)
        a = torch.stack([b[1] for b in batch]).to(DEVICE)
        r = torch.tensor([b[2] for b in batch], dtype=torch.float32, device=DEVICE)
        s_next = torch.stack([b[3] for b in batch]).to(DEVICE)
        done = torch.tensor([b[4] for b in batch], dtype=torch.float32, device=DEVICE)
        return s, a, r, s_next, done

    def __len__(self):
        return len(self.buf)



action_low_t = torch.tensor(action_low, dtype=torch.float32, device=DEVICE)
action_high_t = torch.tensor(action_high, dtype=torch.float32, device=DEVICE)

def latent_to_env(a_lat):
    a_tanh = torch.tanh(a_lat)
    return action_low_t + (a_tanh + 1.0) * (action_high_t - action_low_t) / 2.0




def ddpm_reverse_sample(eps_model: EpsNet,
                        f_phi: TransitionModel,
                        g_phi: RewardModel,
                        q_net1, q_net2,
                        s: torch.Tensor,
                        a_dim: int,
                        s_target: Optional[torch.Tensor] = None,
                        r_target: Optional[torch.Tensor] = None,
                        mask_a_to_snext: Optional[torch.Tensor] = None,
                        mask_s_to_snext: Optional[torch.Tensor] = None,
                        mask_s_to_r: Optional[torch.Tensor] = None,
                        mask_a_to_r: Optional[torch.Tensor] = None,

                        gamma_schedule=None,
                        beta_schedule=None,
                        eta_schedule=None,
                        K_steps: int = K,
                        q_scale: float = 0.8,
                        guidance: bool = True):
    B = s.shape[0]
    a_k = torch.randn(B, a_dim, device=DEVICE)

    if gamma_schedule is None:
        gamma_schedule = [0.5 * (1.0 - t / K_steps) for t in range(K_steps)]
    if beta_schedule is None:
        beta_schedule = [1.0 * (1.0 - t / K_steps) for t in range(K_steps)]
    if eta_schedule is None:
        eta_schedule = [0.2 * (1.0 - t / K_steps) for t in range(K_steps)]

    if s_target is not None:
        s_target = s_target.to(DEVICE)
    if r_target is not None:
        r_target = r_target.to(DEVICE).view(-1)

    for k in reversed(range(K_steps)):
        k_tensor = (torch.ones(B, device=DEVICE) * float(k)).to(DEVICE)
        if guidance:
            a_k.requires_grad_(True)
        else:
            a_k = a_k.detach()
            a_k.requires_grad_(False)

        eps_pred = eps_model(a_k, s, k_tensor)
        alpha_k = alphas[k].to(DEVICE)
        alpha_bar_k = alphas_cumprod[k].to(DEVICE)
        mu_theta = (1.0 / math.sqrt(alpha_k)) * (a_k - (betas[k] / math.sqrt(1.0 - alpha_bar_k)) * eps_pred)
        a0_hat = (a_k - torch.sqrt(1.0 - alpha_bar_k) * eps_pred) / torch.sqrt(alpha_bar_k)

        causal_term = torch.zeros_like(mu_theta)

        if guidance:
            if s_target is not None and mask_a_to_snext is not None:
                a0_tmp = a0_hat.detach().clone().requires_grad_(True)
                a0_tmp = latent_to_env(a0_tmp)

                mask_diag = torch.tensor(mask_s_to_snext.diagonal(), dtype=s.dtype, device=s.device)



                f_pred = f_phi(mask_diag*s, a0_tmp)
                loss_s = 0.5 * ((s_target - f_pred) ** 2).sum()
                grad_s = torch.autograd.grad(loss_s, a0_tmp, retain_graph=True)[0]
                grad_logp_s = - grad_s

                grad_logp_s = grad_logp_s / (grad_logp_s.norm(p=2, dim=1, keepdim=True) + 1e-3)

                causal_term = causal_term - torch.tensor(gamma_schedule[k], device=DEVICE) * grad_logp_s

            if r_target is not None and mask_a_to_r is not None:
                a0_tmp = a0_hat.detach().clone().requires_grad_(True)
                a0_tmp = latent_to_env(a0_tmp)


                s_star = f_phi(s, a0_tmp)
                g_pred = g_phi(mask_s_to_r*s_star, mask_a_to_r*a0_tmp)
                loss_r = 0.5 * ((r_target - g_pred) ** 2).sum()
                grad_r = torch.autograd.grad(loss_r, a0_tmp, retain_graph=True)[0]
                grad_logp_r = - grad_r
                grad_logp_r = grad_logp_r / (grad_logp_r.norm(p=2, dim=1, keepdim=True) + 1e-3)

                causal_term = causal_term - torch.tensor(beta_schedule[k], device=DEVICE) * grad_logp_r


            a0_tmp = a0_hat.detach().clone().requires_grad_(True)
            a0_tmp = latent_to_env(a0_tmp)


            q1_val = q_net1(s, a0_tmp)
            q2_val = q_net2(s, a0_tmp)


            q_val = torch.min(q1_val, q2_val)



            q_val = q_val.sum()


            grad_q = torch.autograd.grad(q_val, a0_tmp, retain_graph=False, create_graph=False)[0]

            gnorm = grad_q.norm(dim=1, keepdim=True).clamp(min=1e-6)
            grad_q = grad_q / gnorm * q_scale


            causal_term -= eta_schedule[k] * grad_q * 0.5




        mu_guided = mu_theta + causal_term
        if k > 0:
            noise = torch.randn_like(a_k)
            a_k = mu_guided + math.sqrt(betas[k]) * noise
        else:
            a_k = mu_guided
        a_k = a_k.detach()
    return a_k


states, actions, rewards, next_states = [], [], [], []

def train_diffusion_qlearning_online(env, num_episodes, batch_size, lr=3e-4, alpha_policy=0.5):

    M_a_to_snext = torch.ones(action_dim, dtype=torch.float32, device=DEVICE)
    M_s_to_snext = torch.ones(state_dim, state_dim, dtype=torch.float32, device=DEVICE)
    M_s_to_r = torch.ones(state_dim, dtype=torch.float32, device=DEVICE)
    M_a_to_r = torch.ones(action_dim, dtype=torch.float32, device=DEVICE)


    eps_model = EpsNet(action_dim, state_dim).to(DEVICE)
    f_phi = TransitionModel(state_dim, action_dim).to(DEVICE)
    g_phi = RewardModel(state_dim, action_dim).to(DEVICE)

    q_net1 = QNetwork(state_dim, action_dim).to(DEVICE)
    q_net2 = QNetwork(state_dim, action_dim).to(DEVICE)
    q_net_target1 = QNetwork(state_dim, action_dim).to(DEVICE)
    q_net_target2 = QNetwork(state_dim, action_dim).to(DEVICE)
    q_net_target1.load_state_dict(q_net1.state_dict())
    q_net_target2.load_state_dict(q_net2.state_dict())

    optim_q1 = optim.Adam(q_net1.parameters(), lr=1e-3)
    optim_q2 = optim.Adam(q_net2.parameters(), lr=1e-3)


    optim_eps = optim.Adam(eps_model.parameters(), lr=5e-3)
    optim_f = optim.Adam(f_phi.parameters(), lr=5e-3)
    optim_g = optim.Adam(g_phi.parameters(), lr=5e-3)




    replay = ReplayBuffer(capacity=50000)
    gamma = 0.99
    tau = 0.005
    episode_rewards = []

    tau_min, tau_max = 0.001, 0.01
    for ep in range(num_episodes):


        tau = min(tau_max, tau_min + 1e-4 * ep)



        reset_out = env.reset()
        s_raw = reset_out[0] if isinstance(reset_out, tuple) else reset_out
        done = False
        ep_reward = 0.0

        while not done:
            s_tensor = torch.tensor(np.asarray(s_raw, dtype=np.float32), device=DEVICE).unsqueeze(0)

            s_target = s_tensor

            best_rewards = sorted(episode_rewards)[-10:]
            if len(best_rewards) > 0:
                r_target_val = np.mean(best_rewards)
            else:
                r_target_val = None

            r_target = torch.tensor([3*r_target_val], device=DEVICE) if r_target_val is not None else None


            a_lat = ddpm_reverse_sample(
                eps_model, f_phi, g_phi, q_net1, q_net2,
                s_tensor, action_dim,
                s_target=s_target,
                r_target=r_target,
                mask_a_to_snext=M_a_to_snext,
                mask_s_to_snext=M_s_to_snext,
                mask_s_to_r=M_s_to_r,
                mask_a_to_r = M_a_to_r,
                K_steps=K,
                guidance=True
            )

            a_tanh = torch.tanh(a_lat).squeeze(0).detach().cpu().numpy()
            a_env = action_low + (a_tanh + 1.0) * (action_high - action_low) / 2.0
            a_env = np.asarray(a_env, dtype=np.float32)


            step_out = env.step(a_env)
            if len(step_out) == 5:
                s_next_raw, r, terminated, truncated, info = step_out
                done = terminated or truncated
            else:
                s_next_raw, r, done, info = step_out

            states.append(s_tensor.squeeze(0))
            actions.append(a_env)
            rewards.append(r)
            next_states.append(s_next_raw)
            if len(actions)%200 == 0 and len(actions) > 200:

                M_s_to_snext, M_a_to_snext, M_a_s_to_r, M_s_s_to_r = compute_mask(
                np.array(states[-200:], dtype=np.float32),
                np.array(actions[-200:], dtype=np.float32),
                np.array(rewards[-200:], dtype=np.float32),
                np.array(next_states[-200:], dtype=np.float32))

            replay.push(s_raw, a_env, r, s_next_raw, done)
            ep_reward += float(r)

            s_raw = s_next_raw

            if len(replay) >= batch_size:
                s_b, a_b, r_b, s_next_b, done_b = replay.sample(batch_size)

                optim_f.zero_grad()
                pred_s_next = f_phi(s_b, a_b)
                loss_f = F.mse_loss(pred_s_next, s_next_b)
                loss_f.backward(); optim_f.step()

                optim_g.zero_grad()
                pred_r = g_phi(s_next_b, a_b)
                loss_g = F.mse_loss(pred_r, r_b)
                loss_g.backward(); optim_g.step()

                optim_q1.zero_grad()
                optim_q2.zero_grad()

                with torch.no_grad():

                    a_next = ddpm_reverse_sample(eps_model, f_phi, g_phi, q_net_target1, q_net_target2,
                                                 s_next_b, action_dim, K_steps=K, guidance=False)
                    a_next_tanh = torch.tanh(a_next)
                    action_low_t = torch.tensor(action_low, dtype=torch.float32, device=DEVICE)
                    action_high_t = torch.tensor(action_high, dtype=torch.float32, device=DEVICE)
                    a_next_env = action_low_t + (a_next_tanh + 1.0) * (action_high_t - action_low_t) / 2.0


                    q1_next = q_net_target1(s_next_b, a_next_env)
                    q2_next = q_net_target2(s_next_b, a_next_env)
                    q_next = torch.min(q1_next, q2_next)

                    td_target = r_b + gamma * (1 - done_b) * q_next
                    td_target = td_target.detach()


                q1_pred = q_net1(s_b, a_b)
                q2_pred = q_net2(s_b, a_b)

                loss_q1 = F.mse_loss(q1_pred, td_target)
                loss_q2 = F.mse_loss(q2_pred, td_target)

                loss_q1.backward()
                torch.nn.utils.clip_grad_norm_(q_net1.parameters(), 1.0)
                optim_q1.step()

                loss_q2.backward()
                torch.nn.utils.clip_grad_norm_(q_net2.parameters(), 1.0)
                optim_q2.step()


                for param, param_t in zip(q_net1.parameters(), q_net_target1.parameters()):
                    param_t.data.copy_(tau * param.data + (1 - tau) * param_t.data)
                for param, param_t in zip(q_net2.parameters(), q_net_target2.parameters()):
                    param_t.data.copy_(tau * param.data + (1 - tau) * param_t.data)

                optim_eps.zero_grad()
                B = s_b.shape[0]
                k_rand = torch.randint(0, K, (B,), device=DEVICE)
                eps_true = torch.randn(B, action_dim, device=DEVICE)
                alpha_bars = alphas_cumprod[k_rand].unsqueeze(-1).to(DEVICE)
                a0_clean = a_b
                a_k_noisy = torch.sqrt(alpha_bars) * a0_clean + torch.sqrt(1 - alpha_bars) * eps_true
                eps_pred_batch = eps_model(a_k_noisy, s_b, k_rand.float())
                loss_eps = F.mse_loss(eps_pred_batch, eps_true)

                a0_hat_batch = (a_k_noisy - torch.sqrt(1 - alpha_bars) * eps_pred_batch) / torch.sqrt(alpha_bars)
                a0_hat_tanh = torch.tanh(a0_hat_batch)
                action_low_t = torch.tensor(action_low, dtype=torch.float32, device=DEVICE)
                action_high_t = torch.tensor(action_high, dtype=torch.float32, device=DEVICE)
                a0_hat_env = action_low_t + (a0_hat_tanh + 1.0) * (action_high_t - action_low_t) / 2.0

                q1_policy = q_net1(s_b, a0_hat_env)
                q2_policy = q_net2(s_b, a0_hat_env)
                q_policy = torch.min(q1_policy, q2_policy)

                # policy loss
                loss_policy_q = - q_policy.sum()


                loss_total_eps = loss_eps + alpha_policy * loss_policy_q
                loss_total_eps.backward()
                torch.nn.utils.clip_grad_norm_(eps_model.parameters(), 1.0)
                optim_eps.step()

        episode_rewards.append(ep_reward)
        print(f"Episode {ep+1}/{num_episodes}, Reward: {ep_reward:.2f}")

    print("Training finished!")
    return episode_rewards


if __name__ == "__main__":
    print("Training Start")
    rewards = train_diffusion_qlearning_online(env, num_episodes=2000, batch_size=256, lr=3e-4, alpha_policy=5)
    print("Collected episode rewards:", rewards)
