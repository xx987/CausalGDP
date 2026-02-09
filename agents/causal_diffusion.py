# Reference:@article{wang2022diffusion,
#   title={Diffusion Policies as an Expressive Policy Class for Offline Reinforcement Learning},
#   author={Wang, Zhendong and Hunt, Jonathan J and Zhou, Mingyuan},
#   journal={arXiv preprint arXiv:2208.06193},
#   year={2022}
# }" The github: https://github.com/Zhendong-Wang/Diffusion-Policies-for-Offline-RL


import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils.logger import logger

from agents.diffusion import Diffusion
from agents.model import MLP
from agents.helpers import EMA

def f_masked(f_phi, s, a, mask_s_to_snext=None, mask_a_to_snext=None):

    s_in = s
    if mask_s_to_snext is not None:
        s_in = s * mask_s_to_snext.diagonal().view(1, -1)  # (B, sd)


    if mask_a_to_snext is None:
        a_feat = a  # (B, ad) 退化情况
        x = torch.cat([s_in, a_feat], dim=-1)
        return f_phi(x)

    if mask_a_to_snext.dim() == 1:

        a_in = a * mask_a_to_snext.view(1, -1)  # (B, ad)
        x = torch.cat([s_in, a_in], dim=-1)
        return f_phi(x)


    a_exp = a.unsqueeze(1) * mask_a_to_snext.unsqueeze(0)  # (B, sd, ad)
    a_feat = a_exp.reshape(a.shape[0], -1)                 # (B, sd*ad)

    x = torch.cat([s_in, a_feat], dim=-1)                  # (B, sd + sd*ad)
    return f_phi(x)



def g_masked(g_phi, s_next, a, mask_s_to_r=None, mask_a_to_r=None):
    s_in = s_next
    a_in = a
    if mask_s_to_r is not None:
        s_in = s_next * mask_s_to_r.view(1, -1)
    if mask_a_to_r is not None:
        a_in = a * mask_a_to_r.view(1, -1)

    x = torch.cat([s_in, a_in], dim=-1)
    return g_phi(x)





class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.q1_model = nn.Sequential(nn.Linear(state_dim + action_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, 1))

        self.q2_model = nn.Sequential(nn.Linear(state_dim + action_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, 1))

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q1_model(x), self.q2_model(x)

    def q1(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q1_model(x)

    def q_min(self, state, action):
        q1, q2 = self.forward(state, action)
        return torch.min(q1, q2)


class Diffusion_Causal(object):
    def __init__(self,
                 state_dim,
                 action_dim,
                 max_action,
                 device,
                 discount,
                 tau,
                 max_q_backup=False,
                 eta=1.0,
                 beta_schedule='linear',
                 n_timesteps=100,
                 ema_decay=0.995,
                 step_start_ema=1000,
                 update_ema_every=5,
                 lr=3e-4,
                 lr_decay=False,
                 lr_maxt=1000,
                 grad_norm=1.0,
                 ):

        self.model = MLP(state_dim=state_dim, action_dim=action_dim, device=device)


        self.actor = Diffusion(state_dim=state_dim, action_dim=action_dim, model=self.model, max_action=max_action,
                               beta_schedule=beta_schedule, n_timesteps=n_timesteps,).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.lr_decay = lr_decay
        self.grad_norm = grad_norm

        self.step = 0
        self.step_start_ema = step_start_ema
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.actor)
        self.update_ema_every = update_ema_every

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        if lr_decay:
            self.actor_lr_scheduler = CosineAnnealingLR(self.actor_optimizer, T_max=lr_maxt, eta_min=0.)
            self.critic_lr_scheduler = CosineAnnealingLR(self.critic_optimizer, T_max=lr_maxt, eta_min=0.)

        self.state_dim = state_dim
        self.max_action = max_action
        self.action_dim = action_dim
        self.discount = discount
        self.tau = tau
        self.eta = eta  # q_learning weight
        self.device = device
        self.max_q_backup = max_q_backup

        self.mask_a_to_r = torch.ones(action_dim, device=self.device)
        self.mask_s_to_r = torch.ones(state_dim, device=self.device)


        "#############mask updating the causal dynamical model ######"
        self.mask_s_to_snext = torch.ones(state_dim, state_dim, device=self.device)
        self.mask_a_to_snext = torch.ones(state_dim, action_dim, device=self.device)  # (sd, ad)




        hidden = 256


        f_in_dim = state_dim + state_dim * action_dim
        self.f_phi = nn.Sequential(
            nn.Linear(f_in_dim, hidden),
            nn.Mish(),
            nn.Linear(hidden, hidden),
            nn.Mish(),
            nn.Linear(hidden, state_dim),
        ).to(self.device)


        self.g_omega = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden),
            nn.Mish(),
            nn.Linear(hidden, hidden),
            nn.Mish(),
            nn.Linear(hidden, 1),
        ).to(self.device)

        self.dyn_lr = lr
        self.f_optimizer = torch.optim.Adam(self.f_phi.parameters(), lr=self.dyn_lr)
        self.g_optimizer = torch.optim.Adam(self.g_omega.parameters(), lr=self.dyn_lr)
        "#############mask updating the causal dynamical model ######"






    def step_ema(self):
        if self.step < self.step_start_ema:
            return
        self.ema.update_model_average(self.ema_model, self.actor)

    def train(self, replay_buffer, iterations, batch_size=100, log_writer=None):

        metric = {'bc_loss': [], 'ql_loss': [], 'actor_loss': [], 'critic_loss': []}
        for _ in range(iterations):
            # Sample replay buffer / batch
            state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

            # =========================================================
            # (C) Update causal dynamics models
            # =========================================================

            # 1) predict next_state with masked inputs
            pred_next_state = f_masked(
                self.f_phi,
                state,
                action,
                mask_s_to_snext=self.mask_s_to_snext,
                mask_a_to_snext=self.mask_a_to_snext,
            )
            loss_f = F.mse_loss(pred_next_state, next_state)

            self.f_optimizer.zero_grad()
            loss_f.backward()
            if self.grad_norm > 0:
                nn.utils.clip_grad_norm_(self.f_phi.parameters(), max_norm=self.grad_norm, norm_type=2)
            self.f_optimizer.step()

            # 2) predict reward with masked inputs
            pred_reward = g_masked(
                self.g_omega,
                next_state,
                action,
                mask_s_to_r=self.mask_s_to_r,
                mask_a_to_r=self.mask_a_to_r,
            )

            # reward shape sometimes (B,1) already; ensure match
            if reward.dim() == 1:
                reward_target = reward.view(-1, 1)
            else:
                reward_target = reward

            loss_g = F.mse_loss(pred_reward, reward_target)

            self.g_optimizer.zero_grad()
            loss_g.backward()
            if self.grad_norm > 0:
                nn.utils.clip_grad_norm_(self.g_omega.parameters(), max_norm=self.grad_norm, norm_type=2)
            self.g_optimizer.step()

            """ Q Training """

            new_action = action * self.mask_a_to_r.view(1, -1)
            new_state = state * self.mask_s_to_r.view(1, -1)

            current_q1, current_q2 = self.critic(new_state, new_action)


            if self.max_q_backup:


                next_state_rpt = torch.repeat_interleave(next_state, repeats=10, dim=0)
                next_action_rpt = self.ema_model(next_state_rpt)

                # ===== 新增：mask target inputs =====
                next_action_rpt = next_action_rpt * self.mask_a_to_r.view(1, -1)
                next_state_rpt_m = next_state_rpt * self.mask_s_to_r.view(1, -1)
                # ===================================

                target_q1, target_q2 = self.critic_target(next_state_rpt_m, next_action_rpt)
                target_q1 = target_q1.view(batch_size, 10).max(dim=1, keepdim=True)[0]
                target_q2 = target_q2.view(batch_size, 10).max(dim=1, keepdim=True)[0]
                target_q = torch.min(target_q1, target_q2)
            else:
                next_action = self.ema_model(next_state)


                next_action = next_action * self.mask_a_to_r.view(1, -1)
                next_state_m = next_state * self.mask_s_to_r.view(1, -1)
                # ===================================

                target_q1, target_q2 = self.critic_target(next_state_m, next_action)
                target_q = torch.min(target_q1, target_q2)

            target_q = (reward + not_done * self.discount * target_q).detach()

            critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            if self.grad_norm > 0:
                critic_grad_norms = nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.grad_norm, norm_type=2)
            self.critic_optimizer.step()

            """ Causal Policy Training """
            bc_loss = self.actor.loss(action, state)
            new_action = self.actor(state)

            masked_new_action = new_action * self.mask_a_to_r.view(1, -1)
            masked_new_state = state * self.mask_s_to_r.view(1, -1)


            q1_new_action, q2_new_action = self.critic(masked_new_state, masked_new_action)
            if np.random.uniform() > 0.5:
                q_loss = - q1_new_action.mean() / q2_new_action.abs().mean().detach()
            else:
                q_loss = - q2_new_action.mean() / q1_new_action.abs().mean().detach()
            actor_loss = bc_loss + self.eta * q_loss

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            if self.grad_norm > 0: 
                actor_grad_norms = nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.grad_norm, norm_type=2)
            self.actor_optimizer.step()


            """ Step Target network """
            if self.step % self.update_ema_every == 0:
                self.step_ema()

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            self.step += 1

            """ Log """
            if log_writer is not None:
                if self.grad_norm > 0:
                    log_writer.add_scalar('Actor Grad Norm', actor_grad_norms.max().item(), self.step)
                    log_writer.add_scalar('Critic Grad Norm', critic_grad_norms.max().item(), self.step)
                log_writer.add_scalar('BC Loss', bc_loss.item(), self.step)
                log_writer.add_scalar('QL Loss', q_loss.item(), self.step)
                log_writer.add_scalar('Critic Loss', critic_loss.item(), self.step)
                log_writer.add_scalar('Target_Q Mean', target_q.mean().item(), self.step)

            metric['actor_loss'].append(actor_loss.item())
            metric['bc_loss'].append(bc_loss.item())
            metric['ql_loss'].append(q_loss.item())
            metric['critic_loss'].append(critic_loss.item())

        if self.lr_decay: 
            self.actor_lr_scheduler.step()
            self.critic_lr_scheduler.step()

        return metric

    def sample_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)

        state_rpt = torch.repeat_interleave(state, repeats=50, dim=0)

        # mask_s_to_r: (state_dim,)
        s_mask = self.mask_s_to_r.view(1, -1)  # (1, state_dim)
        state_rpt_masked = state_rpt * s_mask  # (50, state_dim)



        with torch.no_grad():
            action = self.actor.sample(state_rpt)  # (50, action_dim) 也可以用 state_rpt（看你想mask作用在哪）
            a_mask = self.mask_a_to_r.view(1, -1)  # (1, action_dim)
            action_masked = action * a_mask
            q_value = self.critic_target.q_min(state_rpt_masked, action_masked).flatten()
            idx = torch.multinomial(F.softmax(q_value), 1)

        return action[idx].cpu().data.numpy().flatten()

    def save_model(self, dir, id=None):
        if id is not None:
            torch.save(self.actor.state_dict(), f'{dir}/actor_{id}.pth')
            torch.save(self.critic.state_dict(), f'{dir}/critic_{id}.pth')
        else:
            torch.save(self.actor.state_dict(), f'{dir}/actor.pth')
            torch.save(self.critic.state_dict(), f'{dir}/critic.pth')

    def load_model(self, dir, id=None):
        if id is not None:
            self.actor.load_state_dict(torch.load(f'{dir}/actor_{id}.pth'))
            self.critic.load_state_dict(torch.load(f'{dir}/critic_{id}.pth'))
        else:
            self.actor.load_state_dict(torch.load(f'{dir}/actor.pth'))
            self.critic.load_state_dict(torch.load(f'{dir}/critic.pth'))


