import math
from typing import Optional
import torch


DEVICE = None
K = None
betas = None
alphas = None
alphas_cumprod = None

action_low_t = None
action_high_t = None


def init_ddpm_globals(device, K_in, betas_in, alphas_in, alphas_cumprod_in,
                      action_low_in, action_high_in):

    global DEVICE, K, betas, alphas, alphas_cumprod, action_low_t, action_high_t
    DEVICE = device
    K = K_in
    betas = betas_in
    alphas = alphas_in
    alphas_cumprod = alphas_cumprod_in

    action_low_t = torch.tensor(action_low_in, dtype=torch.float32, device=DEVICE)
    action_high_t = torch.tensor(action_high_in, dtype=torch.float32, device=DEVICE)


def latent_to_env(a_lat):

    a_tanh = torch.tanh(a_lat)
    return action_low_t + (a_tanh + 1.0) * (action_high_t - action_low_t) / 2.0


def ddpm_reverse_sample(eps_model,
                        f_phi,
                        g_phi,
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
                        K_steps: int = None,
                        q_scale: float = 0.8,
                        guidance: bool = True):

    if K_steps is None:
        K_steps = K

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

                f_pred = f_phi(mask_diag * s, a0_tmp)
                loss_s = 0.5 * ((s_target - f_pred) ** 2).sum()
                grad_s = torch.autograd.grad(loss_s, a0_tmp, retain_graph=True)[0]
                grad_logp_s = -grad_s
                grad_logp_s = grad_logp_s / (grad_logp_s.norm(p=2, dim=1, keepdim=True) + 1e-3)

                causal_term = causal_term - torch.tensor(gamma_schedule[k], device=DEVICE) * grad_logp_s

            if r_target is not None and mask_a_to_r is not None:
                a0_tmp = a0_hat.detach().clone().requires_grad_(True)
                a0_tmp = latent_to_env(a0_tmp)

                s_star = f_phi(s, a0_tmp)
                g_pred = g_phi(mask_s_to_r * s_star, mask_a_to_r * a0_tmp)
                loss_r = 0.5 * ((r_target - g_pred) ** 2).sum()
                grad_r = torch.autograd.grad(loss_r, a0_tmp, retain_graph=True)[0]
                grad_logp_r = -grad_r
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
