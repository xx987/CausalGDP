# Reference:@article{wang2022diffusion,
#   title={Diffusion Policies as an Expressive Policy Class for Offline Reinforcement Learning},
#   author={Wang, Zhendong and Hunt, Jonathan J and Zhou, Mingyuan},
#   journal={arXiv preprint arXiv:2208.06193},
#   year={2022}
# }" The github: https://github.com/Zhendong-Wang/Diffusion-Policies-for-Offline-RL


import argparse
import gym
import numpy as np
import os
import torch
import json

import d4rl
from utils import utils
from utils.data_sampler import Data_Sampler
from utils.logger import logger, setup_logger
from torch.utils.tensorboard import SummaryWriter
from Causal_mask_temporal_windows_updating import compute_and_save_masks as compute_mask


hyperparameters = {

    'halfcheetah-medium-v2':         {'lr': 3e-4, 'eta': 1.0,   'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 50, 'num_epochs': 2000, 'gn': 9.0,  'top_k': 1},
    'hopper-medium-v2':              {'lr': 3e-4, 'eta': 1.0,   'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 50, 'num_epochs': 2000, 'gn': 9.0,  'top_k': 2},
    'walker2d-medium-v2':            {'lr': 3e-4, 'eta': 1.0,   'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 50, 'num_epochs': 2000, 'gn': 1.0,  'top_k': 1},
    'halfcheetah-medium-replay-v2':  {'lr': 3e-4, 'eta': 1.0,   'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 50, 'num_epochs': 2000, 'gn': 2.0,  'top_k': 0},
    'hopper-medium-replay-v2':       {'lr': 3e-4, 'eta': 1.0,   'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 50, 'num_epochs': 2000, 'gn': 4.0,  'top_k': 2},
    'walker2d-medium-replay-v2':     {'lr': 3e-4, 'eta': 1.0,   'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 50, 'num_epochs': 2000, 'gn': 4.0,  'top_k': 1},
    'halfcheetah-medium-expert-v2':  {'lr': 3e-4, 'eta': 1.0,   'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 50, 'num_epochs': 2000, 'gn': 7.0,  'top_k': 0},
    'hopper-medium-expert-v2':       {'lr': 3e-4, 'eta': 1.0,   'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 50, 'num_epochs': 2000, 'gn': 5.0,  'top_k': 2},
    'walker2d-medium-expert-v2':     {'lr': 3e-4, 'eta': 1.0,   'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 50, 'num_epochs': 2000, 'gn': 5.0,  'top_k': 1},
    'antmaze-umaze-v0':              {'lr': 3e-4, 'eta': 0.5,   'max_q_backup': False,  'reward_tune': 'cql_antmaze', 'eval_freq': 50, 'num_epochs': 1000, 'gn': 2.0,  'top_k': 2},
    'antmaze-umaze-diverse-v0':      {'lr': 3e-4, 'eta': 2.0,   'max_q_backup': True,   'reward_tune': 'cql_antmaze', 'eval_freq': 50, 'num_epochs': 1000, 'gn': 3.0,  'top_k': 2},
    'antmaze-medium-play-v0':        {'lr': 1e-3, 'eta': 2.0,   'max_q_backup': True,   'reward_tune': 'cql_antmaze', 'eval_freq': 50, 'num_epochs': 1000, 'gn': 2.0,  'top_k': 1},
    'antmaze-medium-diverse-v0':     {'lr': 3e-4, 'eta': 3.0,   'max_q_backup': True,   'reward_tune': 'cql_antmaze', 'eval_freq': 50, 'num_epochs': 1000, 'gn': 1.0,  'top_k': 1},
    'antmaze-large-play-v0':         {'lr': 3e-4, 'eta': 4.5,   'max_q_backup': True,   'reward_tune': 'cql_antmaze', 'eval_freq': 50, 'num_epochs': 1000, 'gn': 10.0, 'top_k': 2},
    'antmaze-large-diverse-v0':      {'lr': 3e-4, 'eta': 3.5,   'max_q_backup': True,   'reward_tune': 'cql_antmaze', 'eval_freq': 50, 'num_epochs': 1000, 'gn': 7.0,  'top_k': 1},
    'pen-human-v1':                  {'lr': 3e-5, 'eta': 0.15,  'max_q_backup': False,  'reward_tune': 'normalize',   'eval_freq': 50, 'num_epochs': 1000, 'gn': 7.0,  'top_k': 2},
    'pen-cloned-v1':                 {'lr': 3e-5, 'eta': 0.1,   'max_q_backup': False,  'reward_tune': 'normalize',   'eval_freq': 50, 'num_epochs': 1000, 'gn': 8.0,  'top_k': 2},

    'door-human-v0':                 {'lr': 3e-3, 'eta': 3, 'max_q_backup': True, 'reward_tune': 'normalize', 'eval_freq': 20,'num_epochs': 1000, 'gn': 7.0, 'top_k': 4},
    'hammer-human-v0':               {'lr': 3e-4, 'eta': 0.4, 'max_q_backup': True, 'reward_tune': 'normalize', 'eval_freq': 20,'num_epochs': 1000, 'gn': 8.0, 'top_k': 2},


    'maze2d-umaze-v1':               {'lr': 3e-4, 'eta': 5, 'max_q_backup': True, 'reward_tune': 'no', 'eval_freq': 20, 'num_epochs': 1000, 'gn': 7.0, 'top_k': 2},

    'maze2d-medium-v1':             {'lr': 3e-4, 'eta': 5, 'max_q_backup': True, 'reward_tune': 'no', 'eval_freq': 20, 'num_epochs': 1000, 'gn': 7.0, 'top_k': 2},

    'maze2d-large-v1':              {'lr': 3e-4, 'eta': 5, 'max_q_backup': True, 'reward_tune': 'no', 'eval_freq': 20, 'num_epochs': 1000, 'gn': 7.0, 'top_k': 2},


    'maze2d-umaze-dense-v1':        {'lr': 3e-4, 'eta': 5, 'max_q_backup': True, 'reward_tune': 'no', 'eval_freq': 20, 'num_epochs': 1000, 'gn': 7.0, 'top_k': 2},

    'maze2d-medium-dense-v1':       {'lr': 3e-4, 'eta': 5, 'max_q_backup': True, 'reward_tune': 'no', 'eval_freq': 20, 'num_epochs': 1000, 'gn': 7.0, 'top_k': 2},

    'maze2d-large-dense-v1':         {'lr': 3e-4, 'eta': 5, 'max_q_backup': True, 'reward_tune': 'no', 'eval_freq': 20, 'num_epochs': 1000, 'gn': 7.0, 'top_k': 2},


    'kitchen-complete-v0':           {'lr': 3e-4, 'eta': 0.005, 'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 50, 'num_epochs': 250 , 'gn': 9.0,  'top_k': 2},
    'kitchen-partial-v0':            {'lr': 3e-4, 'eta': 0.005, 'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 50, 'num_epochs': 1000, 'gn': 10.0, 'top_k': 2},
    'kitchen-mixed-v0':              {'lr': 3e-4, 'eta': 0.005, 'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 50, 'num_epochs': 1000, 'gn': 10.0, 'top_k': 0},
}




def train_agent(env, state_dim, action_dim, max_action, device, output_dir, args):
    # Load buffer
    dataset = d4rl.qlearning_dataset(env)
    data_sampler = Data_Sampler(dataset, device, args.reward_tune)
    utils.print_banner('Loaded buffer')

    '####################The causal information is added here#############################'
    states, actions, rewards, next_states = [], [], [], []

    def feed_mask_buffer_from_sampler(data_sampler, n_transitions: int = 200):
        """
        从离线 Data_Sampler 里抽 n_transitions 条 (s, a, r, s')
        用于 compute_mask
        """
        s, a, s2, r, not_done = data_sampler.sample(n_transitions)

        s_np = s.detach().cpu().numpy().astype(np.float32)
        a_np = a.detach().cpu().numpy().astype(np.float32)
        r_np = r.detach().cpu().numpy().astype(np.float32).squeeze(-1)
        s2_np = s2.detach().cpu().numpy().astype(np.float32)

        for i in range(n_transitions):
            states.append(s_np[i])
            actions.append(a_np[i])
            rewards.append(r_np[i])
            next_states.append(s2_np[i])


    if args.algo == 'ql':
        from agents.causal_diffusion import Diffusion_Causal as Agent
        agent = Agent(state_dim=state_dim,
                      action_dim=action_dim,
                      max_action=max_action,
                      device=device,
                      discount=args.discount,
                      tau=args.tau,
                      max_q_backup=args.max_q_backup,
                      beta_schedule=args.beta_schedule,
                      n_timesteps=args.T,
                      eta=args.eta,
                      lr=args.lr,
                      lr_decay=args.lr_decay,
                      lr_maxt=args.num_epochs,
                      grad_norm=args.gn)
    elif args.algo == 'bc':
        from agents.bc_diffusion import Diffusion_BC as Agent
        agent = Agent(state_dim=state_dim,
                      action_dim=action_dim,
                      max_action=max_action,
                      device=device,
                      discount=args.discount,
                      tau=args.tau,
                      beta_schedule=args.beta_schedule,
                      n_timesteps=args.T,
                      lr=args.lr)

    early_stop = False
    stop_check = utils.EarlyStopping(tolerance=1, min_delta=0.)
    writer = None  # SummaryWriter(output_dir)

    M_a_to_r = np.ones(action_dim, dtype=np.float32)



    evaluations = []
    training_iters = 0
    max_timesteps = args.num_epochs * args.num_steps_per_epoch
    metric = 100.
    utils.print_banner(f"Training Start", separator="*", num_star=90)

    M_s_to_snext = np.ones((state_dim, state_dim), dtype=np.float32)
    M_a_to_snext = np.ones((state_dim, action_dim), dtype=np.float32)
    M_s_to_r = np.ones((state_dim,), dtype=np.float32)
    M_a_to_r = np.ones((action_dim,), dtype=np.float32)

    while (training_iters < max_timesteps) and (not early_stop):
        iterations = int(args.eval_freq * args.num_steps_per_epoch)

        feed_mask_buffer_from_sampler(data_sampler, n_transitions=50)



        loss_metric = agent.train(data_sampler,
                                  iterations=iterations,
                                  batch_size=args.batch_size,
                                  log_writer=writer)

        if len(actions) >= 200 and len(actions) % 200 == 0:
            M_s_to_snext, M_a_to_snext, M_s_to_r, M_a_to_r = compute_mask(
                np.array(states[-200:], dtype=np.float32),
                np.array(actions[-200:], dtype=np.float32),
                np.array(rewards[-200:], dtype=np.float32),
                np.array(next_states[-200:], dtype=np.float32),
            )



        m_s_s = torch.tensor(M_s_to_snext, dtype=torch.float32, device=agent.device)  # (sd, sd)
        m_a_s = torch.tensor(M_a_to_snext, dtype=torch.float32, device=agent.device)  # 期望 (sd, ad)
        m_s_r = torch.tensor(M_s_to_r, dtype=torch.float32, device=agent.device).view(-1)  # (sd,)
        m_a_r = torch.tensor(M_a_to_r, dtype=torch.float32, device=agent.device).view(-1)  # (ad,)

        # sanity checks
        if m_a_r.numel() != agent.action_dim:
            raise RuntimeError(f"M_a_to_r dim mismatch: got {m_a_r.numel()} expected {agent.action_dim}")
        if m_s_r.numel() != agent.state_dim:
            raise RuntimeError(f"M_s_to_r dim mismatch: got {m_s_r.numel()} expected {agent.state_dim}")
        if m_s_s.shape != (agent.state_dim, agent.state_dim):
            raise RuntimeError(
                f"M_s_to_snext shape mismatch: got {m_s_s.shape} expected {(agent.state_dim, agent.state_dim)}")

        if m_a_s.dim() != 2 or m_a_s.shape != (agent.state_dim, agent.action_dim):
            raise RuntimeError(
                f"M_a_to_snext shape mismatch: got {tuple(m_a_s.shape)} expected {(agent.state_dim, agent.action_dim)}")

        agent.mask_a_to_r.copy_(m_a_r)
        agent.mask_s_to_r.copy_(m_s_r)
        agent.mask_s_to_snext.copy_(m_s_s)
        agent.mask_a_to_snext.copy_(m_a_s)

        training_iters += iterations
        curr_epoch = int(training_iters // int(args.num_steps_per_epoch))




        # Logging
        utils.print_banner(f"Train step: {training_iters}", separator="*", num_star=90)
        logger.record_tabular('Trained Epochs', curr_epoch)
        logger.record_tabular('BC Loss', np.mean(loss_metric['bc_loss']))
        logger.record_tabular('QL Loss', np.mean(loss_metric['ql_loss']))
        logger.record_tabular('Actor Loss', np.mean(loss_metric['actor_loss']))
        logger.record_tabular('Critic Loss', np.mean(loss_metric['critic_loss']))
        logger.dump_tabular()

        # Evaluation
        eval_res, eval_res_std, eval_norm_res, eval_norm_res_std = eval_policy(agent, args.env_name, args.seed,
                                                                               eval_episodes=args.eval_episodes)
        evaluations.append([eval_res, eval_res_std, eval_norm_res, eval_norm_res_std,
                            np.mean(loss_metric['bc_loss']), np.mean(loss_metric['ql_loss']),
                            np.mean(loss_metric['actor_loss']), np.mean(loss_metric['critic_loss']),
                            curr_epoch])
        np.save(os.path.join(output_dir, "eval"), evaluations)
        logger.record_tabular('Average Episodic Reward', eval_res)
        logger.record_tabular('Average Episodic N-Reward', eval_norm_res)
        logger.dump_tabular()

        bc_loss = np.mean(loss_metric['bc_loss'])
        if args.early_stop:
            early_stop = stop_check(metric, bc_loss)

        metric = bc_loss

        if args.save_best_model:
            agent.save_model(output_dir, curr_epoch)

    # Model Selection: online or offline
    scores = np.array(evaluations)
    if args.ms == 'online':
        best_id = np.argmax(scores[:, 2])
        best_res = {'model selection': args.ms, 'epoch': scores[best_id, -1],
                    'best normalized score avg': scores[best_id, 2],
                    'best normalized score std': scores[best_id, 3],
                    'best raw score avg': scores[best_id, 0],
                    'best raw score std': scores[best_id, 1]}
        with open(os.path.join(output_dir, f"best_score_{args.ms}.txt"), 'w') as f:
            f.write(json.dumps(best_res))
    elif args.ms == 'offline':
        bc_loss = scores[:, 4]
        top_k = min(len(bc_loss) - 1, args.top_k)
        where_k = np.argsort(bc_loss) == top_k
        best_res = {'model selection': args.ms, 'epoch': scores[where_k][0][-1],
                    'best normalized score avg': scores[where_k][0][2],
                    'best normalized score std': scores[where_k][0][3],
                    'best raw score avg': scores[where_k][0][0],
                    'best raw score std': scores[where_k][0][1]}

        with open(os.path.join(output_dir, f"best_score_{args.ms}.txt"), 'w') as f:
            f.write(json.dumps(best_res))

    # writer.close()


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, eval_episodes=10):
    eval_env = gym.make(env_name)#, disable_env_checker=True
    eval_env.seed(seed + 100)

    scores = []
    for _ in range(eval_episodes):
        traj_return = 0.
        state, done = eval_env.reset(), False
        while not done:
            action = policy.sample_action(np.array(state))
            state, reward, done, _ = eval_env.step(action)
            traj_return += reward
        scores.append(traj_return)

    avg_reward = np.mean(scores)
    std_reward = np.std(scores)

    normalized_scores = [eval_env.get_normalized_score(s) for s in scores]
    avg_norm_score = eval_env.get_normalized_score(avg_reward)
    std_norm_score = np.std(normalized_scores)

    utils.print_banner(f"Evaluation over {eval_episodes} episodes: {avg_reward:.2f} {avg_norm_score:.2f}")
    return avg_reward, std_reward, avg_norm_score, std_norm_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ### Experimental Setups ###
    parser.add_argument("--exp", default='exp_1', type=str)                    # Experiment ID
    parser.add_argument('--device', default=0, type=int)                       # device, {"cpu", "cuda", "cuda:0", "cuda:1"}, etc
    parser.add_argument("--env_name", default="kitchen-partial-v0", type=str)  # OpenAI gym environment name
    parser.add_argument("--dir", default="results", type=str)                    # Logging directory
    parser.add_argument("--seed", default=0, type=int)                         # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--num_steps_per_epoch", default=1000, type=int)

    ### Optimization Setups ###
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--lr_decay", action='store_true')
    parser.add_argument('--early_stop', action='store_true')
    parser.add_argument('--save_best_model', action='store_true')

    ### RL Parameters ###
    parser.add_argument("--discount", default=0.99, type=float)
    parser.add_argument("--tau", default=0.005, type=float)

    ### Diffusion Setting ###
    parser.add_argument("--T", default=5, type=int)
    parser.add_argument("--beta_schedule", default='vp', type=str)
    ### Algo Choice ###
    parser.add_argument("--algo", default="ql", type=str)  # ['bc', 'ql']
    parser.add_argument("--ms", default='offline', type=str, help="['online', 'offline']")
    # parser.add_argument("--top_k", default=1, type=int)



    args = parser.parse_args()
    args.device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    args.output_dir = f'{args.dir}'

    if args.env_name.startswith("maze2d-"):
        target_steps = 200000
        args.num_epochs = target_steps // args.num_steps_per_epoch

        if target_steps % args.num_steps_per_epoch != 0:
            args.num_epochs += 1
    else:
        args.num_epochs = hyperparameters[args.env_name]['num_epochs']

    args.eval_freq = hyperparameters[args.env_name]['eval_freq']
    args.eval_episodes = 10 if 'v2' in args.env_name else 100

    args.lr = hyperparameters[args.env_name]['lr']
    args.eta = hyperparameters[args.env_name]['eta']
    args.max_q_backup = hyperparameters[args.env_name]['max_q_backup']
    args.reward_tune = hyperparameters[args.env_name]['reward_tune']
    args.gn = hyperparameters[args.env_name]['gn']
    args.top_k = hyperparameters[args.env_name]['top_k']

    # Setup Logging
    file_name = f"{args.env_name}|{args.exp}|diffusion-{args.algo}|T-{args.T}"
    if args.lr_decay: file_name += '|lr_decay'
    file_name += f'|ms-{args.ms}'

    if args.ms == 'offline': file_name += f'|k-{args.top_k}'
    file_name += f'|{args.seed}'

    results_dir = os.path.join(args.output_dir, file_name)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    utils.print_banner(f"Saving location: {results_dir}")

    variant = vars(args)
    variant.update(version=f"Diffusion-Policies-RL")

    env = gym.make(args.env_name)#,disable_env_checker=True

    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    variant.update(state_dim=state_dim)
    variant.update(action_dim=action_dim)
    variant.update(max_action=max_action)
    setup_logger(os.path.basename(results_dir), variant=variant, log_dir=results_dir)
    utils.print_banner(f"Env: {args.env_name}, state_dim: {state_dim}, action_dim: {action_dim}")

    train_agent(env,
                state_dim,
                action_dim,
                max_action,
                args.device,
                results_dir,
                args)
import d4rl