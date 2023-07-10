import os
import time
from collections import deque, OrderedDict

import numpy as np
import torch

from agents import algo, util
from agents.arguments import get_args
from agents.envs import make_vec_envs
from agents.model import Policy
from agents.storage import DictRolloutStorage
from evaluation import evaluate
from lilgym.data.utils import get_data
from lilgym.envs.utils import set_seeds
from lilgym.envs.utils_action import TowerStop, ScatterStop

from agents.common.preprocessing import get_obs_shape
from agents.common.util import get_optimizer_from_name, get_scheduler_from_name

import wandb


def main():
    args = get_args()
    
    if args.wandb:
        wandb.init(
            project=f'{args.env_opt}-{args.learn_opt}',
            name=f'{args.wandb_run_name}',
        )
        wandb.define_metric("train/step")
        wandb.define_metric("train/*", step_metric="train/step")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = os.path.expanduser(args.log_dir)
    eval_log_dir = log_dir + "_eval"
    util.cleanup_log_dir(log_dir)
    util.cleanup_log_dir(eval_log_dir)

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    # Data
    data = get_data(args.env_opt, args.learn_opt, 'train')
    dev_data = get_data(args.env_opt, args.learn_opt, 'dev')
    # valid_data = get_data(args.env_opt, args.learn_opt, 'valid')

    # Set seeds for torch, numpy and random
    set_seeds(args.seed)

    # Initialize env and set random seed for the env
    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                         args.gamma, args.log_dir, device, False, 
                         data=data,
                         stop_forcing=args.stop_forcing)
    
    actor_critic = Policy(
        get_obs_shape(envs.observation_space),
        envs.action_space,
        base_kwargs={'recurrent': args.recurrent_policy,
                    'learn_opt': args.learn_opt,
                    'env_opt': args.env_opt,
                    'text_feat': args.text_feat,
                    'eval_mode': args.eval_mode},
        custom_model=args.model)
    
    optimizer_sd = None
    scheduler_sd = None
    if args.load_model:
        actor_critic.load_state_dict(torch.load(args.load_model + ".h5"), strict=False)
        optimizer_sd = args.load_model + "_optim.h5"
        scheduler_sd = args.load_model + "_scheduler.h5"

    actor_critic.to(device)

    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes
    num_warmup_updates = int(args.warmup_percent * num_updates)

    optimizer = get_optimizer_from_name(args.optim_type, actor_critic, args.lr, args.eps, optimizer_sd)
    scheduler = get_scheduler_from_name(args.scheduler, optimizer, num_warmup_updates, num_updates, scheduler_sd)

    if args.algo == 'ppo':
        agent = algo.PPO(
            actor_critic,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            optimizer,
            scheduler=scheduler,
            max_grad_norm=args.max_grad_norm)
    
    rollouts = DictRolloutStorage(args.num_steps, args.num_processes,
                              get_obs_shape(envs.observation_space), envs.action_space,
                              actor_critic.recurrent_hidden_state_size)
    
    # For fixed text embedding: compute and cache
    if args.text_feat in ["bertfix"]:
        for dataset in [data, dev_data]:
            for k in dataset.keys():
                actor_critic.base.precompute_bert_embedding(dataset[k]["sentence"])

    obs, infos = envs.reset()

    if isinstance(obs, OrderedDict):
        for k, v in obs.items():
            if isinstance(v, torch.Tensor):
                rollouts.obs[k][0].copy_(v)
            elif isinstance(v, np.ndarray):
                if k == "sentence": # type: str
                    rollouts.obs[k][0] = v.copy() 
                else: # "image" or "target"
                    rollouts.obs[k][0] = torch.from_numpy(v.copy())
    else:
        rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)
    episode_acc = deque(maxlen=10)
    episode_acc_nosf = deque(maxlen=10) # nosf: evaluation with non-stop forcing

    train_episode_acc = [0]
    dev_episode_acc = [0]
    
    train_episode_acc_nosf = [0]
    dev_episode_acc_nosf = [0]

    if args.stop_forcing:
        if args.env_opt == "tower":
            STOP_ACTION_TENSOR = torch.Tensor([TowerStop().to_array()]).to(device)
        else:
            STOP_ACTION_TENSOR = torch.Tensor([ScatterStop().to_array()]).to(device)

    start = time.time()

    for j in range(num_updates):

        update_acc = []
        
        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            util.update_linear_schedule(
                agent.optimizer, j, num_updates,
                agent.optimizer.lr if args.algo == "acktr" else args.lr)

        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                if isinstance(rollouts.obs, dict):
                    value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                        rollouts.get_obs(step), rollouts.recurrent_hidden_states[step],
                        rollouts.masks[step])
                else:
                    value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                        rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                        rollouts.masks[step])

            # Obser reward and next obs
            step_action = action.clone()
            obs, reward, done, truncated, infos = envs.step(step_action)

            for idx, info in enumerate(infos):
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])
                    episode_acc.append(info['accuracy'])
                    episode_acc_nosf.append(info['accuracy_nosf'])
                    update_acc.append(info['accuracy'])

                # Stop forcing
                if ('accuracy' in info.keys() and args.stop_forcing):
                    with torch.no_grad():
                        # If a valid goal state is reached
                        if info['accuracy'] == 1:
                            action = STOP_ACTION_TENSOR
                            if isinstance(rollouts.obs, dict):
                                _state = rollouts.get_obs(step)
                            else:
                                _state = torch.unsqueeze(rollouts.obs[step][idx], dim=0)
                            _masks = rollouts.masks[step][idx]
                            _hidden = rollouts.recurrent_hidden_states[step][idx]
                            _, _action_log_prob, _, _ = actor_critic.evaluate_actions(_state, _hidden, _masks, action)
                            action_log_prob[idx] = _action_log_prob

            # Need to uniformize length when putting in storage
            if args.env_opt == "tower":
                padded_action = torch.full((1, 3), -1)
                padded_action[0][:len(action[0])] = action[0]
            elif args.env_opt == "scatter":
                padded_action = torch.full((1, 6), -1)
                padded_action[0][:len(action[0])] = action[0]

            # If done, then clean the history of observations
            assert len(done) == len(truncated)
            done_or_truncated = [(done[i] or truncated[i]) for i in range(len(done))]
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done_or_truncated])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            rollouts.insert(obs, recurrent_hidden_states, padded_action,
                        action_log_prob, value, reward, masks, bad_masks)

        with torch.no_grad():
            if isinstance(rollouts.obs, dict):
                next_value = actor_critic.get_value(
                    rollouts.get_obs(-1), rollouts.recurrent_hidden_states[-1],
                    rollouts.masks[-1]).detach()
            else:
                next_value = actor_critic.get_value(
                    rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                    rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)
        
        value_loss, action_loss, dist_entropy = agent.update(rollouts)
        
        rollouts.after_update()

        # Save for every interval-th episode or for the last epoch
        if (j % args.save_interval == 0
                or j == num_updates - 1) and args.save_dir != "":
            save_path = os.path.join(args.save_dir, args.algo)
            try:
                os.makedirs(save_path)
            except OSError:
                pass
            
            torch.save(actor_critic.state_dict(), 
                       os.path.join(save_path, args.env_name + f"_{j}.h5"))
            torch.save(agent.optimizer.state_dict(), 
                       os.path.join(save_path, args.env_name + f"_{j}_optim.h5"))
            if agent.scheduler:
                 torch.save(agent.scheduler.state_dict(), 
                        os.path.join(save_path, args.env_name + f"_{j}_scheduler.h5"))

        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}  mean/median acc {:.1f}/{:.1f} min/max acc {:.1f}/{:.1f} mean/median acc no sf {:.1f}/{:.1f} min/max acc no sf {:.1f}/{:.1f}\n"# full train acc {:.1f} full dev acc {:.1f}\n"
                .format(j, total_num_steps,
                        int(total_num_steps / (end - start)),
                        len(episode_rewards), 
                        np.mean(episode_rewards), np.median(episode_rewards), 
                        np.min(episode_rewards), np.max(episode_rewards), 
                        np.mean(episode_acc),  np.median(episode_acc), 
                        np.min(episode_acc), np.max(episode_acc),
                        np.mean(episode_acc_nosf), np.median(episode_acc_nosf), 
                        np.min(episode_acc_nosf), np.max(episode_acc_nosf),
                        dist_entropy, value_loss, action_loss))

        if (args.eval_interval is not None and len(episode_rewards) > 1
                and j % args.eval_interval == 0):
            
            train_acc = evaluate(actor_critic, None, args.env_name, args.seed,
                     args.num_processes, eval_log_dir, device, 'Train', data,
                     stop_forcing=args.stop_forcing)
            train_episode_acc.append(train_acc)
            print(f"> Best train_acc so far: {round(max(train_episode_acc), 5)}")
            
            dev_acc = evaluate(actor_critic, None, args.env_name, args.seed,
                     args.num_processes, eval_log_dir, device, 'Dev', dev_data,
                     stop_forcing=args.stop_forcing)
            dev_episode_acc.append(dev_acc)
            print(f"> Best dev_acc so far: {round(max(dev_episode_acc), 5)}")
            
            if args.stop_forcing:
                train_acc_nosf = evaluate(actor_critic, None, args.env_name, args.seed,
                     args.num_processes, eval_log_dir, device, 'Train', data,
                     stop_forcing=False)
                train_episode_acc_nosf.append(train_acc_nosf)
                print(f"> Best train_acc_nosf so far: {round(max(train_episode_acc_nosf), 5)}")

                dev_acc_nosf = evaluate(actor_critic, None, args.env_name, args.seed,
                     args.num_processes, eval_log_dir, device, 'Dev', dev_data,
                     stop_forcing=False)
                dev_episode_acc_nosf.append(dev_acc_nosf)
                print(f"> Best dev_acc_nosf so far: {round(max(dev_episode_acc_nosf), 5)}")
                
                if args.wandb:
                    wandb.log({"train/step": j,
                               "train/train_acc_sf": train_acc,
                               "train/dev_acc_sf": dev_acc, 
                               "train/train_acc_nosf": train_acc_nosf,
                               "train/dev_acc_nosf": dev_acc_nosf})
            else:
                if args.wandb:
                    wandb.log({"train/step": j,
                               "train/train_acc_nosf": train_acc,
                               "train/dev_acc_nosf": dev_acc})


if __name__ == "__main__":
    main()
