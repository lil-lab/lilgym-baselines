import numpy as np
import torch

from agents.envs import make_vec_envs


def evaluate(actor_critic, obs_rms, env_name, seed, num_processes, eval_log_dir,
             device, mode, data, 
             stop_forcing=False):

    eval_envs = make_vec_envs(env_name, seed + num_processes, num_processes,
                              None, eval_log_dir, device, True, eval=True, data=data, stop_forcing=stop_forcing)

    eval_episode_rewards = []
    eval_episode_acc = []

    if "FlipIt" in env_name:
        eval_episode_rewards_tgt_false = []
        eval_episode_acc_tgt_false = []
        eval_episode_rewards_tgt_true = []
        eval_episode_acc_tgt_true = []

    curr_obs = []
    curr_rew = []
    curr_act = []

    obs, infos = eval_envs.reset()
    eval_recurrent_hidden_states = torch.zeros(
        num_processes, actor_critic.recurrent_hidden_state_size, device=device)
    eval_masks = torch.zeros(num_processes, 1, device=device)
    curr_obs.append(obs)

    nb_actions = 0
    tmp_count = 0
    while True:
        with torch.no_grad():
            _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                obs,
                eval_recurrent_hidden_states,
                eval_masks,
                deterministic=True)
        
        curr_act.append(action)
        nb_actions += 1
        step_action = action.clone()

        try:
            obs, step_reward, done, truncated, infos = eval_envs.step(step_action)
            tmp_count += 1

            curr_obs.append(obs)
            curr_rew.append(step_reward)

        except IndexError:
            print("IndexError")
            break

        eval_masks = torch.tensor(
            [[0.0] if done_ else [1.0] for done_ in done],
            dtype=torch.float32,
            device=device)

        for info in infos:
            if 'episode' in info.keys():
                eval_episode_rewards.append(info['episode']['r'])
                eval_episode_acc.append(info['accuracy'])
                if "FlipIt" in env_name:
                    if 'terminal_observation' in info.keys():
                        if info['terminal_observation']['target'] == 0:
                            eval_episode_rewards_tgt_false.append(info['episode']['r'])
                            eval_episode_acc_tgt_false.append(info['accuracy'])
                        if info['terminal_observation']['target'] == 1:
                            eval_episode_rewards_tgt_true.append(info['episode']['r'])
                            eval_episode_acc_tgt_true.append(info['accuracy']) 
        if (done or truncated) and 'nb_to_evaluate' in infos[0].keys() and infos[0]['nb_to_evaluate'] == 0:
            break
 
    eval_envs.close()
    print("{} - Evaluation using {} episodes: mean reward {:.5f} mean accur {:.5f} \n".format(
        mode, len(eval_episode_rewards), np.mean(eval_episode_rewards), np.mean(eval_episode_acc)))

    if "FlipIt" in env_name:
        print("{} - False - Evaluation using {} episodes: mean reward {:.5f} mean accur {:.5f} \n".format(
            mode, len(eval_episode_rewards_tgt_false), np.mean(eval_episode_rewards_tgt_false), np.mean(eval_episode_acc_tgt_false)))
        print("{} - True - Evaluation using {} episodes: mean reward {:.5f} mean accur {:.5f} \n".format(
            mode, len(eval_episode_rewards_tgt_true), np.mean(eval_episode_rewards_tgt_true), np.mean(eval_episode_acc_tgt_true)))

    return np.mean(eval_episode_acc)