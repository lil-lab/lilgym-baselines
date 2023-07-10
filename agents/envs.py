# Based on https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail

import collections

import gymnasium as gym
import numpy as np
import torch

from gymnasium import spaces

from stable_baselines3.common.vec_env import SubprocVecEnv, VecEnvWrapper
from .vec_env.dummy_vec_env import DummyVecEnv
from .common.preprocessing import get_obs_shape


def get_obs_img_shape(obs_space):
    obs_full_shape = get_obs_shape(obs_space)
    if isinstance(obs_full_shape, tuple):
        obs_shape = obs_full_shape
    elif isinstance(obs_full_shape, dict):
        obs_shape = obs_full_shape["image"]
    return obs_shape


def make_env(env_id, seed, rank, log_dir, allow_early_resets, data, eval, stop_forcing):
    def _thunk():
        env = gym.make(
            env_id,
            data=data,
            evaluate=eval,
            stop_forcing=stop_forcing,
            disable_env_checker=True,
        )
        env = OriginalReturnWrapper(env)

        env.seed(seed + rank)
        env.action_space.seed(seed + rank)

        # If the input has shape (W, H, 3), wrap for PyTorch convolutions
        obs_shape = get_obs_img_shape(env.observation_space)

        # Transpose observation space for images
        if len(obs_shape) == 3 and obs_shape[2] in [1, 3]:
            env = TransposeImage(env, op=[2, 0, 1])

        return env

    return _thunk


def make_vec_envs(
    env_name,
    seed,
    num_processes,
    gamma,
    log_dir,
    device,
    allow_early_resets,
    num_frame_stack=None,
    eval=False,
    data=None,
    stop_forcing=None,
):
    envs = [
        make_env(
            env_name,
            seed,
            i,
            log_dir,
            allow_early_resets,
            data,
            eval,
            stop_forcing=stop_forcing,
        )
        for i in range(num_processes)
    ]

    if len(envs) > 1:
        envs = SubprocVecEnv(envs)
    else:
        envs = DummyVecEnv(envs)

    envs = VecPyTorch(envs, device)

    return envs


class OriginalReturnWrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.total_rewards = 0

    def step(self, action):
        try:
            obs, reward, done, truncated, info = self.env.step(action)
        except:
            obs, reward, done, truncated, info = self.env.step(action.item())
        self.total_rewards += reward
        if done or truncated:
            info["episode"] = {"r": self.total_rewards}
            self.total_rewards = 0
        return obs, reward, done, truncated, info

    def reset(self):
        return self.env.reset()


class TransposeObs(gym.ObservationWrapper):
    def __init__(self, env=None):
        """
        Transpose observation space (base class)
        """
        super(TransposeObs, self).__init__(env)


class TransposeImage(TransposeObs):
    def __init__(self, env=None, op=[2, 0, 1]):
        """
        Transpose observation space for images
        ex: (H, W, C) -> (C, H, W)
        """
        super(TransposeImage, self).__init__(env)
        assert len(op) == 3, "Error: Operation, " + str(op) + ", must be dim3"
        self.op = op

        obs_shape = get_obs_shape(self.observation_space)

        if isinstance(self.observation_space, spaces.Box):
            self.observation_space = spaces.Box(
                self.observation_space.low[0, 0, 0],
                self.observation_space.high[0, 0, 0],
                [obs_shape[self.op[0]], obs_shape[self.op[1]], obs_shape[self.op[2]]],
                dtype=self.observation_space.dtype,
            )
        elif isinstance(self.observation_space, spaces.Dict):
            self.observation_space = spaces.Dict(
                {
                    "image": spaces.Box(
                        low=self.observation_space["image"].low[0, 0, 0],
                        high=self.observation_space["image"].high[0, 0, 0],
                        shape=[
                            obs_shape["image"][self.op[0]],
                            obs_shape["image"][self.op[1]],
                            obs_shape["image"][self.op[2]],
                        ],
                    ),
                    "sentence": spaces.Text(max_length=320),
                    "target": spaces.Discrete(2),
                }
            )

    def observation(self, ob):
        if isinstance(ob, dict):
            ob["image"] = ob["image"].transpose(self.op[0], self.op[1], self.op[2])
            return ob
        else:
            return ob.transpose(self.op[0], self.op[1], self.op[2])


class VecPyTorch(VecEnvWrapper):
    def __init__(self, venv, device):
        """Return only every `skip`-th frame"""
        super(VecPyTorch, self).__init__(venv)
        self.device = device

    def reset(self):
        obs, infos = self.venv.reset()
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float().to(self.device)
        elif isinstance(obs, collections.OrderedDict):
            obs["image"] = torch.from_numpy(obs["image"]).float().to(self.device)
            obs["target"] = torch.from_numpy(obs["target"]).to(self.device)
        return obs, infos

    def step_async(self, actions):
        if isinstance(actions, torch.LongTensor):
            # Squeeze the dimension for discrete actions
            actions = actions.squeeze(1)
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, truncated, info = self.venv.step_wait()
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float().to(self.device)
        elif isinstance(obs, collections.OrderedDict):
            obs["image"] = torch.from_numpy(obs["image"]).float().to(self.device)
            obs["target"] = torch.from_numpy(obs["target"]).to(self.device)
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        return obs, reward, done, truncated, info
