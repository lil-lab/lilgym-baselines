from abc import ABC
from operator import itemgetter
import numpy as np

import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


class BaseRolloutStorage(ABC):
    def __init__(self, num_steps, num_processes, obs_shape, action_space):
        super().__init__()
        self.num_steps = num_steps
        self.step = 0
        self.rewards = torch.zeros(num_steps, num_processes, 1)
        self.value_preds = torch.zeros(num_steps + 1, num_processes, 1)
        self.returns = torch.zeros(num_steps + 1, num_processes, 1)
        self.action_log_probs = torch.zeros(num_steps, num_processes, 1)

        action_shape = action_space.shape[0]
        self.actions = torch.zeros(num_steps, num_processes, action_shape)

        self.masks = torch.ones(num_steps + 1, num_processes, 1)

        # Masks that indicate whether it's a true terminal state
        # or time limit end state
        self.bad_masks = torch.ones(num_steps + 1, num_processes, 1)

        self.obs_shape = obs_shape

    def to(self, device):
        raise NotImplementedError()

    def insert(self, *args, **kwargs):
        raise NotImplementedError()

    def after_update(self):
        raise NotImplementedError()

    def compute_returns(self, *args, **kwargs):
        raise NotImplementedError()

    def feed_forward_generator(self, *args, **kwargs):
        raise NotImplementedError()

    def recurrent_generator(self, *args, **kwargs):
        raise NotImplementedError()


class DictRolloutStorage(BaseRolloutStorage):
    def __init__(
        self,
        num_steps,
        num_processes,
        obs_shape,
        action_space,
        recurrent_hidden_state_size,
    ):
        super(DictRolloutStorage, self).__init__(
            num_steps, num_processes, obs_shape, action_space
        )

        assert isinstance(
            self.obs_shape, dict
        ), "DictRolloutStorage needs to be used with a Space.Dict observation space"

        self.obs = {}
        for key, input_shape in self.obs_shape.items():
            if key == "sentence":  # str
                self.obs[key] = [
                    ["" for j in range(num_processes)] for i in range(num_steps + 1)
                ]
            elif key == "target":  # bool
                self.obs[key] = torch.zeros(num_steps + 1, num_processes)
            elif key == "image":
                self.obs[key] = torch.zeros(num_steps + 1, num_processes, *input_shape)

        self.recurrent_hidden_states = torch.zeros(
            num_steps + 1, num_processes, recurrent_hidden_state_size
        )

    def to(self, device):
        for key, obs_input_shape in self.obs_shape.items():
            if key == "image" or key == "target":
                self.obs[key] = self.obs[key].to(device)
        self.recurrent_hidden_states = self.recurrent_hidden_states.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.masks = self.masks.to(device)
        self.bad_masks = self.bad_masks.to(device)

    def get_obs(self, idx):
        return {k: v[idx] for k, v in self.obs.items()}

    def insert(
        self,
        obs,
        recurrent_hidden_states,
        actions,
        action_log_probs,
        value_preds,
        rewards,
        masks,
        bad_masks,
    ):
        for k, v in self.obs.items():
            if isinstance(v, torch.Tensor):
                self.obs[k][self.step + 1].copy_(obs[k])
            elif isinstance(v, np.ndarray):
                if k == "sentence":
                    self.obs[k][self.step + 1] = obs[k].copy()
                # key is "image" or "target"
                else:
                    self.obs[k][self.step + 1] = torch.from_numpy(obs[k].copy())
            elif isinstance(v, list):
                if k == "sentence":
                    self.obs[k][self.step + 1] = obs[k].copy()
            else:
                raise Exception(f"Unsupported type for {v}")

        self.recurrent_hidden_states[self.step + 1].copy_(recurrent_hidden_states)
        self.actions[self.step].copy_(actions)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)
        self.bad_masks[self.step + 1].copy_(bad_masks)

        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        for k, v in self.obs.items():
            if isinstance(v, torch.Tensor):
                self.obs[k][0].copy_(self.obs[k][-1])
            elif isinstance(v, np.ndarray):
                if k == "sentence":
                    self.obs[k][0] = self.obs[k][-1].copy()  # ndarray
                # key is "image" or "target"
                else:
                    self.obs[k][0] = torch.from_numpy(self.obs[k][-1].copy())
            elif isinstance(v, list):
                if k == "sentence":
                    self.obs[k][0] = self.obs[k][-1].copy()
            else:
                raise Exception(f"Unsupported type for {v}")

        self.recurrent_hidden_states[0].copy_(self.recurrent_hidden_states[-1])
        self.masks[0].copy_(self.masks[-1])
        self.bad_masks[0].copy_(self.bad_masks[-1])

    def compute_returns(
        self, next_value, use_gae, gamma, gae_lambda, use_proper_time_limits=True
    ):
        if use_proper_time_limits:
            if use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.size(0))):
                    delta = (
                        self.rewards[step]
                        + gamma * self.value_preds[step + 1] * self.masks[step + 1]
                        - self.value_preds[step]
                    )
                    gae = delta + gamma * gae_lambda * self.masks[step + 1] * gae
                    gae = gae * self.bad_masks[step + 1]
                    self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.size(0))):
                    self.returns[step] = (
                        (
                            self.returns[step + 1] * gamma * self.masks[step + 1]
                            + self.rewards[step]
                        )
                        * self.bad_masks[step + 1]
                        + (1 - self.bad_masks[step + 1]) * self.value_preds[step]
                    )
        else:
            if use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.size(0))):
                    delta = (
                        self.rewards[step]
                        + gamma * self.value_preds[step + 1] * self.masks[step + 1]
                        - self.value_preds[step]
                    )
                    gae = delta + gamma * gae_lambda * self.masks[step + 1] * gae
                    self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.size(0))):
                    self.returns[step] = (
                        self.returns[step + 1] * gamma * self.masks[step + 1]
                        + self.rewards[step]
                    )

    def feed_forward_generator(
        self, advantages, num_mini_batch=None, mini_batch_size=None
    ):
        num_steps, num_processes = self.rewards.size()[0:2]
        batch_size = num_processes * num_steps

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                "PPO requires the number of processes ({}) "
                "* number of steps ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(
                    num_processes, num_steps, num_processes * num_steps, num_mini_batch
                )
            )
            mini_batch_size = batch_size // num_mini_batch
        sampler = BatchSampler(
            SubsetRandomSampler(range(batch_size)), mini_batch_size, drop_last=True
        )
        for indices in sampler:
            obs_batch = {
                "image": self.obs["image"][:-1].view(-1, *self.obs["image"].size()[2:])[
                    indices
                ],
                "sentence": list(
                    itemgetter(*indices)(
                        np.squeeze(np.reshape(self.obs["sentence"][:-1], (-1, 1)), 1)
                    )
                ),
                "target": self.obs["target"][:-1].view(
                    -1, *self.obs["target"].size()[2:]
                )[indices],
            }

            recurrent_hidden_states_batch = self.recurrent_hidden_states[:-1].view(
                -1, self.recurrent_hidden_states.size(-1)
            )[indices]
            actions_batch = self.actions.view(-1, self.actions.size(-1))[indices]
            value_preds_batch = self.value_preds[:-1].view(-1, 1)[indices]
            return_batch = self.returns[:-1].view(-1, 1)[indices]
            masks_batch = self.masks[:-1].view(-1, 1)[indices]
            old_action_log_probs_batch = self.action_log_probs.view(-1, 1)[indices]
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages.view(-1, 1)[indices]

            yield obs_batch, recurrent_hidden_states_batch, actions_batch, value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ
