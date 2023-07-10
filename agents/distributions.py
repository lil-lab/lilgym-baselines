import torch
import torch.nn as nn
import torch.nn.functional as F

from .util import init
from torch.distributions.utils import probs_to_logits, logits_to_probs, lazy_property

"""
Modify standard PyTorch distributions so they are compatible with this code.
"""

#
# Standardize distribution interfaces
#

# Categorical
class FixedCategorical(torch.distributions.Categorical):
    def sample(self):
        return super().sample().unsqueeze(-1)

    def log_probs(self, actions):
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1)
            .unsqueeze(-1)
        )

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)


class Categorical(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Categorical, self).__init__()

        init_ = lambda m: init(
            m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=0.01
        )

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x):
        x = self.linear(x)
        return FixedCategorical(logits=x)


class CategoricalMasked(torch.distributions.Categorical):
    # Based on A Closer Look at Invalid Action Masking in Policy Gradient Algorithms (Huang et al., 2022)
    def __init__(self, probs=None, logits=None, validate_args=None, masks=[]):
        self.masks = masks
        if len(self.masks) == 0:
            super(CategoricalMasked, self).__init__(probs, logits, validate_args)
        else:
            self.masks = masks.type(torch.BoolTensor).to("cuda")
            logits = torch.where(self.masks, logits, torch.tensor(-1e8).to("cuda"))
            super(CategoricalMasked, self).__init__(probs, logits, validate_args)

    def entropy(self):
        if len(self.masks) == 0:
            return super(CategoricalMasked, self).entropy()
        p_log_p = self.logits * self.probs
        p_log_p = torch.where(self.masks, p_log_p, torch.tensor(0.0).to("cuda"))
        return -p_log_p.sum(-1)


class MultiCategorical(nn.Module):
    def __init__(self, latent_dim, nvec):
        """
        # Partly based on stable-baselines3

        Accept masked distribution.
        
        Args:
            latent_dim: Dimension of the last layer of the policy network
            (before the action layer)
            nvec
        """
        super(MultiCategorical, self).__init__()

        self.action_dims = nvec

        init_ = lambda m: init(
            m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=0.01
        )

        # action logits
        self.linear = init_(nn.Linear(latent_dim, sum(self.action_dims)))

    def proba_distribution(self, action_logits: torch.Tensor, action_mask=[]):
        if len(action_mask) == 0:
            self.distribution = [
                torch.distributions.Categorical(logits=split)
                for split in torch.split(action_logits, tuple(self.action_dims), dim=1)
            ]
        else:
            split_action_logits = torch.split(
                action_logits, tuple(self.action_dims), dim=1
            )
            split_action_masks = torch.split(
                action_mask, tuple(self.action_dims), dim=1
            )
            self.distribution = [
                CategoricalMasked(logits=split, masks=iam)
                for (split, iam) in zip(split_action_logits, split_action_masks)
            ]
        return self

    def log_prob(self, actions: torch.Tensor, action_log_mask) -> torch.Tensor:
        # Extract each discrete action and compute log prob for their respective distributions
        if len(action_log_mask) == 0:
            return torch.stack(
                [
                    dist.log_prob(action)
                    for dist, action in zip(
                        self.distribution, torch.unbind(actions, dim=1)
                    )
                ],
                dim=1,
            ).sum(dim=1)
        else:
            action_log_mask = torch.tensor(action_log_mask).to("cuda")

            logs = []
            for dist, action, mask in zip(
                self.distribution,
                torch.unbind(actions, dim=1),
                torch.unbind(action_log_mask, dim=1),
            ):
                output = dist.log_prob(action) * mask.float()
                logs.append(output)
            return torch.stack(logs, dim=1).sum(dim=1)

    def log_probs(self, actions, action_log_mask=[]):
        return (
            self.log_prob(actions.squeeze(-1), action_log_mask)
            .view(actions.size(0), -1)
            .sum(-1)
            .unsqueeze(-1)
        )

    @lazy_property
    def probs(self):
        return [dist.probs for dist in self.distribution]

    def entropy(self) -> torch.Tensor:
        return torch.stack([dist.entropy() for dist in self.distribution], dim=1).sum(
            dim=1
        )

    def sample(self) -> torch.Tensor:
        return torch.stack([dist.sample() for dist in self.distribution], dim=1)

    def mode(self) -> torch.Tensor:
        return torch.stack(
            [torch.argmax(dist.probs, dim=1) for dist in self.distribution], dim=1
        )

    def forward(self, x, action_mask=[]):
        x = self.linear(x)
        if len(action_mask) > 0:
            action_mask = torch.unsqueeze(torch.tensor(action_mask), dim=0)
        return self.proba_distribution(action_logits=x, action_mask=action_mask)
