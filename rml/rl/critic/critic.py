from stable_baselines3.common.policies import BaseModel


from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from torch import nn


from stable_baselines3.common.preprocessing import get_action_dim, is_image_space, maybe_transpose, preprocess_obs
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    create_mlp,
)



class ContinuousCritic(BaseModel):
    """
    Critic network(s) for DDPG/SAC/TD3.
    It represents the action-state value function (Q-value function).
    Compared to A2C/PPO critics, this one represents the Q-value
    and takes the continuous action as input. It is concatenated with the state
    and then fed to the network which outputs a single value: Q(s, a).
    For more recent algorithms like SAC/TD3, multiple networks
    are created to give different estimates.

    By default, it creates two critic networks used to reduce overestimation
    thanks to clipped Q-learning (cf TD3 paper).

    :param observation_space: Obervation space
    :param action_space: Action space
    :param net_arch: Network architecture
    :param features_extractor: Network to extract features
        (a CNN when using images, a nn.Flatten() layer otherwise)
    :param features_dim: Number of features
    :param activation_fn: Activation function
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param n_critics: Number of critic networks to create.
    :param share_features_extractor: Whether the features extractor is shared or not
        between the actor and the critic (this saves computation time)
    """

    features_extractor: BaseFeaturesExtractor

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        net_arch: List[int],
        features_extractor: BaseFeaturesExtractor,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
        n_critics: int = 2,
        share_features_extractor: bool = True,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )

        action_dim = get_action_dim(self.action_space)

        self.share_features_extractor = share_features_extractor
        self.n_critics = n_critics
        self.q_networks: List[nn.Module] = []
        for idx in range(n_critics):
            q_net_list = create_mlp(features_dim + action_dim, 1, net_arch, activation_fn)
            q_net = nn.Sequential(*q_net_list)
            self.add_module(f"qf{idx}", q_net)
            self.q_networks.append(q_net)

    def forward(self, obs: th.Tensor, actions: th.Tensor, determinstic=False) -> Tuple[th.Tensor, ...]:
        # Learn the features extractor using the policy loss only
        # when the features_extractor is shared with the actor
        with th.set_grad_enabled(not self.share_features_extractor):
            features = self.extract_features(obs, self.features_extractor)
        qvalue_input = th.cat([features, actions], dim=1)
        qvalue =  tuple(q_net(qvalue_input) for q_net in self.q_networks)
        return qvalue


class DistributionalCritic(BaseModel):
    features_extractor: BaseFeaturesExtractor

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        net_arch: List[int],
        features_extractor: BaseFeaturesExtractor,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
        n_critics: int = 2,
        share_features_extractor: bool = True,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )

        action_dim = get_action_dim(self.action_space)

        self.share_features_extractor = share_features_extractor
        self.n_critics = n_critics
        self.q_networks: List[nn.Module] = []


        for idx in range(n_critics):
            q_net = self._create_q_net(idx, features_dim, action_dim, net_arch, activation_fn)
            self.q_networks.append(q_net)

    def  _create_q_net(self, idx, features_dim, action_dim, net_arch, activation_fn):
        # output two, mean and log_std
        q_net_list = create_mlp(features_dim + action_dim, 2, net_arch, activation_fn)
        q_net = nn.Sequential(*q_net_list)
        self.add_module(f"qf{idx}", q_net)
        return q_net

    def forward(self, obs: th.Tensor, actions: th.Tensor, determinstic=False) -> Tuple[th.Tensor, ...]:
        # Learn the features extractor using the policy loss only
        # when the features_extractor is shared with the actor
        with th.set_grad_enabled(not self.share_features_extractor):
            features = self.extract_features(obs, self.features_extractor)
        qvalue_input = th.cat([features, actions], dim=1)

        qvalue = ()
        for q_net in self.q_networks:
            qvalues = q_net(qvalue_input)
            qvalue_sample, q_mean, q_std = self._sample_q_value(qvalues)
            if determinstic:
                qvalue += (q_mean, )
            else:
                qvalue += (qvalue_sample, )
        return qvalue


    def _sample_q_value(self, qvalues):
        q_mean = qvalues[:, 0:1]
        q_log_std = qvalues[:, 1:2]

        q_std = th.ones_like(q_mean) * q_log_std.exp()
        dist = th.distributions.Normal(q_mean, q_std)
        qvalue = dist.rsample()
        return qvalue, q_mean, q_std
    
    def get_q_dist(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, ...]:
        with th.set_grad_enabled(not self.share_features_extractor):
            features = self.extract_features(obs, self.features_extractor)
        qvalue_input = th.cat([features, actions], dim=1)

        qvalue_mean = ()
        qvalue_std = ()
        qvalue = ()
        for q_net in self.q_networks:
            qvalues = q_net(qvalue_input)
            qvalue_sample, q_mean, q_std = self._sample_q_value(qvalues)
            qvalue_mean += (q_mean, )
            qvalue_std += (q_std, )
            qvalue += (qvalue_sample, )
        return qvalue, qvalue_mean, qvalue_std