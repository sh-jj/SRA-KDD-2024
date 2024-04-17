

import copy
import os
import random
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import d4rl
import gym
import numpy as np
import pyrallis
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.distributions import Normal
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import TanhTransform
from torch.optim.lr_scheduler import CosineAnnealingLR

TensorBatch = List[torch.Tensor]

MEAN_MIN = -9.0
MEAN_MAX = 9.0
LOG_STD_MIN = -5
LOG_STD_MAX = 2
LOG_PI_NORM_MAX = 10
LOG_PI_NORM_MIN = -20

EPS = 1e-7
EXP_ADV_MAX = 100.0

def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)

def asymmetric_l2_loss(u: torch.Tensor, tau: float) -> torch.Tensor:
    return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)

class Squeeze(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(dim=self.dim)


class MLP(nn.Module):
    def __init__(
        self,
        dims,
        activation_fn: Callable[[], nn.Module] = nn.ReLU,
        output_activation_fn: Callable[[], nn.Module] = None,
        squeeze_output: bool = False,
        dropout: Optional[float] = None,
    ):
        super().__init__()
        n_dims = len(dims)
        if n_dims < 2:
            raise ValueError("MLP requires at least two dims (input and output)")

        layers = []
        for i in range(n_dims - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(activation_fn())

            if dropout is not None:
                layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(dims[-2], dims[-1]))
        if output_activation_fn is not None:
            layers.append(output_activation_fn())
        if squeeze_output:
            if dims[-1] != 1:
                raise ValueError("Last dim must be 1 when squeezing")
            layers.append(Squeeze(-1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)



class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()

        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.mu_head = nn.Linear(256, action_dim)
        self.sigma_head = nn.Linear(256, action_dim)

        self.state_dim = state_dim
        self.action_dim = action_dim

    def _get_outputs(self, state):
        a = F.relu(self.fc1(state))
        a = F.relu(self.fc2(a))
        mu = self.mu_head(a)
        mu = torch.clip(mu, MEAN_MIN, MEAN_MAX)
        log_sigma = self.sigma_head(a)
        log_sigma = torch.clip(log_sigma, LOG_STD_MIN, LOG_STD_MAX)
        sigma = torch.exp(log_sigma)

        a_distribution = TransformedDistribution(
            Normal(mu, sigma), TanhTransform(cache_size=1)
        )
        a_tanh_mode = torch.tanh(mu)
        return a_distribution, a_tanh_mode

    def forward(self, state):
        a_dist, a_tanh_mode = self._get_outputs(state)
        action = a_dist.rsample()
        logp_pi = a_dist.log_prob(action).sum(axis=-1)
        return action, logp_pi, a_tanh_mode

    def get_log_density(self, state, action):
        a_dist, _ = self._get_outputs(state)
        action_clip = torch.clip(action, -1. + EPS, 1. - EPS)
        logp_action = a_dist.log_prob(action_clip)
        return logp_action

    @torch.no_grad()
    def select_action(self, state, device):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        _, _, action = self.forward(state)
        return action.cpu().data.numpy().flatten()

    @torch.no_grad()
    def select_action_multi(self, state, device):
        state = torch.FloatTensor(state.reshape(-1, self.state_dim)).to(device)
        _, _, action = self.forward(state)
        return action.cpu().data.numpy().reshape(-1, self.action_dim)
    
    @torch.no_grad()
    def select_action_with_prob(self, state, device):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        _, _, action = self.forward(state)
        prob = torch.sum(self.get_log_density(state, action),dim=-1)
        
        return action.cpu().data.numpy().flatten(), prob.cpu().data.numpy().flatten()
    
    def act(self, state):

        # print(state)

        _, _, action = self.forward(state)

        # print(action)

        # # print(state.shape)
        # print(action.shape)
        # exit(0)
        return action

class TwinQ(nn.Module):
    def __init__(
        self, state_dim: int, action_dim: int, hidden_dim: int = 256, n_hidden: int = 2
    ):
        super().__init__()
        dims = [state_dim + action_dim, *([hidden_dim] * n_hidden), 1]
        self.q1 = MLP(dims, squeeze_output=True)
        self.q2 = MLP(dims, squeeze_output=True)

    def both(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([state, action], 1)
        return self.q1(sa), self.q2(sa)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return torch.min(*self.both(state, action))


class ValueFunction(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 256, n_hidden: int = 2):
        super().__init__()
        dims = [state_dim, *([hidden_dim] * n_hidden), 1]
        self.v = MLP(dims, squeeze_output=True)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.v(state)


class IQL:
    def __init__(
        self,
        state_dim: int, 
        action_dim: int, 
        max_action: float, 
        lr=3e-4,
        wd=5e-3,
        discount: float = 0.99,
        tau: float = 0.005,
        device: str = 'cpu',
        alpha: float = 2.5,
        beta: float = 3.0,
        iql_tau: float = 0.7,
        bc_freq: int = 1,
        policy_freq:int=1,
        max_steps: int = 500000
    ):
        self.max_action = max_action
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.wd = wd
        self.discount = discount
        self.tau = tau
        self.device = device
        self.beta = beta
        self.iql_tau = iql_tau
        self.bc_freq = bc_freq
        self.policy_freq = policy_freq
        self.max_steps = max_steps
        self.total_it = 0
        self.alpha = alpha
        
        self.qf = TwinQ(state_dim, action_dim).to(device)
        self.q_target = copy.deepcopy(self.qf).requires_grad_(False).to(device)
        self.vf = ValueFunction(state_dim).to(device)
        self.actor = Actor(state_dim,action_dim).to(device)
        self.v_optimizer = torch.optim.Adam(self.vf.parameters(), lr=lr,weight_decay=wd)
        self.q_optimizer = torch.optim.Adam(self.qf.parameters(), lr=lr,weight_decay=wd)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr,weight_decay=wd)
        self.actor_lr_schedule = CosineAnnealingLR(self.actor_optimizer, max_steps)
        
        
    def _update_v(self, observations, actions, log_dict) -> torch.Tensor:
        # Update value function
        with torch.no_grad():
            target_q = self.q_target(observations, actions)

        v = self.vf(observations)
        adv = target_q - v
        v_loss = asymmetric_l2_loss(adv, self.iql_tau)
        log_dict["value_loss"].append(v_loss.item())
        self.v_optimizer.zero_grad()
        v_loss.backward()
        self.v_optimizer.step()
        return adv

    def _update_q(
        self,
        next_v: torch.Tensor,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        terminals: torch.Tensor,
        log_dict: Dict,
    ):
        targets = rewards + (1.0 - terminals.float()) * self.discount * next_v.detach()
        qs = self.qf.both(observations, actions)
        q_loss = sum(F.mse_loss(q, targets) for q in qs) / len(qs)
        log_dict["q_loss"].append(q_loss.item())
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        # Update target Q network
        soft_update(self.q_target, self.qf, self.tau)

    def train(self, buffer_e, buffer_o, iterations=1, batch_size=256):
        self.actor.train()

        log_dict = {"actor_loss": [], "value_loss": [], "actor_loss_iql": [], "actor_loss_bc": [], "q_loss": []}

        for iter_i in range(iterations):
            
            
            self.total_it += 1
            state_e, action_e, next_state_e, reward_e, not_done_e, _ = buffer_e.sample(batch_size)

            state_o, action_o, next_state_o, reward_o, not_done_o, _ = buffer_o.sample(batch_size)

            state = torch.cat([state_e, state_o], dim=0)
            action = torch.cat([action_e, action_o], dim=0)
            next_state = torch.cat([next_state_e, next_state_o], dim=0)
            reward = torch.cat([reward_e, reward_o], dim=0)
            not_done = torch.cat([not_done_e, not_done_o], dim=0)
            
            dones = 1 - not_done


            with torch.no_grad():
                next_v = self.vf(next_state)
            # Update value function
            adv = self._update_v(state, action, log_dict)
            reward = reward.squeeze(dim=-1)
            dones = dones.squeeze(dim=-1)
            # Update Q function
            self._update_q(next_v, state, action, reward, dones, log_dict)
            # Update actor
            actor_loss = 0
            if self.total_it % self.bc_freq == 0:
                log_pi_e = self.actor.get_log_density(state_e, action_e)

                bckl_loss = -torch.sum(log_pi_e, 1)

                # lmbda2 = 1. / bckl_loss.abs().mean().detach()

                log_dict["actor_loss_bc"].append(bckl_loss.mean().item())

                actor_loss_bc = bckl_loss.mean()  # * lmbda2

                
                log_pi_o = self.actor.get_log_density(state_o, action_o)


                bckl_loss_o = -torch.sum(log_pi_o, 1)
                bckl_loss_o = (bckl_loss_o * reward_o).mean()

                actor_loss_bc += bckl_loss_o


                actor_loss += actor_loss_bc
            if self.total_it % self.policy_freq == 0:
                # self._update_policy(adv, state, action, log_dict)
            
            
                exp_adv = torch.exp(self.beta * adv.detach()).clamp(max=EXP_ADV_MAX)

                log_pi_u = self.actor.get_log_density(state, action)
                bc_losses = -torch.sum(log_pi_u, 1)

                policy_loss = torch.mean(exp_adv * bc_losses)
                actor_loss += self.alpha * policy_loss
                log_dict["actor_loss_iql"].append(policy_loss.item())
            log_dict["actor_loss"].append(actor_loss.item())

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            self.actor_lr_schedule.step()

        return log_dict
    
    def save(self, filename):
        torch.save(self.qf.state_dict(), filename + "_qf")
        torch.save(self.q_optimizer.state_dict(), filename + "_q_optimizer")
        torch.save(self.vf.state_dict(), filename + "_vf")
        torch.save(self.v_optimizer.state_dict(), filename + "_v_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")
        # torch.save(self.actor_lr_schedule.state_dict, filename+"_actor_lr_schedule")
    def load(self, filename):
        if not torch.cuda.is_available():
            self.qf.load_state_dict(torch.load(filename + "_qf", map_location=torch.device('cpu')))
            self.q_optimizer.load_state_dict(torch.load(filename + "_q_optimizer", map_location=torch.device('cpu')))
            self.q_target = copy.deepcopy(self.qf)

            self.vf.load_state_dict(torch.load(filename + "_vf", map_location=torch.device('cpu')))
            self.v_optimizer.load_state_dict(torch.load(filename + "_v_optimizer", map_location=torch.device('cpu')))

            self.actor.load_state_dict(torch.load(filename + "_actor", map_location=torch.device('cpu')))
            self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer", map_location=torch.device('cpu')))
            # self.actor_lr_schedule.load_state_dict(torch.load(filename+"_actor_lr_schedule", map_location=torch.device('cpu')))
        else:
            self.qf.load_state_dict(torch.load(filename + "_qf"))
            self.q_optimizer.load_state_dict(torch.load(filename + "_q_optimizer"))
            self.q_target = copy.deepcopy(self.qf)

            self.vf.load_state_dict(torch.load(filename + "_vf"))
            self.v_optimizer.load_state_dict(torch.load(filename + "_v_optimizer"))

            self.actor.load_state_dict(torch.load(filename + "_actor"))
            self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
            # self.actor_lr_schedule.load_state_dict(torch.load(filename+"_actor_lr_schedule"))
            
            
            