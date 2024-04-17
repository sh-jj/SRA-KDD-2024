






from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import asdict, dataclass
import os
import sys
from pathlib import Path
import random
import uuid

import d4rl
import gym
import numpy as np
import pyrallis
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import time
from datetime import datetime

from torch.distributions import Normal
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import TanhTransform


TensorBatch = List[torch.Tensor]

MEAN_MIN = -9.0
MEAN_MAX = 9.0
LOG_STD_MIN = -5
LOG_STD_MAX = 2
LOG_PI_NORM_MAX = 10
LOG_PI_NORM_MIN = -20

EPS = 1e-7




class StochasticActor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(StochasticActor, self).__init__()

        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.mu_head = nn.Linear(256, action_dim)
        self.sigma_head = nn.Linear(256, action_dim)

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
    
class StochasticBC(object):  # noqa
    def __init__(
        self,
        state_dim, 
        action_dim, 
        max_action: np.ndarray, 
        discount: float = 0.99,
        lr: float = 1e-4,
        wd: float = 5e-3, 
        device: str = "cpu",
    ):
        

        actor = StochasticActor(state_dim, action_dim).to(device)
        actor_optimizer = torch.optim.Adam(actor.parameters(), lr=lr, weight_decay=wd)

        self.actor = actor
        self.actor_optimizer = actor_optimizer
        self.max_action = max_action
        self.discount = discount

        self.total_it = 0
        self.device = device

    def train(self, buffer, iters=1, batch_size=100):

        self.actor.train()

        log_dict = {"actor_loss": []}
        for i in range(iters):
            self.total_it += 1
        
            # batch_data, batch_belong = buffer.sample(batch_size, return_belong=True)
            # state, action, _, _, _, _ = batch_data

            
            state, action, _, _, _, _ = buffer.sample(batch_size)

            # Compute actor loss
            # pi = self.actor(state)
            # actor_loss = F.mse_loss(pi, action)

            log_pi_e = self.actor.get_log_density(state, action)


            # log_pi_e_clip = torch.clip(log_pi_e, LOG_PI_NORM_MIN, LOG_PI_NORM_MAX)
            # # log_pi_e = log_pi_e_clip
            # log_pi_e_norm = (log_pi_e_clip - LOG_PI_NORM_MIN) / (LOG_PI_NORM_MAX - LOG_PI_NORM_MIN)
            # log_pi_e = log_pi_e_norm


            bc_loss = -torch.sum(log_pi_e, 1)
            actor_loss = torch.mean(bc_loss)

            # print(batch_belong)
            # batch_belong = np.array(batch_belong)

            # print("action: ")
            # print(action[batch_belong == 0])
            # print(action[batch_belong == 1])

            # print("log_pi: ")
            # print(log_pi_e[batch_belong == 0])
            # print(log_pi_e[batch_belong == 1])
            
            # print(bc_loss[batch_belong == 0])
            # print(bc_loss[batch_belong == 1])
            # print(bc_loss[batch_belong == 0].mean(), bc_loss[batch_belong == 1].mean())

            # print("overall: ", actor_loss)

            
            # exit(0)

            log_dict["actor_loss"].append(actor_loss.item())

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

        return log_dict

    @torch.no_grad()
    def val(self, batch: TensorBatch) -> Dict[str, float]:

        self.actor.eval()

        log_dict = {}
        state, action, _, _, _, _ = batch

        # Compute actor loss
        pi = self.actor(state)
        actor_loss = F.mse_loss(pi, action)
        log_dict["actor_loss"] = actor_loss.item()

        return log_dict

    def state_dict(self) -> Dict[str, Any]:
        return {
            "actor": self.actor.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "total_it": self.total_it,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.actor.load_state_dict(state_dict["actor"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.total_it = state_dict["total_it"]

    def save(self, filename):
        torch.save(self.actor.state_dict(), filename + "_actor.pt")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer.pt")


    def load(self, filename):
        if not torch.cuda.is_available():

            self.actor.load_state_dict(torch.load(filename + "_actor.pt", map_location=torch.device('cpu')))
            self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer.pt", map_location=torch.device('cpu')))

        else:

            self.actor.load_state_dict(torch.load(filename + "_actor.pt"))
            self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer.pt"))






class DeterministicActor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, max_action: float):
        super(DeterministicActor, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh(),
        )

        self.max_action = max_action

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.max_action * self.net(state)

    @torch.no_grad()
    def select_action(self, state: np.ndarray, device: str = "cpu") -> np.ndarray:
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        return self(state).cpu().data.numpy().flatten()


class DeterministicBC:
    def __init__(
        self,
        state_dim, 
        action_dim, 
        max_action: np.ndarray, 
        discount: float = 0.99,
        lr: float = 3e-4, 
        device: str = "cpu",
    ):
        

        actor = DeterministicActor(state_dim, action_dim, max_action).to(device)
        actor_optimizer = torch.optim.Adam(actor.parameters(), lr=lr, weight_decay=0.005)

        self.actor = actor
        self.actor_optimizer = actor_optimizer
        self.max_action = max_action
        self.discount = discount

        self.total_it = 0
        self.device = device


    def train(self, buffer, iters=1, batch_size=100):

        self.actor.train()

        log_dict = {"actor_loss": []}
        for i in range(iters):
            self.total_it += 1
        
            state, action, _, _, _, _ = buffer.sample(batch_size)

            # Compute actor loss
            pi = self.actor(state)
            actor_loss = F.mse_loss(pi, action)
            log_dict["actor_loss"] = actor_loss.item()
            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

        return log_dict

    def state_dict(self) -> Dict[str, Any]:
        return {
            "actor": self.actor.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "total_it": self.total_it,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.actor.load_state_dict(state_dict["actor"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.total_it = state_dict["total_it"]

    def save(self, filename):
        torch.save(self.actor.state_dict(), filename + "_actor.pt")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer.pt")


    def load(self, filename):
        if not torch.cuda.is_available():

            self.actor.load_state_dict(torch.load(filename + "_actor.pt", map_location=torch.device('cpu')))
            self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer.pt", map_location=torch.device('cpu')))

        else:

            self.actor.load_state_dict(torch.load(filename + "_actor.pt"))
            self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer.pt"))
