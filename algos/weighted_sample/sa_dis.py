

import copy
import os
import random
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

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


class Discriminator(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Discriminator, self).__init__()

        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        d = F.relu(self.fc1(sa))
        d = F.relu(self.fc2(d))
        d = F.sigmoid(self.fc3(d))
        d = torch.clip(d, 0.1, 0.9)
        return d
    

class SADiscriminator:
    def __init__(self, 
                 state_dim: int, 
                 action_dim: int, 
                 lr: float = 1e-4, 
                 device: str = "cpu",
                 ):
        

        self.discriminator = Discriminator(state_dim, action_dim).to(device)
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=lr)


        self.loss_func= (lambda x: F.sigmoid(-x))

        self.device = device

        self.beta = 0

        self.total_it = 0

    def train(self, buffer_e, buffer_o, iterations=1, batch_size=256):
        self.discriminator.train()

        log_dict = {"dis_loss": [], "accuracy": []}

        for iter_i in range(iterations):
            self.total_it += 1

            state_e, action_e, next_state_e, reward_e, not_done_e, _ = buffer_e.sample(batch_size)

            state_o, action_o, next_state_o, reward_o, not_done_o, _ = buffer_o.sample(batch_size)


            d_e = self.discriminator(state_e, action_e)
            d_o = self.discriminator(state_o, action_o)


            d_loss_e = -torch.log(d_e)
            d_loss_o = -torch.log(1 - d_o)
            d_loss = torch.mean(d_loss_e) + torch.mean(d_loss_o)


            self.discriminator_optimizer.zero_grad()
            d_loss.backward()
            self.discriminator_optimizer.step()

            log_dict["dis_loss"].append(d_loss.item())

            output = torch.cat([d_e, d_o], dim=0)
            pseudo_label = (output - 0.5).detach().reshape(-1)
            pseudo_label[pseudo_label > 0] = 1.
            pseudo_label[pseudo_label < 1] = 0.

            # print(output[:10], output[-10:])
            # print(pseudo_label.shape, positive_target.shape)

            target = torch.zeros_like(pseudo_label).to(self.device)
            target[:batch_size] = 1
            accuracy = (pseudo_label == target).sum() / len(pseudo_label)
            log_dict["accuracy"].append(accuracy.item())


            # print("pseudo_accuracy: ", pseudo_accuracy)

            # if self.total_it > 10:
            #     break


        return log_dict
    

    def save(self, filename):
        torch.save(self.discriminator.state_dict(), filename + "_discriminator")
        torch.save(self.discriminator_optimizer.state_dict(), filename + "_discriminator_optimizer")


    def load(self, filename):
        self.discriminator.load_state_dict(torch.load(filename + "_discriminator"))
        self.discriminator_optimizer.load_state_dict(torch.load(filename + "_discriminator_optimizer"))
    

    def weight_sample(self, state, action):

        self.discriminator.eval()

        pred_prob = self.discriminator(state, action)

        weight = pred_prob / (1 - pred_prob)

        # scale to [0,1]
        weight = weight / 10

        return weight
