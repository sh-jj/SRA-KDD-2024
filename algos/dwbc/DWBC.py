import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import TanhTransform

MEAN_MIN = -9.0
MEAN_MAX = 9.0
LOG_STD_MIN = -5
LOG_STD_MAX = 2
LOG_PI_NORM_MAX = 10
LOG_PI_NORM_MIN = -20

EPS = 1e-7

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()

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
    
    

    def select_action(self, state, device):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        _, _, action = self(state)
        return action.cpu().data.numpy().flatten()
    
    @torch.no_grad()
    def select_action_with_prob(self, state, device):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        _, _, action = self.forward(state)
        prob = torch.sum(self.get_log_density(state, action),dim=-1)
        
        return action.cpu().data.numpy().flatten(), prob.cpu().data.numpy().flatten()
    # def act(self, state):
    #     a_dist, a_tanh_mode = self._get_outputs(state)
    #     action = a_tanh_mode
    #     logp_pi = a_dist.log_prob(action).sum(axis=-1)
    #     return action, logp_pi

class Discriminator(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Discriminator, self).__init__()

        self.fc1_1 = nn.Linear(state_dim + action_dim, 128)
        self.fc1_2 = nn.Linear(action_dim, 128)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, state, action, log_pi):
        sa = torch.cat([state, action], 1)
        d1 = F.relu(self.fc1_1(sa))
        d2 = F.relu(self.fc1_2(log_pi))
        d = torch.cat([d1, d2], 1)
        d = F.relu(self.fc2(d))
        d = F.sigmoid(self.fc3(d))
        d = torch.clip(d, 0.1, 0.9)
        return d


class  DWBC(object):
    def __init__(
            self,
            state_dim,
            action_dim,
            alpha=7.5,
            no_pu=False,
            eta=0.5,
            d_update_num=100,
            learning_rate=1e-4,
            device:str="cpu"
    ):
        self.device = device
        self.policy = Actor(state_dim, action_dim).to(device)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate, weight_decay=0.005)

        self.discriminator = Discriminator(state_dim, action_dim).to(device)
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=learning_rate)

        self.alpha = alpha
        self.no_pu_learning = no_pu
        self.eta = eta
        self.d_update_num = d_update_num

        self.total_it = 0

    def select_action(self, state, device):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        _, _, action = self.policy(state)
        return action.cpu().data.numpy().flatten()

    def _train(self, buffer_e, buffer_o, ablation):
        self.total_it += 1
        log_dict = {}
        # Sample from D_e and D_o
        state_e, action_e, _, _, _, _ = buffer_e
        state_o, action_o, _, _, _, _ = buffer_o
        log_pi_e = self.policy.get_log_density(state_e, action_e)
        log_pi_o = self.policy.get_log_density(state_o, action_o)

        # Compute discriminator loss
        log_pi_e_clip = torch.clip(log_pi_e, LOG_PI_NORM_MIN, LOG_PI_NORM_MAX)
        log_pi_o_clip = torch.clip(log_pi_o, LOG_PI_NORM_MIN, LOG_PI_NORM_MAX)
        log_pi_e_norm = (log_pi_e_clip - LOG_PI_NORM_MIN) / (LOG_PI_NORM_MAX - LOG_PI_NORM_MIN)
        log_pi_o_norm = (log_pi_o_clip - LOG_PI_NORM_MIN) / (LOG_PI_NORM_MAX - LOG_PI_NORM_MIN)
        d_e = self.discriminator(state_e, action_e, log_pi_e_norm.detach())
        d_o = self.discriminator(state_o, action_o, log_pi_o_norm.detach())

        if self.no_pu_learning:
            d_loss_e = -torch.log(d_e)
            d_loss_o = -torch.log(1 - d_o)
            d_loss = torch.mean(d_loss_e + d_loss_o)
        else:
            d_loss_e = -torch.log(d_e)
            d_loss_o = -torch.log(1 - d_o) / self.eta + torch.log(1 - d_e)
            d_loss = torch.mean(d_loss_e + d_loss_o)
        log_dict['discriminator_loss'] = d_loss.item()

        # Optimize the discriminator
        if self.total_it % self.d_update_num == 0:
            self.discriminator_optimizer.zero_grad()
            d_loss.backward()
            self.discriminator_optimizer.step()

        # Compute policy loss
        d_e_clip = torch.squeeze(d_e).detach()
        d_o_clip = torch.squeeze(d_o).detach()
        d_o_clip[d_o_clip < 0.5] = 0.0

        bc_loss = -torch.sum(log_pi_e, 1)
        corr_loss_e = -torch.sum(log_pi_e, 1) * (self.eta / (d_e_clip * (1.0 - d_e_clip)) + 1.0)
        corr_loss_o = -torch.sum(log_pi_o, 1) * (1.0 / (1.0 - d_o_clip) - 1.0)
        
        if(ablation):
            p_loss = torch.mean(bc_loss)
        else :
            p_loss = self.alpha * torch.mean(bc_loss) - torch.mean(corr_loss_e) + torch.mean(corr_loss_o)

        log_dict['policy_loss'] = p_loss.item()
        # Optimize the policy
        self.policy_optimizer.zero_grad()
        p_loss.backward()
        self.policy_optimizer.step()
        return log_dict

    def save(self, filename):
        torch.save(self.discriminator.state_dict(), filename + "_discriminator")
        torch.save(self.discriminator_optimizer.state_dict(), filename + "_discriminator_optimizer")

        torch.save(self.policy.state_dict(), filename + "_policy")
        torch.save(self.policy_optimizer.state_dict(), filename + "_policy_optimizer")

    def load(self, filename):
        self.discriminator.load_state_dict(torch.load(filename + "_discriminator"))
        self.discriminator_optimizer.load_state_dict(torch.load(filename + "_discriminator_optimizer"))

        self.policy.load_state_dict(torch.load(filename + "_policy"))
        self.policy_optimizer.load_state_dict(torch.load(filename + "_policy_optimizer"))
    
    def eval(self):
        self.policy.eval()
        self.discriminator.eval()
    

    def train(self, buffer_expert=None, buffer_offline=None, iters=1, batch_size=100,ablation=False):
        self.policy.train()
        self.discriminator.train()

        if(buffer_expert is None and buffer_offline is None):
            return
        if(buffer_expert is None or buffer_offline is None):
            raise ValueError("Both buffer_expert and buffer_offline should be either None of not None")

        log_dict = {"policy_loss": [],
                    "discriminator_loss":[]}
        for i in range(iters):
    

            batch_e = buffer_expert.sample(batch_size)
            batch_o = buffer_offline.sample(batch_size)

            batch_e = [b.to(self.device) for b in batch_e]
            batch_o = [b.to(self.device) for b in batch_o]

            log_dict_ = self._train(buffer_e=batch_e, buffer_o=batch_o,ablation=ablation)


            log_dict["policy_loss"].append(log_dict_["policy_loss"])
            log_dict["discriminator_loss"].append(log_dict_["discriminator_loss"])


        return log_dict
