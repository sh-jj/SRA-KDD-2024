import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



# Vanilla Variational Auto-Encoder 
class VAE(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim, max_action, device):
        super(VAE, self).__init__()
        self.e1 = nn.Linear(state_dim + action_dim, 750)
        self.e2 = nn.Linear(750, 750)

        self.mean = nn.Linear(750, latent_dim)
        self.log_std = nn.Linear(750, latent_dim)

        self.d1 = nn.Linear(state_dim + latent_dim, 750)
        self.d2 = nn.Linear(750, 750)
        self.d3 = nn.Linear(750, action_dim)

        self.max_action = max_action
        self.latent_dim = latent_dim
        self.device = device

    def forward(self, state, action):
        z = F.relu(self.e1(torch.cat([state, action], 1)))
        z = F.relu(self.e2(z))

        mean = self.mean(z)
        # Clamped for numerical stability
        log_std = self.log_std(z).clamp(-4, 15)
        std = torch.exp(log_std)
        z = mean + std * torch.randn_like(std)

        u = self.decode(state, z)

        return u, mean, std

    def decode(self, state, z=None):
        # When sampling from the VAE, the latent vector is clipped to [-0.5, 0.5]
        if z is None:
            z = torch.randn((state.shape[0], self.latent_dim)).to(self.device).clamp(-0.5, 0.5)

        a = F.relu(self.d1(torch.cat([state, z], 1)))
        a = F.relu(self.d2(a))
        return self.max_action * torch.tanh(self.d3(a))


class ReverseBC:
    def __init__(self, state_dim, action_dim, max_action, device, entropy_weight=0.5):
        latent_dim = action_dim * 2

        self.vae = VAE(state_dim, action_dim, latent_dim, max_action, device).to(device)
        self.vae_optimizer = torch.optim.Adam(self.vae.parameters())

        self.max_action = max_action
        self.action_dim = action_dim
        self.device = device

        self.entropy_weight = entropy_weight

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            action = self.vae.decode(state)
        return action.cpu().data.numpy()

    def train(self, replay_buffer, iterations, batch_size=100):
    
        loss_sum = 0
        sample_count = 0
        for it in range(iterations):
            # Sample replay buffer / batch
            state, action, next_state, reward, weight, not_done = replay_buffer.sample(batch_size)

            # Variational Auto-Encoder Training
            recon, mean, std = self.vae(next_state, action)
            recon_loss = F.mse_loss(recon, action)
            KL_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
            vae_loss = recon_loss + self.entropy_weight * KL_loss

            self.vae_optimizer.zero_grad()
            vae_loss.backward()
            self.vae_optimizer.step()
            
            loss_sum += vae_loss.item()
            sample_count += batch_size

        return loss_sum / sample_count

    def save(self, filename):
        torch.save(self.vae.state_dict(), filename)
        # torch.save(self.vae.state_dict(), filename + "_RBC_vae.pt")
        # torch.save(self.vae_optimizer.state_dict(), filename + "_RBC_vae_optimizer.pt")

    def load(self, filename):
        if not torch.cuda.is_available():
            self.vae.load_state_dict(torch.load(filename, map_location=torch.device('cpu')))
            # self.vae.load_state_dict(torch.load(filename + "_RBC_vae.pt", map_location=torch.device('cpu')))
            # self.vae_optimizer.load_state_dict(torch.load(filename + "_RBC_vae_optimizer.pt", map_location=torch.device('cpu')))
        else:
            self.vae.load_state_dict(torch.load(filename))
            # self.vae.load_state_dict(torch.load(filename + "_RBC_vae.pt"))
            # self.vae_optimizer.load_state_dict(torch.load(filename + "_RBC_vae_optimizer.pt"))
