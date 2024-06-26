import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action, phi=0.05):
		super(Actor, self).__init__()
		self.l1 = nn.Linear(state_dim + action_dim, 400)
		self.l2 = nn.Linear(400, 300)
		self.l3 = nn.Linear(300, action_dim)

		self.max_action = max_action
		self.phi = phi

	def forward(self, state, action):
		a = F.relu(self.l1(torch.cat([state, action], 1)))
		a = F.relu(self.l2(a))
		a = self.phi * self.max_action * torch.tanh(self.l3(a))
		return (a + action).clamp(-self.max_action, self.max_action)
	


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Critic, self).__init__()
		self.l1 = nn.Linear(state_dim + action_dim, 400)
		self.l2 = nn.Linear(400, 300)
		self.l3 = nn.Linear(300, 1)

		self.l4 = nn.Linear(state_dim + action_dim, 400)
		self.l5 = nn.Linear(400, 300)
		self.l6 = nn.Linear(300, 1)

	def forward(self, state, action):
		q1 = F.relu(self.l1(torch.cat([state, action], 1)))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)

		q2 = F.relu(self.l4(torch.cat([state, action], 1)))
		q2 = F.relu(self.l5(q2))
		q2 = self.l6(q2)
		return q1, q2

	def q1(self, state, action):
		q1 = F.relu(self.l1(torch.cat([state, action], 1)))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)
		return q1


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


class BCQ(object):
	def __init__(self, state_dim, action_dim, max_action, device, lr=1e-3, discount=0.99, tau=0.005, lmbda=0.75, phi=0.05):
		latent_dim = action_dim * 2

		self.actor = Actor(state_dim, action_dim, max_action, phi).to(device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

		self.critic = Critic(state_dim, action_dim).to(device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

		self.vae = VAE(state_dim, action_dim, latent_dim, max_action, device).to(device)
		self.vae_optimizer = torch.optim.Adam(self.vae.parameters())

		self.max_action = max_action
		self.action_dim = action_dim
		self.discount = discount
		self.tau = tau
		self.lmbda = lmbda
		self.device = device

	def select_action(self, state, device="cuda"):
		with torch.no_grad():
			state = torch.FloatTensor(state.reshape(1, -1)).repeat(100, 1).to(self.device)
			action = self.actor(state, self.vae.decode(state))
			q1 = self.critic.q1(state, action)
			ind = q1.argmax(0)
		return action[ind].cpu().data.numpy().flatten()
	
	def eval(self):
		self.actor.eval()
		self.vae.eval()
		self.critic.eval()
	
	def train_mode(self):
		self.actor.train()
		self.vae.train()
		self.critic.train()




	def buffer_sample(self, buffer, batch_size):
		data = []
		for item in buffer:
			sub_batch_size = int(batch_size * item[1] + 0.1)
			sub_data = item[0].sample(sub_batch_size)
			data.append(sub_data)
		#state, action, next_state, reward, not_done
		output = []
		for item in zip(*data):
			sub_data = torch.cat(item, axis=0)
			output.append(sub_data)
		return output

	def train(self, buffers, iterations, batch_size=100):
		
		self.actor.train()
		self.vae.train()
		self.critic.train()

		log_dict = {"actor_loss": [], "critic_loss": []}
		for it in range(iterations):
			# Sample replay buffer / batch
			state, action, next_state, reward, not_done, _ = buffers.sample(batch_size)

			# Variational Auto-Encoder Training
			recon, mean, std = self.vae(state, action)
			recon_loss = F.mse_loss(recon, action)
			KL_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
			vae_loss = recon_loss + 0.5 * KL_loss

			self.vae_optimizer.zero_grad()
			vae_loss.backward()
			self.vae_optimizer.step()

			# Critic Training
			with torch.no_grad():
				# Duplicate next state 10 times
				next_state = torch.repeat_interleave(next_state, 10, 0)

				# Compute value of perturbed actions sampled from the VAE
				target_Q1, target_Q2 = self.critic_target(next_state,
				                                          self.actor_target(next_state, self.vae.decode(next_state)))

				# Soft Clipped Double Q-learning
				target_Q = self.lmbda * torch.min(target_Q1, target_Q2) + (1. - self.lmbda) * torch.max(target_Q1,
				                                                                                        target_Q2)
			
				# Take max over each action sampled from the VAE
				target_Q = target_Q.reshape(state.shape[0], -1).max(1)[0].reshape(-1, 1)

				target_Q = reward + not_done * self.discount * target_Q

			current_Q1, current_Q2 = self.critic(state, action)
			critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

			self.critic_optimizer.zero_grad()
			critic_loss.backward()
			self.critic_optimizer.step()

			# Pertubation Model / Action Training
			sampled_actions = self.vae.decode(state)
			perturbed_actions = self.actor(state, sampled_actions)

			# Update through DPG
			actor_loss = -self.critic.q1(state, perturbed_actions).mean()

			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()

			# Update Target Networks
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
			
			log_dict["actor_loss"].append(actor_loss.item())
			log_dict["critic_loss"].append(critic_loss.item())
			
		return log_dict


	def warm_Q(self, buffers, warm_iterations, sample_func, device, upper_Q=333, batch_size=100):
		
		for it in range(warm_iterations):
			# Sample replay buffer / batch
            
			batch, belong_idx = sample_func(buffers, batch_size, return_belong=True) 

			state, action, next_state, _, _, _ = batch

			# Critic Training
			target_Q = torch.Tensor(belong_idx).to(device).reshape(-1, 1)
            # belong_idx = 0 -> expert -> Q = rollout_length + 3
            # belong_idx = 1 -> aug    -> Q = 0
			target_Q = (1 - target_Q) * upper_Q

            # print(belong_idx)
            # print(target_Q)
            
			current_Q1, current_Q2 = self.critic(state, action)
			critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
            
			self.critic_optimizer.zero_grad()
			critic_loss.backward()
			self.critic_optimizer.step()
            
			if it % 1000 == 0:
				print("warm [{}/{}]".format(it, warm_iterations), "critic_loss: ", critic_loss.item())
		

	def save(self, filename):
		torch.save(self.critic.state_dict(), filename + "_critic")
		torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

		torch.save(self.actor.state_dict(), filename + "_actor")
		torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

		torch.save(self.vae.state_dict(), filename + "_vae")
		torch.save(self.vae_optimizer.state_dict(), filename + "_vae_optimizer")

	def load(self, filename):
		if not torch.cuda.is_available():
			self.critic.load_state_dict(torch.load(filename + "_critic", map_location=torch.device('cpu')))
			self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer", map_location=torch.device('cpu')))
			self.critic_target = copy.deepcopy(self.critic)

			self.actor.load_state_dict(torch.load(filename + "_actor", map_location=torch.device('cpu')))
			self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer", map_location=torch.device('cpu')))
			self.actor_target = copy.deepcopy(self.actor)

			self.vae.load_state_dict(torch.load(filename + "_vae", map_location=torch.device('cpu')))
			self.vae_optimizer.load_state_dict(torch.load(filename + "_vae_optimizer", map_location=torch.device('cpu')))
		else:
			self.critic.load_state_dict(torch.load(filename + "_critic"))
			self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
			self.critic_target = copy.deepcopy(self.critic)

			self.actor.load_state_dict(torch.load(filename + "_actor"))
			self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
			self.actor_target = copy.deepcopy(self.actor)

			self.vae.load_state_dict(torch.load(filename + "_vae"))
			self.vae_optimizer.load_state_dict(torch.load(filename + "_vae_optimizer"))

