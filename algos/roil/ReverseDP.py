import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F




class Value(nn.Module):
	def __init__(self, state_dim):
		super(Value, self).__init__()
		self.l1 = nn.Linear(state_dim, 400)
		self.l2 = nn.Linear(400, 300)
		self.l3 = nn.Linear(300, 1)

	def forward(self, state):
		q1 = F.relu(self.l1(state))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)

		return q1


class ValueNetwork(object):
	def __init__(self, state_dim, device, discount=0.99, tau=0.005, lmbda=0.75, phi=0.05):

		self.value = Value(state_dim).to(device)
		self.value_target = copy.deepcopy(self.value)
		self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=1e-3)

		self.discount = discount
		self.tau = tau
		self.lmbda = lmbda
		self.device = device

	def buffer_sample(self, buffer, batch_size):
		state_col, value_col = [], []
		for item in buffer:
			sub_batch_size = int(batch_size * item[1] + 0.1)

			state_i, value_i = item[0]

			rand_index = torch.randperm(state_i.size(0))


			state_col.append(state_i[rand_index[: sub_batch_size]])
			value_col.append(value_i[rand_index[: sub_batch_size]])

			# print(state_i.shape, value_i.shape)
			# print(state_col[-1].shape)
			# print(value_col[-1].shape)

		state_col = torch.cat(state_col, axis=0)
		value_col = torch.cat(value_col, axis=0)

		return state_col, value_col
	

	def train(self, buffer, iterations, batch_size=100):
		

		log_dict = {"value_loss": []}
		for it in range(iterations):
			# Sample replay buffer / batch
			state, value = self.buffer_sample(buffer, batch_size)

			state, value = state.to(self.device), value.to(self.device)

			# Value Training

			target_V = value

			current_V = self.value(state)

			value_loss = F.mse_loss(current_V, target_V)

			self.value_optimizer.zero_grad()
			value_loss.backward()
			self.value_optimizer.step()


			log_dict["value_loss"].append(value_loss.item())

			
		return log_dict

	@torch.no_grad()
	def test(self, state, value, batch_size=100):

		tl = 0

		log_dict = {"value_loss": []}
		while tl < state.shape[0]:

			tr = min(tl + batch_size, state.shape[0])

			state_i, value_i = state[tl: tr].to(self.device), value[tl: tr].to(self.device)


			target_V = value_i
			current_V = self.value(state_i)
			value_loss = F.mse_loss(current_V, target_V)
			
			
			log_dict["value_loss"].append(value_loss.item())
			tl += batch_size

		return log_dict

	@torch.no_grad()
	def calc_value(self, state):
		return self.value(state)

	def warm_Q(self, buffers, warm_iterations, sample_func, device, upper_Q=333, batch_size=100):
		
		for it in range(warm_iterations):
			# Sample replay buffer / batch
            
			batch, belong_idx = sample_func(buffers, batch_size, return_belong=True) 

			state, action, next_state, _, _, _ = batch

			# value Training
			target_Q = torch.Tensor(belong_idx).to(device).reshape(-1, 1)
            # belong_idx = 0 -> expert -> Q = rollout_length + 3
            # belong_idx = 1 -> aug    -> Q = 0
			target_Q = (1 - target_Q) * upper_Q

            # print(belong_idx)
            # print(target_Q)
            
			current_Q1, current_Q2 = self.value(state, action)
			value_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
            
			self.value_optimizer.zero_grad()
			value_loss.backward()
			self.value_optimizer.step()
            
			if it % 1000 == 0:
				print("warm [{}/{}]".format(it, warm_iterations), "value_loss: ", value_loss.item())
		

	def save(self, filename):
		torch.save(self.value.state_dict(), filename + "_value")
		torch.save(self.value_optimizer.state_dict(), filename + "_value_optimizer")

		torch.save(self.actor.state_dict(), filename + "_actor")
		torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

		torch.save(self.vae.state_dict(), filename + "_vae")
		torch.save(self.vae_optimizer.state_dict(), filename + "_vae_optimizer")

	def load(self, filename):
		if not torch.cuda.is_available():
			self.value.load_state_dict(torch.load(filename + "_value", map_location=torch.device('cpu')))
			self.value_optimizer.load_state_dict(torch.load(filename + "_value_optimizer", map_location=torch.device('cpu')))
			self.value_target = copy.deepcopy(self.value)

		else:
			self.value.load_state_dict(torch.load(filename + "_value"))
			self.value_optimizer.load_state_dict(torch.load(filename + "_value_optimizer"))
			self.value_target = copy.deepcopy(self.value)


