import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from .network import *

is_with_reward = False

def soft_clamp(x: torch.Tensor, _min=None, _max=None):
    if _max is not None:
        x = _max - F.softplus(_max - x)
    if _min is not None:
        x = _min + F.softplus(x - _min)
    return x

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
    

# class EnsembleLinear(torch.nn.Module):

#     def __init__(self, in_features, out_features, ensemble_size=7):
#         super().__init__()
#         self.ensemble_size = ensemble_size
#         self.register_parameter('weight', torch.nn.Parameter(torch.zeros(ensemble_size, in_features, out_features)))
#         self.register_parameter('bias', torch.nn.Parameter(torch.zeros(ensemble_size, 1, out_features)))
#         torch.nn.init.trunc_normal_(self.weight, std=1 / (2 * in_features ** 0.5))
#         self.select = list(range(0, self.ensemble_size))

#     def forward(self, x):
#         weight = self.weight[self.select]
#         bias = self.bias[self.select]
#         if len(x.shape) == 2:
#             x = torch.einsum('ij,bjk->bik', x, weight)
#         else:
#             x = torch.einsum('bij,bjk->bik', x, weight)
#         x = x + bias

#         return x

#     def set_select(self, indexes):
#         assert len(indexes) <= self.ensemble_size and max(indexes) < self.ensemble_size
#         self.select = indexes


# class EnsembleTransition(torch.nn.Module):
#     def __init__(self, obs_dim, action_dim, hidden_features, hidden_layers, ensemble_size=7, mode='local',
#                  with_reward=is_with_reward):
#         super().__init__()
#         self.obs_dim = obs_dim
#         self.mode = mode
#         self.with_reward = with_reward
#         self.ensemble_size = ensemble_size
#         self.activation = Swish()
#         module_list = []
#         for i in range(hidden_layers):
#             if i == 0:
#                 module_list.append(EnsembleLinear(obs_dim + action_dim, hidden_features, ensemble_size))
#             else:
#                 module_list.append(EnsembleLinear(hidden_features, hidden_features, ensemble_size))
#         self.backbones = torch.nn.ModuleList(module_list)
#         self.output_layer = EnsembleLinear(hidden_features, 2 * (obs_dim + self.with_reward), ensemble_size)
#         self.register_parameter('max_logstd', torch.nn.Parameter(torch.ones(obs_dim + self.with_reward) * torch.tensor(1), requires_grad=True))
#         self.register_parameter('min_logstd', torch.nn.Parameter(torch.ones(obs_dim + self.with_reward) * torch.tensor(-5), requires_grad=True))

#     def forward(self, state, action):
#         obs_action = torch.cat([state,action],dim=-1)
#         # print('state')
#         output = obs_action
#         for layer in self.backbones:
#             output = self.activation(layer(output))
#         mu, logstd = torch.chunk(self.output_layer(output), 2, dim=-1)
#         logstd = soft_clamp(logstd, self.min_logstd, self.max_logstd)
#         # print(f"min_logstd:[{self.min_logstd}], max_logstd:[{self.max_logstd}]")
#         if self.mode == 'local': # next_obs = obs + delta(from model)
#             if self.with_reward:
#                 obs, reward = torch.split(mu, [self.obs_dim, 1], dim=-1)
#                 obs = obs + state
#                 mu = torch.cat([obs, reward], dim=-1)
#             else:
#                 mu = mu + state
#         return torch.distributions.Normal(mu, torch.exp(logstd))

#     def train(self, replay_buffer, batch_size=256, epoch=100):

#         valid_size = min(1000,int(replay_buffer.size * 0.2))
#         train_size = replay_buffer.size - valid_size
#         train_part, valid_part = torch.utils.data.random_split(range(replay_buffer.size), (train_size, valid_size))
#         iteration = int(train_size / batch_size) + int(train_size % batch_size != 0)
        
#         best_val_loss = [float('inf')] * self.ensemble_size
#         best_epoch = [0] * self.ensemble_size
#         best_model = [None]*self.ensemble_size
#         for epoch_num in range(epoch):
#             idxs = torch.randint(low=0,high=train_size, size=(self.ensemble_size, train_size))
#             pbar = tqdm(range(iteration), desc='Train Dynamic Model',ncols=150)
#             total_loss = 0
#             self.backbones.train()
#             for batch_num in pbar:
#                 l = batch_num * batch_size
#                 r = min((batch_num+1)*batch_size,train_size)
#                 if(l >= r):break
#                 idx = torch.Tensor(train_part.indices)[:,idxs[:,l:r]]
#                 state = torch.FloatTensor(replay_buffer.state[idx]).to(self.device)
#                 action= torch.FloatTensor(replay_buffer.action[idx]).to(self.device)
#                 next_state=torch.FloatTensor(replay_buffer.next_state[idx]).to(self.device)
#                 reward= torch.FloatTensor(replay_buffer.reward[idx]).to(self.device)
#                 dist = self.forward(state, action)

#                 if(self.use_reward):
#                     ground_truth = torch.cat([next_state,reward],dim=-1)
#                 else:
#                     ground_truth = next_state
#                 ground_truth = torch.cat([ground_truth.unsqueeze(0) for i in range(self.ensemble_size)]
#                                          ,dim=0)
                
#                 loss = -dist.log_prob(ground_truth)
#                 loss = loss.mean() + 0.01*self.max_logstd.mean() - 0.01*self.min_logstd.mean()
                
#                 self.optim.zero_grad()
#                 loss.backward()
#                 self.optim.step()
#                 total_loss += loss.item()
#                 if(batch_num % 100 == 0):
#                     pbar.set_postfix({"Epoch":epoch_num,"train loss":loss})
#             total_loss /= iteration
            
            
            
#             self.models.eval()
#             with torch.no_grad():
#                 valid_state = torch.FloatTensor(replay_buffer.state[valid_part.indices]).to(self.device)
#                 valid_state = torch.cat([valid_state.unsqueeze(0) for i in range(self.ensemble_size)]
#                                          ,dim=0)
#                 valid_action= torch.FloatTensor(replay_buffer.action[valid_part.indices]).to(self.device)
#                 valid_action = torch.cat([valid_action.unsqueeze(0) for i in range(self.ensemble_size)]
#                                          ,dim=0)
#                 valid_next_state=torch.FloatTensor(replay_buffer.next_state[valid_part.indices]).to(self.device)
#                 valid_reward=torch.FloatTensor(replay_buffer.reward[valid_part.indices]).to(self.device)
#                 valid_dist = self.forward(valid_state,valid_action)
#                 if(self.use_reward):
#                     ground_truth = torch.cat([valid_next_state,valid_reward],dim=-1)
#                 else:
#                     ground_truth = valid_next_state
#                 ground_truth = torch.cat([ground_truth.unsqueeze(0) for i in range(self.ensemble_size)]
#                                          ,dim=0)
#                 valid_loss = ((valid_dist.mean - ground_truth)**2).mean(dim=(1,2))
#                 valid_loss = valid_loss.cpu().numpy()
#             # print('valid_loss.shape : ',valid_loss.shape)
#             for i in range(self.ensemble_size):
#                 if(valid_loss[i] < best_val_loss[i]):
#                     best_val_loss[i] = valid_loss[i]
#                     best_epoch[i] = epoch_num+1
#                     best_model[i] = self.models[i].state_dict()
#             print(f"Epoch [{epoch_num+1}/{epoch}] train_loss:{total_loss}, valid_loss:{valid_loss.mean()}")
#         for model, best_params in zip(self.models, best_model):
#             model.load_state_dict(best_params)
            
#         print(">>> Finsh Dynamic Model Training.")
#         print(f">>> best epoch:[{best_epoch}]")
#         print(f">>> best valid loss:[{best_val_loss}]")   
#         select_index = self._select_model(best_val_loss)
#         print(f">>> select model:[{select_index}]")   
        
#         # test
#         # valid_state = torch.Tensor(replay_buffer.state[-1]).to(self.device)
#         # valid_action = torch.Tensor(replay_buffer.action[-1]).to(self.device)
#         # valid_next_state = torch.Tensor(replay_buffer.next_state[-1]).to(self.device)
#         # valid_dist = self.forward(valid_state, valid_action)
#         # print(valid_dist.mean)
#         # print(valid_next_state)
        
        
#         # for params in best_model:
#         #     print(" MODEL: ",params)
#         # print('==================')
#         # for name, param in self.models.named_parameters():
#         #     if param.requires_grad:
#         #         print(name, param.data)


#     def set_select(self, indexes):
#         # print("indexes :",indexes)
#         for layer in self.backbones:
#             layer.set_select(indexes)
#         self.output_layer.set_select(indexes)



class EnsembleDynamicModel(nn.Module): # mean
    def __init__(self, 
                 state_dim, 
                 action_dim, 
                 hidden_size=[256,256,256,128],
                 ensemble_size=7,
                 select_size=5,
                 use_reward=False,
                 learning_rate=1e-3,
                 device='cpu'):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim=action_dim
        self.hidden_size=hidden_size
        self.ensemble_size=ensemble_size
        self.select_size=select_size
        self.select_model = list(range(ensemble_size))
        
        self.use_reward = use_reward
        self.device=device
        if(use_reward):
            self.models = [CustomMultiHeadMLP(input_dim=state_dim+action_dim,
                                          output_dim=[state_dim+1, state_dim+1],
                                          hidden_size=hidden_size,
                                          activation=Swish()) for i in range(ensemble_size)]
        else:
            self.models = [CustomMultiHeadMLP(input_dim=state_dim+action_dim,
                                              output_dim=[state_dim, state_dim],
                                              hidden_size=hidden_size,
                                              activation=Swish()) for i in range(ensemble_size)]
        self.models = nn.ModuleList(self.models).to(device)
        self.optim = torch.optim.Adam(self.models.parameters(),lr=learning_rate,weight_decay=0.000075)

        # 声明参数并将其放置在 CUDA 设备上
        max_logstd = torch.ones(state_dim + use_reward).fill_(1).to(device)
        min_logstd = torch.ones(state_dim + use_reward).fill_(-5).to(device)

        # 将参数注册到模型中
        self.register_parameter('max_logstd', nn.Parameter(max_logstd, requires_grad=True))
        self.register_parameter('min_logstd', nn.Parameter(min_logstd, requires_grad=True))
        

   
    def forward(self, state, action):
        if(len(state.shape) == 2):# used for predict
            state = torch.cat([state.unsqueeze(0) for i in range(self.ensemble_size)], dim=0) 
            action= torch.cat([action.unsqueeze(0) for i in range(self.ensemble_size)], dim=0) 
        x = torch.cat([state,action],dim=-1)
        mus,sigmas = [],[]
        for id,model in enumerate(self.models):
            mu,log_sigma = model(x[id])  # 执行每个模型的前向传播
            log_sigma = soft_clamp(log_sigma, self.min_logstd, self.max_logstd)
            sigma = torch.exp(log_sigma)
            
            if self.use_reward: # next_state = state + model(state,action)
                obs, reward = torch.split(mu, [self.state_dim, 1], dim=-1)
                obs = obs + state[id]
                mu = torch.cat([obs, reward], dim=-1)
            else:
                mu = mu + state[id]

            mus.append(mu)
            sigmas.append(sigma)
            # dist = torch.distributions.Normal(mu, sigma)
            # dists.append(dist)
        mus = torch.stack(mus,dim=0)[self.select_model]
        sigmas=torch.stack(sigmas,dim=0)[self.select_model]
        outputs = torch.distributions.Normal(mus,sigmas)
        return outputs      # 返回形状为 (ensemble_size, batch_size, dim1) 的张量
    
    
    def _select_model(self, matric):
        sorted_indices = sorted(range(len(matric)), key=lambda i: matric[i])
        self.select_model = sorted_indices[:self.select_size]
        return self.select_model

    
    def train(self, replay_buffer, batch_size=256, epoch=100):

        valid_size = min(1000,int(replay_buffer.size * 0.2))
        train_size = replay_buffer.size - valid_size
        train_part, valid_part = torch.utils.data.random_split(range(replay_buffer.size), (train_size, valid_size))
        train_idx = torch.tensor(train_part.indices, dtype=torch.int32)
        valid_idx = torch.tensor(valid_part.indices, dtype=torch.int32)
        iteration = int(train_size / batch_size) + int(train_size % batch_size != 0)
        
        best_val_loss = [float('inf')] * self.ensemble_size
        best_epoch = [0] * self.ensemble_size
        best_model = [None]*self.ensemble_size
        for epoch_num in range(epoch):
            
            all_index = torch.tensor(list(range(train_size))).repeat((self.ensemble_size,1))
            for i in range(self.ensemble_size):
                all_index[i] = all_index[i][torch.randperm(train_size)]
            
            # pbar = tqdm(range(iteration), desc='Train Dynamic Model',ncols=150)
            total_loss = 0
            self.models.train()

            # for batch_num in pbar:
            for batch_num in range(iteration): 

                l = batch_num * batch_size
                r = min((batch_num+1)*batch_size,train_size)
                if(l >= r):break
                idx = train_idx[all_index[:,l:r]]
                state = torch.FloatTensor(replay_buffer.state[idx]).to(self.device)
                action= torch.FloatTensor(replay_buffer.action[idx]).to(self.device)
                next_state=torch.FloatTensor(replay_buffer.next_state[idx]).to(self.device)
                reward= torch.FloatTensor(replay_buffer.reward[idx]).to(self.device)
                dist = self.forward(state, action)

                if(self.use_reward):
                    ground_truth = torch.cat([next_state,reward],dim=-1)
                else:
                    ground_truth = next_state
                
                loss = -dist.log_prob(ground_truth)
                loss = loss.mean() + 0.01*self.max_logstd.mean() - 0.01*self.min_logstd.mean()
                
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                total_loss += loss.item()
                
                # if(batch_num % 100 == 0):
                #     pbar.set_postfix({"Epoch":epoch_num,"train loss":loss})

            total_loss /= iteration
            
            
            
            self.models.eval()
            with torch.no_grad():
                valid_state = torch.FloatTensor(replay_buffer.state[valid_idx]).to(self.device)
                valid_action= torch.FloatTensor(replay_buffer.action[valid_idx]).to(self.device)
                valid_next_state=torch.FloatTensor(replay_buffer.next_state[valid_idx]).to(self.device)
                valid_reward=torch.FloatTensor(replay_buffer.reward[valid_idx]).to(self.device)
                valid_dist = self.forward(valid_state,valid_action)
                if(self.use_reward):
                    ground_truth = torch.cat([valid_next_state,valid_reward],dim=-1)
                else:
                    ground_truth = valid_next_state
                ground_truth = torch.cat([ground_truth.unsqueeze(0) for i in range(self.ensemble_size)]
                                         ,dim=0)
                valid_loss = ((valid_dist.mean - ground_truth)**2).mean(dim=(1,2))
                valid_loss = valid_loss.cpu().numpy()
            # print('valid_loss.shape : ',valid_loss.shape)
            for i in range(self.ensemble_size):
                if(valid_loss[i] < best_val_loss[i]):
                    best_val_loss[i] = valid_loss[i]
                    best_epoch[i] = epoch_num+1
                    best_model[i] = self.models[i].state_dict()
            print(f"Epoch [{epoch_num+1}/{epoch}] train_loss:{total_loss}, valid_loss:{valid_loss.mean()}")
        for model, best_params in zip(self.models, best_model):
            model.load_state_dict(best_params)
            
        print(f">>> Finsh Dynamic Model Training.")
        print(f">>> best epoch:[{best_epoch}]")
        print(f">>> best valid loss:[{best_val_loss}]")   
        select_index = self._select_model(best_val_loss)
        print(f">>> select model:[{select_index}]")   
        
        # test
        # valid_state = torch.Tensor(replay_buffer.state[-1]).to(self.device)
        # valid_action = torch.Tensor(replay_buffer.action[-1]).to(self.device)
        # valid_next_state = torch.Tensor(replay_buffer.next_state[-1]).to(self.device)
        # valid_dist = self.forward(valid_state, valid_action)
        # print(valid_dist.mean)
        # print(valid_next_state)
        
        
        # for params in best_model:
        #     print(" MODEL: ",params)
        # print('==================')
        # for name, param in self.models.named_parameters():
        #     if param.requires_grad:
        #         print(name, param.data)
        
    def set_select_model(self, replay_buffer):
        
        print(">>> dynamic model select_num : ",self.select_size)
        
        self.models.eval()
        with torch.no_grad():
            state = torch.FloatTensor(replay_buffer.state).to(self.device)
            action= torch.FloatTensor(replay_buffer.action).to(self.device)
            next_state=torch.FloatTensor(replay_buffer.next_state).to(self.device)
            reward=torch.FloatTensor(replay_buffer.reward).to(self.device)
            dist = self.forward(state,action)
            if(self.use_reward):
                ground_truth = torch.cat([next_state,reward],dim=-1)
            else:
                ground_truth = next_state
            ground_truth = torch.cat([ground_truth.unsqueeze(0) for i in range(self.ensemble_size)]
                                        ,dim=0)
            loss = ((dist.mean - ground_truth)**2).mean(dim=(1,2))
            loss = loss.cpu().numpy()
        # print('loss.shape : ',loss.shape)
        select_index = self._select_model(loss)
        print(f">>> select model:[{select_index}]")   
        
        
    
    
    def save(self, filename):
        torch.save(self.models.state_dict(), filename + "_dynamicmodel")
        torch.save(self.optim.state_dict(), filename + "_dynamicmodel_optim")


    def load(self, filename):
        if not torch.cuda.is_available():
            self.models.load_state_dict(torch.load(filename + "_dynamicmodel", map_location=torch.device('cpu')))
            self.optim.load_state_dict(torch.load(filename + "_dynamicmodel_optim", map_location=torch.device('cpu')))

        else:
            self.models.load_state_dict(torch.load(filename + "_dynamicmodel"))
            self.optim.load_state_dict(torch.load(filename + "_dynamicmodel_optim"))
            
         