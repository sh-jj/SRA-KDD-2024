import torch
import torch.nn as nn

import numpy as np

from .network import *
import copy
from utils import ReplayBuffer




def update_log(keywords, value, log_dict):
    if(keywords not in log_dict):
        log_dict[keywords] = []
    log_dict[keywords].append(value)

def hard_update(target: nn.Module, source: nn.Module):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)

        
class SAC:
    def __init__(
        self,
        state_dim: int, 
        action_dim: int, 
        max_action: float, 
        lr=3e-4, 
        wd=5e-3,
        discount: float = 0.99,
        tau: float = 0.005, # soft update parameter
        
        entropy_coeff: float=0.2,
        target_update_interval:int=1,
        automatic_entropy_tuning:bool=True,
        
        device: str = "cpu",
    ):
        # print('>> sac device : ',device)
        # actor = GaussianActor(state_dim, action_dim, max_action).to(device)
        # actor_optimizer = torch.optim.Adam(actor.parameters(), lr=lr, weight_decay=wd)

        # critic_1 = CriticNetwork(state_dim=state_dim, action_dim=action_dim).to(device)
        # critic_1_optimizer = torch.optim.Adam(critic_1.parameters(), lr=lr)
        # critic_2 = CriticNetwork(state_dim=state_dim, action_dim=action_dim).to(device)
        # critic_2_optimizer = torch.optim.Adam(critic_2.parameters(), lr=lr)

        self.actor = GaussianActor(state_dim, action_dim, max_action).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr, weight_decay=wd)
        self.critic1 = CriticNetwork(state_dim=state_dim, action_dim=action_dim).to(device)
        self.critic1_target = CriticNetwork(state_dim=state_dim, action_dim=action_dim).to(device)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=lr)
        hard_update(self.critic1_target, self.critic1)
        
        self.critic2 = CriticNetwork(state_dim=state_dim, action_dim=action_dim).to(device)
        self.critic2_target = CriticNetwork(state_dim=state_dim, action_dim=action_dim).to(device)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=lr)
        hard_update(self.critic2_target, self.critic2)
        

        self.max_action = max_action
        self.discount = discount # gamma
        self.tau = tau
        self.device = device    

        self.alpha = entropy_coeff # alpha in paper, entropy coefficient
        self.target_update_interval = target_update_interval
        self.automatic_entropy_tuning = automatic_entropy_tuning
        if self.automatic_entropy_tuning is True:
            self.target_entropy = -torch.prod(torch.Tensor([action_dim]).to(self.device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=lr)

        self.total_it = 0
    
    def update_parameters(self, batch, expert_batch=None, regularization_weight=None, log_dict=None):
        self.critic1.train()
        self.critic2.train()
        self.actor.train()
        state,action,next_state,reward,not_done = batch
        
        
        # print('policy_rewrd.mean : ',reward.mean())
        with torch.no_grad():
            next_action, next_action_logpi, _ = self.actor(next_state)
            # print(next_action.mean())
            q1_next_target = self.critic1_target(next_state,next_action)
            q2_next_target = self.critic2_target(next_state,next_action)
            min_q_next_target = torch.min(q1_next_target,q2_next_target) - self.alpha * next_action_logpi
            next_q_value = reward + (not_done * self.discount * min_q_next_target)
        q1 = self.critic1(state, action)
        q2 = self.critic2(state, action)
        update_log(keywords='q1',value=q1.mean().item(),log_dict=log_dict)
        update_log(keywords='q2',value=q2.mean().item(),log_dict=log_dict)
        update_log(keywords='next_q_value',value=next_q_value.mean().item(),log_dict=log_dict)
        q1_loss = F.mse_loss(q1, next_q_value)
        q2_loss = F.mse_loss(q2, next_q_value)
        q_loss = q1_loss + q2_loss
        # print(f'q1 : {q1.mean().item()}, q2 : {q2.mean().item()}, next_q-value : {next_q_value.mean().item()}, q1_loss : {q1_loss.item()}, q2_loss : {q2_loss.item()}')
        self.critic1_optimizer.zero_grad()
        self.critic2_optimizer.zero_grad()
        q_loss.backward()
        self.critic1_optimizer.step()
        self.critic2_optimizer.step()
        
        action_predict, action_predict_logpi,_ = self.actor(state)
        q1_pi = self.critic1(state,action_predict)
        q2_pi = self.critic2(state,action_predict)
        min_q_pi = torch.min(q1_pi,q2_pi)
        
        # if(expert_batch is None):
        entropy_loss = (self.alpha * action_predict_logpi).mean()
        # q = min_q_pi
        # lmbda = 1. / q.abs().mean().detach()
        # actor_q_loss = (-lmbda * q).mean()
        actor_q_loss = (-min_q_pi).mean()
        
        # print(entropy_loss.shape, actor_q_loss.shape)
        if(expert_batch is None):
            actor_loss = entropy_loss + actor_q_loss
        else:
            state_e,action_e,next_state_e,reward_e,not_done_e,_ = expert_batch
            kl_div = -self.actor.get_log_density(state_e, action_e).mean()
            actor_loss =  kl_div + regularization_weight * (entropy_loss + actor_q_loss).mean()# 这里把正则项系数改为了q系数
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (action_predict_logpi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()            
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
        else:
            alpha_loss = torch.tensor(0.).to(self.device)

        if(self.total_it % self.target_update_interval == 0):
            soft_update(self.critic1_target, self.critic1, self.tau)
            soft_update(self.critic2_target, self.critic2, self.tau)
        
        self.critic1.eval()
        self.critic2.eval()
        self.actor.eval()
        
        return q1_loss.item(), q2_loss.item(), actor_loss.item(), alpha_loss.item(), entropy_loss.item(), actor_q_loss.item()
    
    def save(self, filename):
        torch.save(self.critic1.state_dict(), filename + "_critic1")
        torch.save(self.critic1_optimizer.state_dict(), filename + "_critic1_optimizer")
        torch.save(self.critic2.state_dict(), filename + "_critic2")
        torch.save(self.critic2_optimizer.state_dict(), filename + "_critic2_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")


    def load(self, filename):
        if not torch.cuda.is_available():
            self.critic1.load_state_dict(torch.load(filename + "_critic1", map_location=torch.device('cpu')))
            self.critic1_optimizer.load_state_dict(torch.load(filename + "_critic1_optimizer", map_location=torch.device('cpu')))
            self.critic1_target = copy.deepcopy(self.critic1)

            self.critic2.load_state_dict(torch.load(filename + "_critic2", map_location=torch.device('cpu')))
            self.critic2_optimizer.load_state_dict(torch.load(filename + "_critic2_optimizer", map_location=torch.device('cpu')))
            self.critic2_target = copy.deepcopy(self.critic2)

            self.actor.load_state_dict(torch.load(filename + "_actor", map_location=torch.device('cpu')))
            self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer", map_location=torch.device('cpu')))

        else:
            self.critic1.load_state_dict(torch.load(filename + "_critic1"))
            self.critic1_optimizer.load_state_dict(torch.load(filename + "_critic1_optimizer"))
            self.critic1_target = copy.deepcopy(self.critic1)

            self.critic2.load_state_dict(torch.load(filename + "_critic2"))
            self.critic2_optimizer.load_state_dict(torch.load(filename + "_critic2_optimizer"))
            self.critic2_target = copy.deepcopy(self.critic2)


            self.actor.load_state_dict(torch.load(filename + "_actor"))
            self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
    
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
    
class RBFLinearCost:
    """
    MMD cost implementation with rff feature representations

    NOTE: Currently hardcoded to cpu

    :param expert_data: (torch Tensor) expert data used for feature matching
    :param feature_dim: (int) feature dimension for rff
    :param input_type: (str) state (s), state-action (sa), state-next state (ss),
                       state-action-next state (sas)
    :param cost_range: (list) inclusive range of costs
    :param bw_quantile: (float) quantile used to fit bandwidth for rff kernel
    :param bw_samples: (int) number of samples used to fit bandwidth
    :param lambda_b: (float) weight parameter for bonus and cost
    :param lr: (float) learning rate for discriminator/cost update. 0.0 = closed form update
    :param seed: (int) random seed to set cost function
    """
    def __init__(self,
                 expert_data,
                 feature_dim=1024,
                 input_type='sa',
                 cost_range=[-1.,0.],
                 bw_quantile=0.1,
                 bw_samples=100000,
                 lambda_b=1.0,
                 lr=0.0,
                 dynamic_model=None,
                 device='cpu',
                 ):

        self.dynamic_model = dynamic_model

        self.expert_data = expert_data
        input_dim = expert_data.size(1)
        self.input_type = input_type
        self.feature_dim = feature_dim
        self.cost_range = cost_range
        if cost_range is not None:
            self.c_min, self.c_max = cost_range
        self.lambda_b = lambda_b
        self.lr = lr
        self.device = device
        # Fit Bandwidth
        self.quantile = bw_quantile
        self.bw_samples = bw_samples
        self.bw = self.fit_bandwidth(expert_data)

        # Define Phi and Cost weights
        self.rff = nn.Linear(input_dim, feature_dim)
        self.rff.bias.data = (torch.rand_like(self.rff.bias.data)-0.5)*2.0*np.pi
        self.rff.weight.data = torch.rand_like(self.rff.weight.data)/(self.bw+1e-8)

        # W Update Init
        self.w = None

        # Compute Expert Phi Mean
        self.expert_rep = self.get_rep(expert_data)
        self.phi_e = self.expert_rep.mean(dim=0)

    def __call__(self, state, action, next_state):
        bonus_cost, cost_info = self.get_bonus_costs(states=state,
                                                     actions=action, 
                                                     next_states=next_state)
        bonus_cost = bonus_cost[:,0]
        rewards = -1.0 * bonus_cost.cpu().numpy()
        return rewards

    def init(self, replay_buffer, batch_size):
        results = []
        iters = int(replay_buffer.size / batch_size)+int(replay_buffer.size % batch_size != 0)
        for iter in range(iters):
            l = iter * batch_size
            r = min((iter+1)*batch_size, replay_buffer.size)
            state = torch.FloatTensor(replay_buffer.state[l:r]).to(self.device)
            action= torch.FloatTensor(replay_buffer.action[l:r]).to(self.device)
            results.append(self.compute_discrepancy(state, action))
        self.threshold = torch.cat(results, dim=0).max().item()
        print('MILO dynamic model threshold : ',self.threshold)
        
        
    def compute_discrepancy(self, state, action):
        """
        Computes the maximum discrepancy for a given state and action
        """
        # with torch.no_grad():
        #     preds = torch.cat([model.forward(state, action).unsqueeze(0) for model in self.models], dim=0)
        # disc = torch.cat([torch.norm(preds[i]-preds[j], p=2, dim=1).unsqueeze(0) \
        #            for i in range(preds.shape[0]) for j in range(i+1,preds.shape[0])], dim=0) # (n_pairs*batch)
        # return disc.max(0).values.to(torch.device('cpu'))
    
        with torch.no_grad():
            preds_dist = self.dynamic_model(state, action)
            preds = preds_dist.mean
        # print('preds.shape : ',preds.shape)
        disc = torch.cat([torch.norm(preds[i]-preds[j], p=2, dim=1).unsqueeze(0) \
                   for i in range(preds.shape[0]) for j in range(i+1,preds.shape[0])], dim=0) # (n_pairs*batch)
        return disc.max(0).values.to(torch.device('cpu'))


    def get_action_discrepancy(self, state, action):
        """
        Computes the discrepancy of a given (s,a) pair
        """
        # Add Batch Dimension
        if len(state.shape) == 1: state.unsqueeze(0)
        if len(action.shape) == 1: action.unsqueeze(0)

        state = state.float().to(self.device)
        action = action.float().to(self.device)
        return self.compute_discrepancy(state, action)
        
        
        

    def get_rep(self, x):
        """
        Returns an RFF representation given an input
        """
        with torch.no_grad():
            out = self.rff(x.cpu())
            out = torch.cos(out)*np.sqrt(2/self.feature_dim)
        return out

    def fit_bandwidth(self, data):
        """
        Uses the median trick to fit the bandwidth for the RFF kernel
        """
        num_data = data.shape[0]
        idxs_0 = torch.randint(low=0, high=num_data, size=(self.bw_samples,))
        idxs_1 = torch.randint(low=0, high=num_data, size=(self.bw_samples,))
        norm = torch.norm(data[idxs_0, :]-data[idxs_1, :], dim=1)
        bw = torch.quantile(norm, q=self.quantile).item()
        return bw

    def fit_cost(self, data_pi):
        """
        Updates the weights of the cost with the closed form solution
        """
        phi = self.get_rep(data_pi).mean(0)
        feat_diff = phi - self.phi_e

        # Closed form solution
        self.w = feat_diff

        return torch.dot(self.w, feat_diff).item()

    def get_costs(self, x):
        """
        Returrns the IPM (MMD) cost for a given input
        """
        data = self.get_rep(x)
        if self.cost_range is not None:
            return torch.clamp(torch.mm(data, self.w.unsqueeze(1)), self.c_min, self.c_max)
        return torch.mm(data, self.w.unsqueeze(1))

    def get_expert_cost(self):
        """
        Returns the mean expert cost given our current discriminator weights and representations
        """
        return (1-self.lambda_b)*torch.clamp(torch.mm(self.expert_rep, self.w.unsqueeze(1)), self.c_min, self.c_max).mean()

    def get_bonus_costs(self, states, actions, next_states=None):
        """
        Computes the cost with pessimism
        """
        if self.input_type == 'sa':
            rff_input = torch.cat([states, actions], dim=1)
        elif self.input_type == 'ss':
            assert(next_states is not None)
            rff_input = torch.cat([states, next_states], dim=1)
        elif self.input_type == 'sas':
            rff_input = torch.cat([states, actions, next_states], dim=1)
        elif self.input_type == 's':
            rff_input = states
        else:
            raise NotImplementedError("Input type not implemented")

        # Get Linear Cost 
        rff_cost = self.get_costs(rff_input)

        if self.cost_range is not None:
            # Get Bonus from Ensemble
            discrepancy = self.get_action_discrepancy(states, actions)/self.threshold
            discrepancy = discrepancy.view(-1, 1)
            discrepancy[discrepancy>1.0] = 1.0
            # Bonus is LOW if (s,a) is unknown
            bonus = discrepancy * self.c_min
        else:
            bonus = self.get_action_discrepancy(states, actions).view(-1,1)

        # Weight cost components
        ipm = (1-self.lambda_b)*rff_cost

        # Conservative/Pessimism Penalty term
        weighted_bonus = self.lambda_b*bonus.cpu() # Note cpu hardcoding

        # Cost
        cost = ipm - weighted_bonus

        # Logging info
        info = {'bonus': weighted_bonus, 'ipm': ipm, 'v_targ': rff_cost, 'cost': cost}

        return cost, info


# class SAC:
#     def __init__(
#         self,
#         state_dim: int, 
#         action_dim: int, 
#         max_action: float, 
#         lr=3e-4, 
#         wd=5e-3,
#         discount: float = 0.99,
#         tau: float = 0.005, # soft update parameter
        
#         entropy_coeff: float=0.2,
#         target_update_interval:int=1,
#         automatic_entropy_tuning:bool=True,
        
#         device: str = "cpu",
#     ):
#         # print('>> sac device : ',device)
#         # actor = GaussianActor(state_dim, action_dim, max_action).to(device)
#         # actor_optimizer = torch.optim.Adam(actor.parameters(), lr=lr, weight_decay=wd)

#         # critic_1 = CriticNetwork(state_dim=state_dim, action_dim=action_dim).to(device)
#         # critic_1_optimizer = torch.optim.Adam(critic_1.parameters(), lr=lr)
#         # critic_2 = CriticNetwork(state_dim=state_dim, action_dim=action_dim).to(device)
#         # critic_2_optimizer = torch.optim.Adam(critic_2.parameters(), lr=lr)

#         self.actor = GaussianActor(state_dim, action_dim, max_action).to(device)
#         self.actor_optimizer = actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr, weight_decay=wd)
#         self.critic1 = CriticNetwork(state_dim=state_dim, action_dim=action_dim).to(device)
#         self.critic1_target = copy.deepcopy(self.critic1)
#         self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=lr)
        
#         self.critic2 = CriticNetwork(state_dim=state_dim, action_dim=action_dim).to(device)
#         self.critic2_target = copy.deepcopy(self.critic2)
#         self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=lr)

#         self.max_action = max_action
#         self.discount = discount # gamma
#         self.tau = tau
#         self.device = device    

#         self.alpha = entropy_coeff # alpha in paper
#         self.target_update_interval = target_update_interval
#         self.automatic_entropy_tuning = automatic_entropy_tuning
#         if self.automatic_entropy_tuning is True:
#             self.target_entropy = -torch.prod(torch.Tensor([action_dim]).to(self.device)).item()
#             self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
#             self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=lr)

#         self.total_it = 0
    
#     def update_parameters(self, batch, expert_batch=None, regularization_weight=None):
#         self.critic1.train()
#         self.critic2.train()
#         self.actor.train()
#         state,action,next_state,reward,not_done = batch
        
        
        
#         with torch.no_grad():
#             next_action, next_action_logpi, _ = self.actor(next_state)
#             q1_next_target = self.critic1_target(next_state,next_action)
#             q2_next_target = self.critic2_target(next_state,next_action)
#             min_q_next_target = torch.min(q1_next_target,q2_next_target) - self.alpha * next_action_logpi
#             next_q_value = reward + not_done * self.discount * min_q_next_target
#         q1 = self.critic1(state, action)
#         q2 = self.critic2(state, action)
#         q1_loss = F.mse_loss(q1, next_q_value)
#         q2_loss = F.mse_loss(q2, next_q_value)
#         q_loss = q1_loss + q2_loss
        
#         self.critic1_optimizer.zero_grad()
#         self.critic2_optimizer.zero_grad()
#         q_loss.backward()
#         self.critic1_optimizer.step()
#         self.critic2_optimizer.step()
        
#         action_predict, action_predict_logpi,_ = self.actor(state)
#         q1_pi = self.critic1(state,action_predict)
#         q2_pi = self.critic2(state,action_predict)
#         min_q_pi = torch.min(q1_pi,q2_pi)
        
#         if(expert_batch is None):
#             actor_loss = ((self.alpha*action_predict_logpi) - min_q_pi).mean()
#         else:
#             state_e,action_e,next_state_e,reward_e,not_done_e,_ = expert_batch
#             kl_div = -self.actor.get_log_density(state_e, action_e).mean()
#             actor_loss = regularization_weight * kl_div + ((self.alpha*action_predict_logpi) - min_q_pi).mean()
#         self.actor_optimizer.zero_grad()
#         actor_loss.backward()
#         self.actor_optimizer.step()
        
#         if self.automatic_entropy_tuning:
#             alpha_loss = -(self.log_alpha * (action_predict_logpi + self.target_entropy).detach()).mean()

#             self.alpha_optim.zero_grad()
#             alpha_loss.backward()
#             self.alpha_optim.step()

#             self.alpha = self.log_alpha.exp()
#         else:
#             alpha_loss = torch.tensor(0.).to(self.device)

#         self.total_it += 1
#         if(self.total_it % self.target_update_interval == 0):
#             soft_update(self.critic1_target, self.critic1, self.tau)
#             soft_update(self.critic2_target, self.critic2, self.tau)
        
#         self.critic1.eval()
#         self.critic2.eval()
#         self.actor.eval()
        
#         return q1_loss.item(), q2_loss.item(), actor_loss.item(), alpha_loss.item()
    
#     def save(self, filename):
#         torch.save(self.critic1.state_dict(), filename + "_critic1")
#         torch.save(self.critic1_optimizer.state_dict(), filename + "_critic1_optimizer")
#         torch.save(self.critic2.state_dict(), filename + "_critic2")
#         torch.save(self.critic2_optimizer.state_dict(), filename + "_critic2_optimizer")

#         torch.save(self.actor.state_dict(), filename + "_actor")
#         torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")


#     def load(self, filename):
#         if not torch.cuda.is_available():
#             self.critic1.load_state_dict(torch.load(filename + "_critic1", map_location=torch.device('cpu')))
#             self.critic1_optimizer.load_state_dict(torch.load(filename + "_critic1_optimizer", map_location=torch.device('cpu')))
#             self.critic1_target = copy.deepcopy(self.critic1)

#             self.critic2.load_state_dict(torch.load(filename + "_critic2", map_location=torch.device('cpu')))
#             self.critic2_optimizer.load_state_dict(torch.load(filename + "_critic2_optimizer", map_location=torch.device('cpu')))
#             self.critic2_target = copy.deepcopy(self.critic2)

#             self.actor.load_state_dict(torch.load(filename + "_actor", map_location=torch.device('cpu')))
#             self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer", map_location=torch.device('cpu')))

#         else:
#             self.critic1.load_state_dict(torch.load(filename + "_critic1"))
#             self.critic1_optimizer.load_state_dict(torch.load(filename + "_critic1_optimizer"))
#             self.critic1_target = copy.deepcopy(self.critic1)

#             self.critic2.load_state_dict(torch.load(filename + "_critic2"))
#             self.critic2_optimizer.load_state_dict(torch.load(filename + "_critic2_optimizer"))
#             self.critic2_target = copy.deepcopy(self.critic2)


#             self.actor.load_state_dict(torch.load(filename + "_actor"))
#             self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))

class MILO:
    def __init__(self,
                 state_dim,
                 action_dim,
                 max_action,
                 dynamic_model,
                 buffer_e,
                 # reward
                 reward_feature_dim:int=512,
                 reward_bw_quantile:float=0.2,
                 reward_lambda_b:float=0.1,
                 
                 # sac
                 lr:float=1e-4,
                 wd:float=1e-5,
                 target_update_interval:int=1,
                 device:str='cpu',
                 
                 # 
                 collection_freq:int=20,
                 buffer_size:int=1e6,
                 data_collection_per_times:int=5000,
                 horizon:int=5,
                 
                 # expert bc
                 use_expert_behavior:bool=True,
                 regularization_weight:float=0.2
                 
                 ):
        self.state_dim=state_dim
        self.action_dim=action_dim
        self.max_action=max_action
        self.dynamic_model = dynamic_model
        self.device=device
        self.buffer_e = buffer_e
        # self.seed = seed
        self.collect_freq = collection_freq
        self.data_collection_per_times = data_collection_per_times
        self.horizon=horizon
        self.reward_init = False
        self.use_expert_behavior = use_expert_behavior
        self.regularization_weight=regularization_weight
        self.buffer_size = buffer_size
        self.total_it = 0
        
        expert_state = torch.Tensor(buffer_e.state).to(device)
        expert_action = torch.Tensor(buffer_e.action).to(device)
        self.reward = RBFLinearCost(
            torch.cat([expert_state, expert_action], dim=-1), 
            feature_dim=reward_feature_dim, 
            bw_quantile=reward_bw_quantile, 
            lambda_b=reward_lambda_b, 
            # seed=seed,
            dynamic_model=dynamic_model,
            device=device)
        
        self.sac_agent = SAC(state_dim=state_dim,
                             action_dim=action_dim,
                             max_action=max_action,
                             device=device,
                             target_update_interval=target_update_interval,
                             lr=lr,
                             wd=wd)
        self.modelbuffer = ReplayBuffer(state_dim=self.state_dim,
                                  action_dim=self.action_dim,
                                  max_size=self.buffer_size,
                                  device=self.device)
        
    @property
    def actor(self):
        return self.sac_agent.actor
        
        
    def collect_data(self, buffer_e, batch_size):
        # states_list = []
        # actions_list = []
        # next_states_list = []
        # dones_list = []
        with torch.no_grad():
            state,_,_,_,_,_ = buffer_e.sample(batch_size)
            for t in range(self.horizon):
                _ ,_,action = self.sac_agent.actor(state)
                # print(f'state.shape : [{state.shape}], action.shape:[{action.shape}]')
                next_state_dist = self.dynamic_model(state, action)
                next_states = next_state_dist.sample()
                if self.dynamic_model.use_reward:
                    next_states = next_states[:, :, :-1]
                else:
                    next_states = next_states
                model_indexes = np.random.randint(0, next_states.shape[0], size=(state.shape[0]))
                next_state = next_states[model_indexes, np.arange(state.shape[0])]
                # states_list.append(state)
                # actions_list.append(action)
                # next_states_list.append(next_state)
                
                reward_input = torch.cat([state, action],dim=-1)
                self.reward.fit_cost(reward_input)
                
                with torch.no_grad():
                    reward = torch.Tensor(self.reward(state=state,action=action,next_state=next_state)).to(self.device)
                    reward = torch.unsqueeze(reward, dim=1)
                    # print('reward min/max/sum/mean/median : ',torch.min(reward).item(),torch.max(reward).item(),torch.sum(reward).item(),torch.mean(reward).item(),torch.median(reward).item(),reward.shape)
                
                if(t != self.horizon - 1):
                    dones = torch.zeros_like(reward)
                else:
                    dones = torch.ones_like(reward)
                # dones = torch.zeros_like(reward)
                batch_data = dict(
                    state=state.cpu().numpy(),
                    action=action.cpu().numpy(),
                    next_state=next_state.cpu().numpy(),
                    reward=reward.cpu().numpy(),
                    done=dones.cpu().numpy()
                )
                # print('generate data, reward.mean : ',reward.mean())
                if(t > 0): 
                    # self.modelbuffer.add_batch(states=batch_data['state'],
                    #                         actions=batch_data['action'],
                    #                         next_states=batch_data['next_state'],
                    #                         rewards=batch_data['reward'],
                    #                         dones=batch_data['done'])
                    for i in range(batch_size):
                        self.modelbuffer.add(state=batch_data['state'][i],
                                            action=batch_data['action'][i],
                                            next_state=batch_data['next_state'][i],
                                            reward=batch_data['reward'][i],
                                            done=batch_data['done'][i])
                
                
                state = next_state
                
    def train(self, buffer_expert, buffer_offline, buffer_all, iters, batch_size=256):
        if(self.reward_init == False):
            self.reward.init(buffer_all,batch_size)
            self.reward_init = True
        log_dict = {}
        for iter in range(iters):
            if(self.total_it % self.collect_freq == 0):
                self.collect_data(buffer_e=buffer_expert,
                                    batch_size=self.data_collection_per_times)
            
            
            
            self.total_it += 1
            state, action, next_state, reward, not_done, flag = self.modelbuffer.sample(batch_size)  
            batch = (state,action,next_state,reward,not_done)
            if(self.use_expert_behavior):
                batch_e = buffer_expert.sample(batch_size)
                critic_1_loss, critic_2_loss, actor_loss, ent_loss, entropy_loss, actor_q_loss = self.sac_agent.update_parameters(batch, batch_e, self.regularization_weight,log_dict=log_dict)
            else:
                batch_a = buffer_all.sample(batch_size)
                critic_1_loss, critic_2_loss, actor_loss, ent_loss, entropy_loss, actor_q_loss = self.sac_agent.update_parameters(batch, batch_a, self.regularization_weight,log_dict=log_dict)
                
            critic_loss = critic_1_loss + critic_2_loss
        
            update_log(keywords='critic_loss',value=critic_loss,log_dict=log_dict)
            update_log(keywords='critic_1_loss',value=critic_1_loss,log_dict=log_dict)
            update_log(keywords='critic_2_loss',value=critic_2_loss,log_dict=log_dict)
            update_log(keywords='entropy_loss',value=entropy_loss,log_dict=log_dict)
            update_log(keywords='actor_q_loss',value=actor_q_loss,log_dict=log_dict)
            update_log(keywords='actor_loss',value=actor_loss,log_dict=log_dict)
            update_log(keywords='ent_loss',value=ent_loss,log_dict=log_dict)
            update_log(keywords='alpha',value=self.sac_agent.alpha.detach().cpu().numpy(),log_dict=log_dict)
            
        return log_dict
            
    def save(self, filename):
        # torch.save(self.reward.state_dict(), filename + "_reward")
        # torch.save(self.reward_optimizer.state_dict(), filename + "_reward_optimizer")

        self.sac_agent.save(filename)

    def load(self, filename):
        # if not torch.cuda.is_available():
        #     self.reward.load_state_dict(torch.load(filename + "_reward", map_location=torch.device('cpu')))
        #     self.reward_optimizer.load_state_dict(torch.load(filename + "_reward_optimizer", map_location=torch.device('cpu')))

        # else:
        #     self.reward.load_state_dict(torch.load(filename + "_reward"))
        #     self.reward_optimizer.load_state_dict(torch.load(filename + "_reward_optimizer"))
            
        self.sac_agent.load(filename)   
                    
                
                
                
                
                