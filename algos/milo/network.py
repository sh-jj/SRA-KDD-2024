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

def get_activation_funtion(activation):
    if isinstance(activation, str):
        activation_str = activation.lower()
        
        if(activation_str == 'relu'):
            return nn.ReLU()
        elif(activation_str=='sigmoid'):
            return nn.Sigmoid()
        elif(activation_str=='tanh'):
            return nn.Tanh()
        else:
            raise ValueError(f"Unknown activation function:{activation_str}, please choose from [relu, sigmoid, tanh].")
    elif isinstance(activation, nn.Module):
        return activation
    else:
        raise TypeError("Activation must be either a string or an instance of torch.nn.Module")



class CustomMultiHeadMLP(nn.Module):

    def __init__(self, 
                 input_dim, 
                 output_dim, 
                 hidden_size,
                 activation='relu',):
        
        """多头输出的多层感知器（MLP）网络。

        Args:
            input_dim (int): 输入维度。
            output_dims (list of int): 多个输出头部的维度列表。
            hidden_size (list of int): 隐藏层大小的列表。
            activation (torch.nn.Module, optional): 激活函数，默认为 nn.ReLU()。

        Attributes:
            input_dim (int): 输入维度。
            output_dims (list of int): 多个输出头部的维度列表。
            hidden_size (list of int): 隐藏层大小的列表。
            activation (torch.nn.Module): 激活函数。

            hidden_layers (torch.nn.ModuleList): 隐藏层的线性层列表。
            output_layers (torch.nn.ModuleList): 多个输出头部的线性层列表。

        """
        super(CustomMultiHeadMLP, self).__init__()
        self.input_dim = input_dim
        self.output_dim=output_dim
        self.hidden_size=hidden_size
        self.activation=get_activation_funtion(activation)
        
        layer_sizes = [input_dim] + hidden_size
        self.hidden_layers = nn.ModuleList()
        for i in range(len(layer_sizes)-1):
            self.hidden_layers.append(nn.Linear(layer_sizes[i],layer_sizes[i+1]))
        
        self.output_layers = nn.ModuleList()
        if(isinstance(output_dim,int)):
            output_dim = [output_dim]
        for output_dim in output_dim:
            self.output_layers.append(nn.Linear(hidden_size[-1], output_dim))
        
    def forward(self, x):

        for layer in self.hidden_layers:
            x = self.activation(layer(x))
            
        outputs = [output_layer(x) for output_layer in self.output_layers]
        return outputs
    
    def select_action(self, state, device):
        raise NotImplementedError




class DeterministicActor(CustomMultiHeadMLP):
    """确定性 Actor 类，继承自多头 MLP 类。

    Args:
        state_dim (int): 观测空间维度。
        action_dim (int): 动作空间维度。
        max_action (int): 动作空间范围(-max_action, max_action)
        hidden_size (list of int): 隐藏层大小的列表。
        activation (torch.nn.Module, optional): 激活函数，默认为 nn.ReLU()。

    """

    def __init__(self, state_dim, action_dim, max_action=1, hidden_size=[256,256], activation=nn.ReLU()):
        # 调用父类构造函数初始化多头 MLP
        super(DeterministicActor, self).__init__(input_dim=state_dim,
                                                 output_dim=[action_dim], 
                                                 hidden_size=hidden_size, 
                                                 activation=activation)
        self.max_action = max_action
        

    def forward(self, x):

        outputs = super(DeterministicActor, self).forward(x)

        return outputs[0]
    
    def select_action(self, state, device):
        state = torch.tensor(state.reshape(1,-1), device=device, dtype=torch.float32)
        action = self(state)
        action = torch.clamp(action * self.max_action, -self.max_action, self.max_action).cpu().data.numpy().flatten()
        return action
    
    # def get_log_density(self, state, action):
    #     output = self(state)
    #     return 

    
    
    
    
class GaussianActor(CustomMultiHeadMLP):
    def __init__(self, state_dim, action_dim, max_action=1, hidden_size=[256,256], activation=nn.ReLU()):
        """
        Gaussian Actor class for policy approximation using a multi-head MLP.

        Args:
        - state_dim (int): Dimension of the state space
        - action_dim (int): Dimension of the action space
        - max_action (float): Maximum action value allowed
        - hidden_size (list): List of integers specifying hidden layer sizes (default=[256,256])
        - activation (torch.nn.Module): Activation function (default=nn.ReLU())
        """
        super(GaussianActor, self).__init__(input_dim=state_dim,
                                            output_dim=[action_dim, action_dim], 
                                            hidden_size=hidden_size,
                                            activation=activation)
        self.max_action = max_action

    def _get_outputs(self, state):

        action_mu, action_sigma = super(GaussianActor, self).forward(state)
        mu = torch.clip(action_mu, MEAN_MIN, MEAN_MAX)
        log_sigma = torch.clip(action_sigma, LOG_STD_MIN, LOG_STD_MAX)
        sigma = torch.exp(log_sigma)
        
        a_distribution = TransformedDistribution(
            Normal(mu, sigma), TanhTransform(cache_size=1)
        )
        
        a_tanh_mode = torch.tanh(mu)
        return a_distribution, a_tanh_mode
    
    def forward(self, state):

        a_dist, a_tanh_mode = self._get_outputs(state)
        action = a_dist.rsample()
        logp_pi = a_dist.log_prob(action).sum(axis=-1,keepdim=True)
        return action, logp_pi, a_tanh_mode
    
    def get_log_density(self, state, action):

        a_dist, _ = self._get_outputs(state)
        action_clip = torch.clip(action,-self.max_action + EPS, self.max_action - EPS)
        logp_action = a_dist.log_prob(action_clip)
        return logp_action 
    
    @torch.no_grad()
    def select_action(self, state, device):

        state = torch.FloatTensor(state.reshape(1,-1)).to(device)
        _, _, action = self.forward(state)
        return action.cpu().data.numpy().flatten()



class CriticNetwork(CustomMultiHeadMLP):
    def __init__(self, state_dim, action_dim=0, hidden_size=[256,256],activation='relu'):
        super(CriticNetwork, self).__init__(input_dim=state_dim+action_dim,
                                            output_dim=[1],
                                            hidden_size=hidden_size,
                                            activation=activation)
    def forward(self, state, action=None):
        input = state
        if(action != None):
            input = torch.cat([state,action],dim=-1)     
        Q = super(CriticNetwork, self).forward(input)
        return Q[0]