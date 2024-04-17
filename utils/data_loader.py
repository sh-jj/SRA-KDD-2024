





from typing import Any, Callable, Iterator, List, Optional, Tuple, Union, cast

import gym
import numpy as np
import torch
from typing_extensions import Protocol

from d3rlpy.dataset import Episode, TransitionMiniBatch
from d3rlpy.preprocessing.reward_scalers import RewardScaler
from d3rlpy.preprocessing.stack import StackedObservation

from d3rlpy.metrics.scorer import AlgoProtocol

import pandas as pd
import os

from copy import deepcopy

import pickle

WINDOW_SIZE = 1024









class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6), device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        self.max_size = max_size
        self.ptr = 0
        
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))
        self.flag = np.zeros((max_size, 1))

        self.device = device

    def clear(self):
        self.ptr = 0
        self.size = 0

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done
    
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def extend(self, replay_):

        if self.size == 0:
            max_size = self.max_size
            self = deepcopy(replay_)

            self.max_size = max_size
            return self


        extend_size = min(self.max_size - self.size, replay_.size)

        self_size = self.size

        self.state = np.concatenate((self.state[:self_size], replay_.state[:extend_size]))
        self.action = np.concatenate((self.action[:self_size], replay_.action[:extend_size]))
        self.next_state = np.concatenate((self.next_state[:self_size], replay_.next_state[:extend_size]))
        self.reward = np.concatenate((self.reward[:self_size], replay_.reward[:extend_size]))
        self.not_done = np.concatenate((self.not_done[:self_size], replay_.not_done[:extend_size]))
        self.flag = np.concatenate((self.flag[:self_size], replay_.flag[:extend_size]))

        self.size += extend_size
        return self
    
    def add_batch(self, states, actions, next_states, rewards, dones, flags=None):
        batch_size = len(states)

        indices = np.arange(self.ptr, min(self.ptr + batch_size, self.max_size))

        self.state[indices] = states
        self.action[indices] = actions
        self.next_state[indices] = next_states
        self.reward[indices] = rewards
        self.not_done[indices] = 1. - dones
        if(flags is not None):
            self.flag[indices] = flags

        self.ptr += len(indices)
        self.size = min(self.size + len(indices), self.max_size)

    def sample_index(self, batch_size):
        ind = np.random.randint(0, self.size , size=batch_size)
        return ind
        
    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device),
            torch.FloatTensor(self.flag[ind]).to(self.device),
        )

    def sample_in_set(self, batch_size, subset):

        ind_ = np.random.randint(0, len(subset), size=batch_size)
        ind = subset[ind_]

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device),
            torch.FloatTensor(self.flag[ind]).to(self.device),
        )
    
    def sample_subset(self, ind):

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device),
            torch.FloatTensor(self.flag[ind]).to(self.device),
        )

    def convert_D4RL(self, dataset):
        self.state = dataset['observations']
        self.action = dataset['actions']
        self.next_state = dataset['next_observations']
        self.reward = dataset['rewards'].reshape(-1, 1)
        self.not_done = 1. - dataset['terminals'].reshape(-1, 1)
        self.flag = dataset['flag'].reshape(-1, 1)
        self.size = self.state.shape[0]
    
    def convert_D3RL(self, dataset, drop_last=True):

        observations, actions, rewards = dataset.observations, dataset.actions, dataset.rewards
        terminals, episode_terminals = dataset.terminals, dataset.episode_terminals

        obs_ = []
        next_obs_ = []
        action_ = []
        reward_ = []
        done_ = []
        flag_ = []

        N = len(observations)

        episode_step = 0

        for i in range(N-1):
            
            obs = observations[i].astype(np.float32)
            new_obs = observations[i+1].astype(np.float32)

            action = actions[i].astype(np.float32)
            reward = rewards[i].astype(np.float32)
            done_bool = bool(episode_terminals[i])
            flag = terminals[i]

            if drop_last and done_bool:
                # Skip this transition and don't apply terminals on the last step of an episode
                # antmaze dataset can not do this
                episode_step = 0
                continue

            obs_.append(obs)
            next_obs_.append(new_obs)
            action_.append(action)
            reward_.append(reward)
            done_.append(done_bool)
            flag_.append(flag)
            episode_step += 1

            if done_bool:
                episode_step = 0

        self.state = np.array(obs_)
        self.action = np.array(action_)
        self.next_state = np.array(next_obs_)
        self.reward = np.array(reward_).reshape(-1, 1)
        self.not_done = 1. - np.array(done_).reshape(-1, 1)
        self.flag = np.array(flag_).reshape(-1, 1)
        self.size = self.state.shape[0]

    def normalize_states(self, eps=1e-3, mean=None, std=None):


        if mean is None and std is None:
            mean = self.state.mean(0, keepdims=True)
            std = self.state.std(0, keepdims=True) + eps

        # print(self.state)

        self.state = (self.state - mean) / std
        self.next_state = (self.next_state - mean) / std

        # print(self.state)

        return mean, std
    
    def save(self, file_name):
        pickle.dump(self, open(file_name, "wb"))
    
    def load(self, file_name):
        
        return pickle.load(open(file_name, "rb"))


class BufferEnsemble(object):
    def __init__(self, buffer_list):

        self.buffers = buffer_list
        self.buffer_num = 0
        for idx, item in enumerate(self.buffers):
            try:
                self.buffer_num += item[0].buffer_num
            except:
                self.buffer_num += 1
    
    def sample(self, batch_size, return_belong=False):

        data = []
        belongs = []
        
        delta = 0
        
        for idx, item in enumerate(self.buffers):
            sub_batch_size = int(batch_size * item[1] + 0.1)

            sub_batch_size = min(sub_batch_size, item[0].size)
            if sub_batch_size == 0:
                try:
                    delta += item[0].buffer_num
                except:
                    delta += 1
                continue
            
            try:
                sub_data, sub_belongs = item[0].sample(sub_batch_size, return_belong=return_belong)
            except:
                sub_data = item[0].sample(sub_batch_size)
                sub_belongs = [0] * sub_batch_size
            
            data.append(sub_data)
            
            for i in range(len(sub_belongs)):
                sub_belongs[i] += delta
            
            belongs = belongs + sub_belongs
            
            try:
                delta += item[0].buffer_num
            except:
                delta += 1
        
        #state, action, next_state, reward, not_done
        output = []
        for item in zip(*data):
            sub_data = torch.cat(item, axis=0)
            output.append(sub_data)
        
        if return_belong:
            return output, belongs
        
        return output
    @property
    def size(self):
        ret = 0
        for idx,item in enumerate(self.buffers):
            ret += item[0].size
        return ret
        



def sample_buffers(buffer, batch_size, return_belong=False):
        
        data = []
        belongs = []
        
        for idx, item in enumerate(buffer):
            sub_batch_size = int(batch_size * item[1] + 0.1)
            sub_data = item[0].sample(sub_batch_size)
            data.append(sub_data)

            belongs = belongs + [idx] * sub_batch_size
        
        #state, action, next_state, reward, not_done
        output = []
        for item in zip(*data):
            sub_data = torch.cat(item, axis=0)
            output.append(sub_data)
        
        if return_belong:
            return output, belongs
        
        return output

