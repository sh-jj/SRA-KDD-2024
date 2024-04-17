import random
import gym
import numpy as np

import d3rlpy
from d3rlpy.datasets import MDPDataset

def mix_mdp_mujoco_datasets(env_id, n, dataset_types, sizes, seed=19260817):
  assert len(dataset_types) == len(sizes)
  assert sum(sizes) == n

  random.seed(seed)
  
  print(f"Dataset size: {n}")
  # if np.all(np.asarray(ratios) != 1.0):
  
  data_name = env_id
  for dataset_type, size in zip(dataset_types, sizes):
    data_name += "_" + str(dataset_type) + "_" + str(size)
  print("data_name: ", data_name)
  mix_data_path = "mix_data/{}.h5".format(data_name)
  
  try:
    
    print("-----------")
    dataset = MDPDataset.load(mix_data_path)
    print("load the mixed dataset from <<< ", mix_data_path)
    print("-----------")
    
    for dataset_type, size in zip(dataset_types, sizes):

      print("Env:", env_id)
      if env_id in ["pen", "hammer", "door", "relocate"]:
        _, env = d3rlpy.datasets.get_d4rl(f"{env_id}-{dataset_type}")
      else:
        _, env = d3rlpy.datasets.get_dataset(f"{env_id}-{dataset_type}")
      break
    
    
  except:
  # if 3 < 33:
    print("build the mixed dataset")
    
    episodes = []
    for dataset_type, size in zip(dataset_types, sizes):
      dataset_id = f"{env_id}-{dataset_type}"

      print("Env:", env_id)
      if env_id in ["pen", "hammer", "door", "relocate"]:
        dataset, env = d3rlpy.datasets.get_d4rl(f"{env_id}-{dataset_type}")
      else:
        dataset, env = d3rlpy.datasets.get_dataset(f"{env_id}-{dataset_type}")

      num_transitions = 0
      print(f"Mix {size} ({size / n }) data from {dataset_type}.")
      
      if size == 0:
            continue
  
      all_tr = 0
      for item in dataset.episodes:
            all_tr += len(item)
      print(all_tr)
      
      
      while num_transitions < size:
        episode = random.choice(dataset.episodes)
        
        # print(episode)
        episodes.append(episode)
        num_transitions += len(episode.transitions)
      print(num_transitions)

    observations = []
    actions = []
    rewards = []
    terminals = []
    episode_terminals = []

    ep_rets = []
    for episode in episodes:
      ep_rets.append(episode.rewards.sum())
      for idx, transition in enumerate(episode):
        observations.append(transition.observation)
        if isinstance(env.action_space, gym.spaces.Box):
          actions.append(np.reshape(transition.action, env.action_space.shape))
        else:
          actions.append(transition.action)
        rewards.append(transition.reward)
        terminals.append(transition.terminal)
        episode_terminals.append(idx == len(episode) - 1)
    print(f"ep_reward_sum   Max/Mean/Median/Min: {np.max(ep_rets)}/{np.mean(ep_rets)}/{np.median(ep_rets)}/{np.min(ep_rets)}")
    dataset = MDPDataset(
      observations=np.stack(observations),
      actions=np.stack(actions),
      rewards=np.stack(rewards),
      terminals=np.stack(terminals).astype(float),
      episode_terminals=np.stack(episode_terminals).astype(float))
    
    
    dataset.dump(mix_data_path)
    print("save mixed dataset >>> ", mix_data_path)
    
  # else:
  #   for dataset_type, ratio in zip(dataset_types, ratios):
  #     if ratio == 1.0:
  #       if env_id in ["pen", "hammer", "door", "relocate"]:
  #         dataset, env = d3rlpy.datasets.get_d4rl(f"{env_id}-{dataset_type}")
  #       else:
  #         dataset, env = d3rlpy.datasets.get_dataset(f"{env_id}-{dataset_type}")
  #       break
  #   ep_rets = []
    
  #   for episode in dataset.episodes:
  #     ep_rets.append(episode.rewards.sum())
  #   print(f"Max/Mean/Median/Min: {np.max(ep_rets)}/{np.mean(ep_rets)}/{np.median(ep_rets)}/{np.min(ep_rets)}")
    
  return env, dataset

if __name__ == "__main__":
      
      
  dataset_types=["expert-v2", "medium-v2"]
  
  sizes = [100000, 900000]
  n = np.sum(sizes)
  
  env, dataset = mix_mdp_mujoco_datasets("hopper",
                      n=n,
                      dataset_types=dataset_types,
                      sizes = sizes)
  
  data_name = "hopper"
  for dataset_type, size in zip(dataset_types, sizes):
    data_name += "_" + str(dataset_type) + "_" + str(size)
  print(data_name)
  dataset.dump("mix_data/{}.h5".format(data_name))
  
  mixed_dataset = MDPDataset.load("mix_data/{}.h5".format(data_name))