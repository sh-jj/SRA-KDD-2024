


import gym
import d4rl # Import required to register environments, you may need to also import the submodule

from d3rlpy.datasets import MDPDataset
import numpy as np
import os

# maze2d-umaze-expert-v1
# maze2d-{type}-{policy}-v1

def make_env_only(env_name):
    if 'maze2d' in env_name:

        try:
            env = gym.make(env_name)
        except:

            env_attr = env_name.split('-')

            env_type = env_attr[1]


            reward_type = env_attr[3]

            if "dense" in env_name:
                env_name_gym = f"maze2d-{env_type}-{reward_type}-v1"
            else:
                env_name_gym = f"maze2d-{env_type}-v1"

            env = gym.make(env_name_gym)
    else:
        env = gym.make(env_name)

    return env

def get_maze2d(env_name):   

    env_attr = env_name.split('-')

    env_type = env_attr[1]

    policy = env_attr[2]
    
    reward_type = env_attr[3]

    if "dense" in env_name:
        env_name_gym = f"maze2d-{env_type}-{reward_type}-v1"
    else:
        env_name_gym = f"maze2d-{env_type}-v1"

    # print(env_attr)
    # print(env_type, policy, reward_type)
    # print(env_name_gym)


    env = gym.make(env_name_gym)

    path = os.path.join('data/maze2d/', 'data', policy, env_type, 'data.npz')

    
    if "dense" in env_name:
        path = os.path.join('data/maze2d/', 'data', policy, env_type + "-dense", 'data.npz')
    # print(path)
    # exit(0)

    dataset = np.load(path)
    print("load data from <<< ", path)

    return dataset, env


def qlearning_dataset_with_timeouts(env,
                                    dataset=None,
                                    terminate_on_end=False,
                                    disable_goal=True,
                                    **kwargs):
    if dataset is None:
        dataset = env.get_dataset(**kwargs)

    N = dataset['rewards'].shape[0]
    obs_ = []
    next_obs_ = []
    action_ = []
    reward_ = []
    done_ = []
    realdone_ = []
    if "infos/goal" in dataset:
        if not disable_goal:
            dataset["observations"] = np.concatenate(
                [dataset["observations"], dataset['infos/goal']], axis=1)
        else:
            pass
        # dataset["observations"] = np.concatenate([
        #     dataset["observations"],
        #     np.zeros([dataset["observations"].shape[0], 2], dtype=np.float32)
        # ], axis=1)
        # dataset["observations"] = np.concatenate([
        #     dataset["observations"],
        #     np.zeros([dataset["observations"].shape[0], 2], dtype=np.float32)
        # ], axis=1)

    episode_step = 0
    for i in range(N - 1):
        obs = dataset['observations'][i]
        new_obs = dataset['observations'][i + 1]
        action = dataset['actions'][i]
        reward = dataset['rewards'][i]
        done_bool = bool(dataset['terminals'][i])
        realdone_bool = bool(dataset['terminals'][i])
        if "infos/goal" in dataset:
            final_timestep = True if (dataset['infos/goal'][i] !=
                                    dataset['infos/goal'][i + 1]).any() else False
        else:
            final_timestep = dataset['timeouts'][i]

        if i < N - 1:
            done_bool += final_timestep

        if (not terminate_on_end) and final_timestep:
            # Skip this transition and don't apply terminals on the last step of an episode
            episode_step = 0
            continue
        if done_bool or final_timestep:
            episode_step = 0

        obs_.append(obs)
        next_obs_.append(new_obs)
        action_.append(action)
        reward_.append(reward)
        done_.append(done_bool)
        realdone_.append(realdone_bool)
        episode_step += 1

    return {
        'observations': np.array(obs_),
        'actions': np.array(action_),
        'next_observations': np.array(next_obs_),
        'rewards': np.array(reward_)[:],
        'terminals': np.array(done_)[:],
        'realterminals': np.array(realdone_)[:],
    }

def get_d4rl2d3rl(env_name: str):
    """Returns d4rl dataset and envrironment.

    The dataset is provided through d4rl.

    .. code-block:: python

        from d3rlpy.datasets import get_d4rl

        dataset, env = get_d4rl('hopper-medium-v0')

    References:
        * `Fu et al., D4RL: Datasets for Deep Data-Driven Reinforcement
          Learning. <https://arxiv.org/abs/2004.07219>`_
        * https://github.com/rail-berkeley/d4rl

    Args:
        env_name: environment id of d4rl dataset.

    Returns:
        tuple of :class:`d3rlpy.dataset.MDPDataset` and gym environment.

    """
    import d4rl  # type: ignore

    if 'maze2d' in env_name:

        try:
            env = gym.make(env_name)
            dataset = env.get_dataset()
        except:
            dataset, env = get_maze2d(env_name)
    else:
        env = gym.make(env_name)
        dataset = env.get_dataset()

    observations = dataset["observations"]
    actions = dataset["actions"]
    rewards = dataset["rewards"]
    terminals = dataset["terminals"]
    timeouts = dataset["timeouts"]
    episode_terminals = np.logical_or(terminals, timeouts)

    # print("terminals.sum : ",terminals.sum())
    # print("timeouts.sum : ",timeouts.sum())
    # exit(0)

    mdp_dataset = MDPDataset(
        observations=np.array(observations, dtype=np.float32),
        actions=np.array(actions, dtype=np.float32),
        rewards=np.array(rewards, dtype=np.float32),
        terminals=np.array(terminals, dtype=np.float32),
        episode_terminals=np.array(episode_terminals, dtype=np.float32),
        )


    return mdp_dataset, env


def build_mdpdata_from_episodes(episodes, env): 

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
    dataset = MDPDataset(
      observations=np.stack(observations),
      actions=np.stack(actions),
      rewards=np.stack(rewards),
      terminals=np.stack(terminals).astype(float),
      episode_terminals=np.stack(episode_terminals).astype(float))
    
    rewards=np.stack(rewards)
    print(f"ep_reward_sum   Max/Mean/Median/Min: {np.max(ep_rets)}/{np.mean(ep_rets)}/{np.median(ep_rets)}/{np.min(ep_rets)}")
    print(f"state-action reward     Max/Mean/Median/Min:   {np.max(rewards)}/{np.mean(rewards)}/{np.median(rewards)}/{np.min(rewards)} ")

    # dataset.dump(mix_data_path)
    # print("save mixed dataset >>> ", mix_data_path)

    return dataset


def get_sorted_idx(data, discount = 0.99):
    
    returns = []
    for episode_i in data.episodes:
        returns.append(episode_i.rewards.sum())

    return np.argsort(returns)[::-1]

def get_offline_imitation_data(expert_name, offline_name, expert_num=10, offline_exp=0):

    data_e, env = get_d4rl2d3rl(expert_name)

    # exp_sorted_idx = get_sorted_idx(data_e)


    # exp_exp_idx = exp_sorted_idx[:expert_num]
    # off_exp_idx = exp_sorted_idx[expert_num: expert_num + offline_exp]

    # expert_episodes = [data_e.episodes[idx] for idx in exp_exp_idx]

    expert_episodes = []
    offline_episodes = []

    if "antmaze" in expert_name:
        # sort the episode by the return

        ep_rets = []
        for episode_i in data_e:
            # print(episode_i.size())
            if(episode_i.size() == 0):
                ep_rets.append(0)
                continue
            ep_rets.append(
                episode_i.rewards.sum() / (1e-4 + np.linalg.norm(episode_i[0].observation[:2])))
        # ep_rets = [
        #     episode_i.rewards.sum() / (1e-4 + np.linalg.norm(episode_i[0].observation[:2])) # robot start location
        #     for episode_i in data_e
        # ]
        
        ranked_idx = np.argsort(ep_rets)[::-1]
        
        for i in range(expert_num):
            expert_episodes.append(data_e.episodes[ranked_idx[i]])
        for i in range(expert_num, len(ranked_idx)):
            if(data_e.episodes[ranked_idx[i]].size() == 0):continue
            offline_episodes.append(data_e.episodes[ranked_idx[i]])
            
        print('antmaze-dataset')
        
    else:
        expert_episodes = data_e.episodes[:expert_num]



        data_o, env = get_d4rl2d3rl(offline_name)


        # offline_episodes = [data_e.episodes[idx] for idx in off_exp_idx]
        for episode_i in data_e.episodes[expert_num:expert_num+offline_exp]:
            offline_episodes.append(episode_i)
        for episode_i in data_o.episodes:
            offline_episodes.append(episode_i)

    print("expert data: {} [{} episodes]".format(expert_name, len(expert_episodes)))
    data_expert = build_mdpdata_from_episodes(expert_episodes, env)


    print("offline data: {} [{} episodes]".format(offline_name, len(offline_episodes)))
    data_offline = build_mdpdata_from_episodes(offline_episodes, env)


    # print(data_expert.observations.shape)
    # print(data_expert.observations[0])

    return data_expert, data_offline, env







    

    



if __name__ == "__main__":

    import argparse

    '''
    data_name = "hopper-expert-v2"

    print(data_name)

    dataset, env = get_d4rl2d3rl(data_name)

    print(len(dataset.episodes), "episodes")

    count = 0
    for item in dataset.episodes:
        count += len(item)
    
    print(count, ": transitions")
    '''

    # get_offline_imitation_data("hopper-expert-v2", "hopper-random-v2")
    # get_offline_imitation_data("maze2d-umaze-expert-v1", "maze2d-umaze-random-v1")
    get_offline_imitation_data("maze2d-umaze-expert-v1", "maze2d-umaze-v1")


def make_env_only(env_name):
    if 'maze2d' in env_name:

        try:
            env = gym.make(env_name)
        except:

            env_attr = env_name.split('-')

            env_type = env_attr[1]


            reward_type = env_attr[3]

            if "dense" in env_name:
                env_name_gym = f"maze2d-{env_type}-{reward_type}-v1"
            else:
                env_name_gym = f"maze2d-{env_type}-v1"

            env = gym.make(env_name_gym)
    else:
        env = gym.make(env_name)

    return env