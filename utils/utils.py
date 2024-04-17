

from typing import Any, Callable, Iterator, List, Optional, Tuple, Union, cast

import gym
import d4rl
import numpy as np
import torch
from typing_extensions import Protocol

from d3rlpy.dataset import Episode, TransitionMiniBatch
from d3rlpy.preprocessing.reward_scalers import RewardScaler
from d3rlpy.preprocessing.stack import StackedObservation

from d3rlpy.metrics.scorer import AlgoProtocol

import random

import pandas as pd
import os

import wandb

import uuid

from copy import deepcopy
import time

WINDOW_SIZE = 1024




def wandb_init(config: dict) -> None:
    wandb.init(
        config=config,
        project=config["project"],
        group=config["group"],
        name=config["name"],
        id=str(uuid.uuid4()),
    )
    wandb.run.save()



def set_seed(
    seed: int, env: Optional[gym.Env] = None, deterministic_torch: bool = False
):
    if env is not None:
        env.seed(seed)
        env.action_space.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)



def wrap_env(
    env: gym.Env,
    state_mean: Union[np.ndarray, float] = 0.0,
    state_std: Union[np.ndarray, float] = 1.0,
    reward_scale: float = 1.0,
) -> gym.Env:
    # PEP 8: E731 do not assign a lambda expression, use a def
    def normalize_state(state):
        return (
            state - state_mean
        ) / state_std  # epsilon should be already added in std.

    def scale_reward(reward):
        # Please be careful, here reward is multiplied by scale!
        return reward_scale * reward

    env = gym.wrappers.TransformObservation(env, normalize_state)
    if reward_scale != 1.0:
        env = gym.wrappers.TransformReward(env, scale_reward)
    return env


def summary_table(evaluations):

        test_mean = np.mean(evaluations)
        test_std = np.std(evaluations)

        num_eval = len(evaluations)

        columns = [str(item) for item in range(num_eval)]
        columns.append("mean")
        columns.append("std")
        evaluations.append(test_mean)
        evaluations.append(test_std)
        table_data = pd.DataFrame(np.array(evaluations).reshape(1, -1), columns=columns)
        return table_data

def summary_table_with_multi_evaluations(evaluations):

    eval_data = np.array(evaluations)

    num_iter = eval_data.shape[0]
    num_eval = eval_data.shape[1]


    columns = ["eval-" + str(item) for item in range(num_eval)]
    columns.append("mean")
    columns.append("std")
    
    test_mean = np.mean(evaluations, axis=1)
    test_std = np.std(evaluations, axis=1)

    evaluations = np.concatenate((evaluations, test_mean.reshape(-1, 1), test_std.reshape(-1, 1)), axis=1)

    table_data = pd.DataFrame(np.array(evaluations).reshape(num_iter, num_eval + 2), columns=columns)

    return table_data




@torch.no_grad()
def eval_actor(
    env, actor, device, eval_episodes, seed, seed_offset=19260817):
    env.seed(seed + seed_offset)
    actor.eval()
    episode_rewards = []
    for iter in range(eval_episodes):
        state, done = env.reset(), False
        episode_reward = 0.0
        while not done:
            action = actor.select_action(state, device)
            state, reward, done, _ = env.step(action)
            episode_reward += reward
        episode_rewards.append(episode_reward)
        
        if(iter % (eval_episodes//5) == 0):
            print(f'[eval process]: {iter}/{eval_episodes}')

    try:
        actor.train()
    except:
        actor.train_mode()
        
    return np.asarray(episode_rewards)


@torch.no_grad()
def eval_actor_fix_start(start_point,state_std, state_mean,
    env, actor, device, eval_episodes, seed, seed_offset=19260817):
    env.seed(seed + seed_offset)
    actor.eval()
    # episode_rewards = []
    # for _ in range(eval_episodes):
    state, done = env.reset(), False
    start_point = start_point.cpu().numpy()
    start_point = start_point * state_std[:2] + state_mean[:2]
    # print('origin_start : ',state)
    state = env.reset_to_location(location = start_point)
    state = (state - state_mean) / state_std
    # print('start_point : ',start_point)
    # print('reset_state : ',state)
    episode_reward = 0.0
    while not done:
        action = actor.select_action(state, device)
        state, reward, done, _ = env.step(action)
        # print('state : ',state)
        episode_reward += reward
        # episode_rewards.append(episode_reward)

    try:
        actor.train()
    except:
        actor.train_mode()
    # exit(0)
    return episode_reward


def evaluate_on_environment(
    env: gym.Env, n_trials: int = 10, epsilon: float = 0.0, render: bool = False
) -> Callable[..., float]:
    """Returns scorer function of evaluation on environment.

    This function returns scorer function, which is suitable to the standard
    scikit-learn scorer function style.
    The metrics of the scorer function is ideal metrics to evaluate the
    resulted policies.

    .. code-block:: python

        import gym

        from d3rlpy.algos import DQN
        from d3rlpy.metrics.scorer import evaluate_on_environment


        env = gym.make('CartPole-v0')

        scorer = evaluate_on_environment(env)

        cql = CQL()

        mean_episode_return = scorer(cql)


    Args:
        env: gym-styled environment.
        n_trials: the number of trials.
        epsilon: noise factor for epsilon-greedy policy.
        render: flag to render environment.

    Returns:
        scoerer function.


    """

    # for image observation
    observation_shape = env.observation_space.shape
    is_image = len(observation_shape) == 3

    def scorer(algo: AlgoProtocol, *args: Any) -> float:
        if is_image:
            stacked_observation = StackedObservation(
                observation_shape, algo.n_frames
            )

        episode_rewards = []
        for _ in range(n_trials):
            observation = env.reset()
            episode_reward = 0.0

            # frame stacking
            if is_image:
                stacked_observation.clear()
                stacked_observation.append(observation)

            while True:
                # take action
                if np.random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    if is_image:
                        action = algo.predict([stacked_observation.eval()])[0]
                    else:
                        action = algo.predict([observation])[0]

                observation, reward, done, _ = env.step(action)
                episode_reward += reward

                if is_image:
                    stacked_observation.append(observation)

                if render:
                    env.render()

                if done:
                    break
            episode_rewards.append(episode_reward)
        return float(np.mean(episode_rewards)), episode_rewards

    return scorer

def save_results(episode_values, mix_data_name, method_name, seed):

    print(episode_values)
    print("{} +/- {} (10 evaluations)".format(np.mean(episode_values), np.std(episode_values)))

    res = pd.DataFrame(episode_values, columns=['return_rewards'])
    experiment_name=f"{method_name}_{mix_data_name}_{seed}"
    
    
    file_dir = os.path.join("d3rlpy_logs", mix_data_name)
    if not os.path.exists(file_dir):
        os.mkdir(file_dir)
    file_dir = os.path.join(file_dir, str(seed))
    if not os.path.exists(file_dir):
        os.mkdir(file_dir)
    
    file_path = os.path.join(file_dir, "results_" + experiment_name + ".csv")
    res.to_csv(file_path)
    
    print("results >>>", file_path)


def summary_table_with_multi_evaluations(evaluations):

    eval_data = np.array(evaluations)

    num_iter = eval_data.shape[0]
    num_eval = eval_data.shape[1]


    columns = ["eval-" + str(item) for item in range(num_eval)]
    columns.append("mean")
    columns.append("std")

    test_mean = np.mean(evaluations, axis=1)
    test_std = np.std(evaluations, axis=1)

    evaluations = np.concatenate((evaluations, test_mean.reshape(-1, 1), test_std.reshape(-1, 1)), axis=1)

    table_data = pd.DataFrame(np.array(evaluations).reshape(num_iter, num_eval + 2), columns=columns)

    return table_data

@torch.no_grad()
def eval_actor_multi(env_list, actor, device, eval_episodes, seed, seed_offset=19260817):

    actor.eval()


    st_time = time.time()


    # episode_rewards = []
    # for i in range(eval_episodes):

    #     env_i = env_list[i]

    #     env_i.seed(seed + seed_offset + i)

    #     state_list = []
    #     action_list = []


    #     state, done = env_i.reset(), False
    #     episode_reward = 0.0

    #     state_list.append(state)

    #     while not done:
    #         action = actor.select_action(state, device)
    #         state, reward, done, _ = env_i.step(action)
    #         episode_reward += reward


    #         state_list.append(state)
    #         action_list.append(action)


    #     # print(state_list)
    #     # print(action_list)

    #     episode_rewards.append(episode_reward)


    # print(episode_rewards)

    time1 = time.time() 
    # print('Time for evaluation: ', time1 - st_time)

    # env.seed(seed + seed_offset)

    obs_list = []

    episode_rewards = np.zeros(eval_episodes)
    not_done_flags = np.ones(eval_episodes)

    for i in range(eval_episodes):

        env_list[i].seed(seed + seed_offset)


        for j in range(i + 1):
            state, done = env_list[i].reset(), False

            # print(state)


        not_done_flags[i] *= (1. - done)

        obs_list.append(state.reshape(1, -1))


    # print(obs_list)

    # exit(0)
    while not_done_flags.sum() > 0:



        state_all = np.array(obs_list).reshape(eval_episodes, -1)

        action_all = actor.select_action_multi(state_all, device)

        # print(state_all[:3])

        # print(action_all)


        new_obs_list = []
        reward_list = []

        for i in range(eval_episodes):

            if not_done_flags[i]:

                # print(obs_list[i])
                state, reward, done, _ = env_list[i].step(action_all[i].reshape(-1))

                reward_list.append(reward)
                new_obs_list.append(state)
                episode_rewards[i] += reward
                not_done_flags[i] = (1. - done)
            else:
                new_obs_list.append(obs_list[i])
                episode_rewards[i] += 0
        obs_list = new_obs_list

        # print(reward_list)



    try:
        actor.train()
    except:
        actor.train_mode()


    # print(episode_rewards)

    time2 = time.time() 
    print("Time taken for evaluation: ", time2 - time1)

    return np.asarray(episode_rewards)


@torch.no_grad()
def eval_actor_multi(env_list, actor, device, eval_episodes, seed, seed_offset=19260817):

    actor.eval()


    st_time = time.time()


    # episode_rewards = []
    # for i in range(eval_episodes):

    #     env_i = env_list[i]

    #     env_i.seed(seed + seed_offset + i)

    #     state_list = []
    #     action_list = []


    #     state, done = env_i.reset(), False
    #     episode_reward = 0.0

    #     state_list.append(state)

    #     while not done:
    #         action = actor.select_action(state, device)
    #         state, reward, done, _ = env_i.step(action)
    #         episode_reward += reward


    #         state_list.append(state)
    #         action_list.append(action)


    #     # print(state_list)
    #     # print(action_list)

    #     episode_rewards.append(episode_reward)


    # print(episode_rewards)

    time1 = time.time() 
    # print('Time for evaluation: ', time1 - st_time)

    # env.seed(seed + seed_offset)

    obs_list = []

    episode_rewards = np.zeros(eval_episodes)
    not_done_flags = np.ones(eval_episodes)

    for i in range(eval_episodes):

        env_list[i].seed(seed + seed_offset)


        for j in range(i + 1):
            state, done = env_list[i].reset(), False

            # print(state)


        not_done_flags[i] *= (1. - done)

        obs_list.append(state.reshape(1, -1))


    # print(obs_list)

    # exit(0)
    while not_done_flags.sum() > 0:



        state_all = np.array(obs_list).reshape(eval_episodes, -1)

        action_all = actor.select_action_multi(state_all, device)

        # print(state_all[:3])

        # print(action_all)


        new_obs_list = []
        reward_list = []

        for i in range(eval_episodes):

            if not_done_flags[i]:

                # print(obs_list[i])
                state, reward, done, _ = env_list[i].step(action_all[i].reshape(-1))

                reward_list.append(reward)
                new_obs_list.append(state)
                episode_rewards[i] += reward
                not_done_flags[i] = (1. - done)
            else:
                new_obs_list.append(obs_list[i])
                episode_rewards[i] += 0
        obs_list = new_obs_list

        # print(reward_list)



    try:
        actor.train()
    except:
        actor.train_mode()


    # print(episode_rewards)

    time2 = time.time() 
    print("Time taken for evaluation: ", time2 - time1)

    return np.asarray(episode_rewards)