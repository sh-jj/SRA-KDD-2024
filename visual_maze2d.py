


import gym

from datasets import load_data
import utils
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
import torch
import time

env_type = "large"



def load_data_env(expert_data, offline_data, expert_num):
    data_e, data_o, env = load_data.get_offline_imitation_data(expert_data, offline_data, 
                                                          expert_num=expert_num, offline_exp=0)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    

    replay_buffer_e = utils.ReplayBuffer(state_dim, action_dim)
    replay_buffer_o = utils.ReplayBuffer(state_dim, action_dim)
    replay_buffer_e.convert_D3RL(data_e)
    replay_buffer_o.convert_D3RL(data_o)


    observations_all = np.concatenate([replay_buffer_e.state, replay_buffer_o.state]).astype(np.float32)

    # print(observations_all.shape)

    state_mean = np.mean(observations_all, 0)
    state_std = np.std(observations_all, 0) + 1e-3

    replay_buffer_e.normalize_states(mean=state_mean, std=state_std)
    replay_buffer_o.normalize_states(mean=state_mean, std=state_std)
    

        
    env = utils.wrap_env(env, state_mean=state_mean, state_std=state_std)




    max_action = float(env.action_space.high[0])

    # Set seeds
    seed = 0
    utils.set_seed(seed, env)


    return replay_buffer_e, replay_buffer_o, env, state_mean, state_std


from sklearn.neighbors import KernelDensity
from scipy.interpolate import griddata

import seaborn as sns
import os

@torch.no_grad()
def eval_actor(
    env, actor, device, eval_episodes, seed, seed_offset=19260817):
    env.seed(seed + seed_offset)
    actor.eval()
    episode_rewards = []
    start_point = []
    for iter in range(eval_episodes):
        state, done = env.reset(), False
        start_point.append([state])
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
        
    return np.asarray(episode_rewards), np.concatenate(start_point,axis=0)


def draw_evalscore(actor,env , buffer, state_mean, state_std, imgname, show_fig=False, device='cpu', xlim=8, ylim=11):
    
    csv_name = imgname[:-4]+".csv"
    
    
    sample_size = buffer.size
    if(buffer.size > 1000):
        sample_size = 1000
    
    # sample_size=50
    # # debug
    # sample_size=10
    # if(os.path.exists(csv_name)):...
        # df = pd.read_csv(csv_name)
    if(False):...
    else:
        
        actor.eval()

        state,_,_,_,_,_ = buffer.sample(batch_size=sample_size)
        state = state.to(device)
        l = 0
        weight = []
        start_time = time.time()
        while(l < sample_size):
            state_i = state[l].to(device)
            # eval_scores = utils.eval_actor(env, policy.actor, device=args.device, eval_episodes=args.eval_episodes, seed=args.seed)
            # start_point = state_i.cpu().numpy()* state_std + state_mean
            weight_i = utils.eval_actor_fix_start(state_i[:2], state_std, state_mean, env, actor, device=device,eval_episodes=1,seed=0)
            weight_i = env.get_normalized_score(weight_i) * 100.0
            weight.append(weight_i)

            l += 1
            if(l % (sample_size // 5) == 0):
                print(f"[eval process] : {l}/{sample_size}")
        # weight, start_point = eval_actor(env,actor,device=device,eval_episodes=sample_size,seed=0)
        end_time = time.time()
        print("Time Used : ", end_time-start_time," seconds")
        
        # print("start_point.shape : ",start_point.shape)
        # print("weight : ",weight.shape)
        
        weight = np.asarray(weight)
        weight = weight - weight.min() + 1
        
        state = state.cpu().numpy()
        state = state * state_std + state_mean
        x,y = state[:,0], state[:,1]
        # start_point = start_point * state_std + state_mean
        # x,y = start_point[:,0],start_point[:,1]
    
        data = {
            "x":x,
            "y":y,
            "z":weight
        }
        
        df = pd.DataFrame(data,columns=["x","y","z"])
        df.to_csv(csv_name)
    
    # print(df.info())
    
    # sns.set_style('white')
    
    # try:
    # p1 = sns.kdeplot(df,x="x",y="y",weights="z",fill=True,cmap='Blues',cbar=True,bw_adjust=0.25)
    # cb = p1.figure.colorbar(p1.collections[0])
    # cb.set_ticks([0.4,0.8,1.2,1.6,2.0,2.4,2.8,3.2,3.6,4.0])
    # except:
    #     print("Fail to draw confidence map : ",imgname)
    # weight = np.exp(weight)
    # umaze
    # plt.xlim(0,4.0)
    # plt.ylim(0,4.0)
    scatter_map = plt.scatter(df['x'],df['y'],c=df['z'],cmap="viridis_r",s=50, edgecolors='k',vmin=0,vmax=300)
    plt.colorbar(scatter_map,ticks=np.linspace(0, 300, 11),label='D4RL normalized score')
    

    # large
    plt.xlim(0,xlim)
    plt.ylim(0,ylim)
    plt.title(f"Score Density, point={sample_size}")
    plt.savefig(imgname)
    if(show_fig):
        plt.show()
    plt.clf()
    

def draw_confidence(actor, buffer, state_mean, state_std,
                    imgname, show_fig=False,device='cpu'):
    csv_name = imgname[:-4]+".csv"
    
    sample_size = buffer.size
    if(buffer.size > 100000):
        sample_size = 100000
    
    if(os.path.exists(csv_name)):
        df = pd.read_csv(csv_name)
    else:
        actor.eval()
        state,_,_,_,_,_ = buffer.sample(batch_size=sample_size)
        # state = state.to(device)
        l = 0
        weight = torch.Tensor([])
        while(l < sample_size):
            r = min(l+1000,sample_size)
            state_i = state[l:r].to(device)
            _,_,action_i = actor(state_i)
            action_i = action_i.to(device)
            
            weight_i = torch.sum(actor.get_log_density(state_i, action_i),dim=-1).detach().cpu()
            weight = torch.cat([weight,weight_i.reshape(-1)],dim=0)
            
            l += 1000
        def sigmoid(x):
            return 1/(1+np.exp(-x))
        weight = sigmoid(weight)
        weight = weight.numpy()
        
        state = state.cpu().numpy()
        state = state * state_std + state_mean
        x,y = state[:,0],state[:,1]
        
        data = {
            "x":x,
            "y":y,
            "z":weight
        }
        csv_name = imgname[:-4]+".csv"
        
        df = pd.DataFrame(data,columns=["x","y","z"])
        df.to_csv(csv_name)
    
    sns.set_style('white')
    
    # try:
    p1 = sns.kdeplot(df,x="x",y="y",weights="z",fill=True,cmap='Blues',cbar=True,bw_adjust=0.25)
    # except:
    #     print("Fail to draw confidence map : ",imgname)
    # weight = np.exp(weight)
    # umaze
    # plt.xlim(0,4.0)
    # plt.ylim(0,4.0)
    
    # large
    plt.xlim(0,7.5)
    plt.ylim(0,10.5)
    plt.title(f"Confidence Density, point={sample_size}")
    plt.savefig(imgname)
    if(show_fig):
        plt.show()
    plt.clf()
    

def draw_density(buffer, state_mean, state_std,
                #   weight,
                  imgname,show_fig=False):
    csv_name = imgname[:-4]+".csv"
    
    sample_size = buffer.size
    if(buffer.size > 10000):
        sample_size = 10000
    target_point = np.array([6,6])
    # if(os.path.exists(csv_name)):...
        # df = pd.read_csv(csv_name)
    if(False):...
    else:
        state_list = []
        delta = 0
        while(len(state_list) < sample_size):
            
            state,_,_,_,_,_ = buffer.sample(batch_size=sample_size)
            for i in range(sample_size):
                point = state[i].cpu().numpy()
                real_point = point * state_std + state_mean
                dist = np.linalg.norm(target_point - real_point[:2])
                if(dist < 0.5):
                    if(delta >= sample_size * 0.1):continue
                    delta += 1
                state_list.append(point.reshape(1,-1))
        state = np.concatenate(state_list,axis=0)
            
        # state = state.cpu().numpy()
        state = state * state_std + state_mean
        # new_state = new_buffer.state * state_std + state_mean
        
        buffer_size = buffer.size
        # new_data_size = new_buffer.size
        x, y = state[:, 0], state[:, 1]
        
        data = {
            "x":x,
            "y":y
        }
        
        df = pd.DataFrame(data,columns=["x","y"])   
        df.to_csv(csv_name)

    # point = state[:,:2]
    sns.set_style('white')
    try:
        p1 = sns.kdeplot(df,x="x",y="y", fill=True,cmap="Blues",cbar=True,bw_adjust=0.8)
    except:
        print("Fail to draw density map : ",imgname)
    # umaze
    # plt.xlim(0,4.0)
    # plt.ylim(0,4.0)
    
    # large
    plt.xlim(0,7)
    plt.ylim(0,7)
    plt.title(f"Density, point={sample_size}")
    plt.savefig(imgname)
    if(show_fig):
        plt.show()
    plt.clf()

def draw_data_with_weight(buffer, state_mean, state_std, imgname, show_fig, alpha=0.3):
    sample_size = buffer.size
    if(buffer.size > 1000):
        sample_size = 1000
    state,_,_,_,_,weight = buffer.sample(batch_size=sample_size)
    state = state.cpu().numpy()
    weight = weight.cpu().numpy()
    state = state * state_std + state_mean
    
    # x_seq, y_seq = state[:data_size, 0], data_state[:data_size, 1]
    x_seq, y_seq = state[:, 0], state[:, 1]
    
    sc = plt.scatter(x_seq, y_seq, c=weight, label=f"{sample_size} transitions", alpha=alpha, cmap='viridis_r', s=25, edgecolors='k')
    plt.colorbar(sc,label="weight")
    plt.legend(loc="lower right")
    plt.xlim(0,7.5)
    plt.ylim(0,10.5)
    plt.title(imgname[:-4])
    plt.savefig(imgname)

    if show_fig:
        plt.show()
    plt.clf()

def draw_data(data, state_mean, state_std, imgname, show_fig):
    data_state = data.state * state_std + state_mean

    data_size = data.size
    # colors = []
    # for i in range(data_size):
    #     if(data.reward[i] < 0.5):
    #         colors.append('b')
    #     else:
    #         colors.append('r')
    x_seq, y_seq = data_state[:data_size, 0], data_state[:data_size, 1]
    plt.scatter(x_seq, y_seq, label=f"{data_size} transitions", s=5, alpha=1, c='blue')
    # plt.legend(loc="lower right")
    
    # umaze
    # plt.xlim(0,4.0)
    # plt.ylim(0,4.0)
    
    # large
    plt.xlim(0,7.5)
    plt.ylim(0,10.5)
    plt.title(imgname[:-4])
    plt.savefig(imgname,dpi=300)

    if show_fig:
        plt.show()
    plt.clf()

def draw_data_2(data1,data2,state_mean,state_std,imgname):
    state1 = data1 * state_std + state_mean
    state2 = data2 * state_std + state_mean
    
    x_seq,y_seq = state1[:,0],state1[:,1]
    plt.scatter(x_seq, y_seq, s = 5,alpha=0.3,c='blue')
    x_seq,y_seq = state2[:,0],state2[:,1]
    plt.scatter(x_seq, y_seq, s = 5,alpha=1,c='red')
    # plt.xlim(0,7.5)
    # plt.ylim(0,10.5)
    plt.title(imgname[:-4])
    plt.savefig(imgname,dpi=300)
    plt.clf()

def draw_data_3(data1=None,data2=None,data3=None,state_mean=0,state_std=1,imgname=None,c1='white',c2='cyan',c3='purple'):
    
    if(data1 is not None):
        state1 = data1 * state_std + state_mean
        x_seq,y_seq = state1[:,0],state1[:,1]
        plt.scatter(x_seq, y_seq, s = 5,alpha=0.3,c=c1)
    if(data2 is not None):
        state2 = data2 * state_std + state_mean
        x_seq,y_seq = state2[:,0],state2[:,1]
        plt.scatter(x_seq, y_seq, s = 5,alpha=1,c=c2)
    if(data3 is not None):
        state3 = data3 * state_std + state_mean
        x_seq,y_seq = state2[:,0],state2[:,1]
        plt.scatter(x_seq, y_seq, s = 5,alpha=1,c=c3)
    



    # plt.xlim(0,8)
    # plt.ylim(0,11)
    plt.xlim(0,7.5)
    plt.ylim(0,10.5)
    plt.title(imgname[:-4])
    plt.savefig(imgname,dpi=300,transparent=True)
    plt.clf()

def draw_action(data, state_mean, state_std, imgname):
    data_state = data.state * state_std + state_mean

    data_size = data.size

    fig = plt.figure()

    action_value = (data.action[: data_size, 0] + 1 ) * 30 + (data.action[: data_size, 1] + 1) * 10

    action_angle = np.arctan(data.action[: data_size, 1] / (data.action[: data_size, 0] + 1e-3)) # arctan(y/x)

    minx, maxx = action_angle.min(), action_angle.max()

    print("angle range: ", minx, maxx)

    action_angle = (action_angle - (-np.pi / 2)) / (np.pi) * 100


    x_seq, y_seq = data_state[:data_size, 0], data_state[:data_size, 1]
    plt.scatter(x_seq, y_seq, label=f"{data_size} transitions", s=5, alpha=0.3, c=action_angle, cmap="brg")
    plt.legend(loc="lower right")

    plt.savefig(imgname)

    plt.show()

def load_info(env_type, expert_num):
    if env_type == "large":
        expert_data = "maze2d-large-expert-v1"
        offline_data = "maze2d-large-v1"
        state_mean = np.array([3.7331274, 5.3092222, -0.0008, -0.001279705])
        state_std  = np.array([1.8081381, 2.5674472, 2.4380245, 2.652119 ])

        aug_data_path = f"logs/maze2d-large-expert-v1-[{expert_num}]/ROIL-AUG/checkpoints-seed[0]/reverse_QL-3steps-100000.pt"
        
        aug_data_path = f"logs/maze2d-large-expert-v1-[{expert_num}]/ROIL-AUG/checkpoints-seed[0]/vae_rollout-3steps-reverse_QL-100000.pt"
        aug_data_path = f"logs/maze2d-large-expert-v1-[{expert_num}]/ROIL-AUG/checkpoints-seed[0]/vae_rollout-3steps-filterInD-reverse_QL-100000.pt"
        aug_data_path = f"logs/maze2d-large-expert-v1-[{expert_num}]/ROIL-AUG/checkpoints-seed[0]/uniform_rollout-5steps-filterInD-KNN-reverse_QL-100000.pt"
        aug_data_path = f"logs/maze2d-large-expert-v1-[{expert_num}]/ROIL-AUG/checkpoints-seed[0]/uniform_rollout-5steps-filterInD-KNN-reverse_QL-100000.pt"
        aug_data_path = f"logs/maze2d-large-expert-v1-[{expert_num}]/ROIL-AUG/checkpoints-seed[0]/uniform_rollout-5steps-filterInD-KNN-reverse_QL-1000000.pt"
        aug_data_path = f"logs/maze2d-large-expert-v1-[{expert_num}]/ROIL-AUG/checkpoints-seed[0]/uniform_rollout-5steps-filterInD-KNN-reverse_QL-100000.pt"
    
    
    if env_type == "umaze":
        expert_data = "maze2d-umaze-expert-v1"
        offline_data = "maze2d-umaze-v1"
        state_mean = np.array( [ 1.9303375, 2.3781407, -0.00424484, 0.02378478])
        state_std  = np.array([0.90039796, 0.70834637, 2.220818, 2.468808 ])

        aug_data_path = f"logs/maze2d-umaze-expert-v1-[{expert_num}]/ROIL-AUG/checkpoints-seed[0]/reverse_QL-3steps-100000.pt"
        aug_data_path = f"logs/maze2d-umaze-expert-v1-[{expert_num}]/ROIL-AUG/checkpoints-seed[0]/uniform_rollout-10steps-reverse_QL-100000.pt"
        aug_data_path = f"logs/maze2d-umaze-expert-v1-[{expert_num}]/ROIL-AUG/checkpoints-seed[0]/uniform_rollout-10steps-filterInD-KNN-reverse_QL-100000.pt"

    if env_type == "medium":
        expert_data = "maze2d-medium-expert-v1"
        offline_data = "maze2d-medium-v1"
        state_mean = np.array( [3.5111730, 3.4797537, -0.00081320840 -0.00087014376])
        state_std  = np.array([1.3348458, 1.5203042, 2.3420413, 2.3728395])

        aug_data_path = f"logs/maze2d-medium-expert-v1-[{expert_num}]/ROIL-AUG/checkpoints-seed[0]/reverse_QL-5steps-100000.pt"
        
        aug_data_path = f"logs/maze2d-medium-expert-v1-[{expert_num}]/ROIL-AUG/checkpoints-seed[0]/uniform_rollout-10steps-filterInD-KNN-reverse_QL-100000.pt"


    return expert_data, offline_data, state_mean, state_std, aug_data_path


if __name__ == "__main__":


    env_type = "medium"
    expert_num = 10
    expert_data, offline_data, state_mean, state_std, aug_data_path = load_info(env_type, expert_num)

    exp_data, off_data, env, state_mean, state_std = load_data_env(expert_data, offline_data, expert_num)
    aug_data = utils.ReplayBuffer(state_dim=4, action_dim=2)
    aug_data = aug_data.load(aug_data_path)
    print("load augmented data <<< ", aug_data_path)

    print("state-mean: ", state_mean, "state-std: ", state_std)

    import matplotlib.pyplot as plt

    print(aug_data.size)
    print(aug_data.state)

    # draw_data(off_data, state_mean, state_std, f"images/maze2d-{env_type}-exp{expert_num}-off.png")
    # draw_data(aug_data, state_mean, state_std, f"images/maze2d-{env_type}-exp{expert_num}-aug.png")
    # draw_data(exp_data, state_mean, state_std, f"images/maze2d-{env_type}-exp{expert_num}-exp.png")

    # draw_action(off_data, state_mean, state_std, f"images/maze2d-{env_type}-exp{expert_num}-action-off.png")
    draw_action(aug_data, state_mean, state_std, f"images/maze2d-{env_type}-exp{expert_num}-action-aug.png")
    draw_action(exp_data, state_mean, state_std, f"images/maze2d-{env_type}-exp{expert_num}-action-exp.png")


