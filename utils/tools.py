import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import torch
import utils
import gym
import d4rl

import os
from scipy.ndimage import gaussian_filter
import time
from datasets import load_data
from sklearn.neighbors import KernelDensity
from scipy.interpolate import griddata
import seaborn as sns


@torch.no_grad()
def eval_actor_multi(env_list, actor, device, eval_episodes, seed, seed_offset=19260817):

    actor.eval()


    st_time = time.time()
    time1 = time.time() 


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
def eval_actor(
    env, actor, device, eval_episodes, seed, seed_offset=19260817):
    env.seed(seed + seed_offset)
    # actor.to(device)
    actor.eval()
    episode_rewards = []
    
    init_state = []
    state_list = []
    for iters in range(eval_episodes):
        state, done = env.reset(), False
        episode_reward = 0.0
        init_state.append(state)
        while not done:
            action = actor.select_action(state, device)
            state, reward, done, _ = env.step(action)
            state_list.append(state)
            episode_reward += reward
        episode_rewards.append(episode_reward)
        if( (iters + 1) % (max(int(eval_episodes / 10),1)) == 0):
            print(f'[ evaluation process ] {iters+1}/{eval_episodes} episodes')

    try:
        actor.train()
    except:
        actor.train_mode()
        
    return np.asarray(episode_rewards),np.asarray(init_state),np.asarray(state_list)


def eval_policy(env,actor,device,eval_episodes,seed, data, state_mean, state_std, imgname, show_fig, alpha=0.3):
    
    eval_scores,init_state,state_list = eval_actor(env,actor,device,eval_episodes,seed)
    eval_score = eval_scores.mean()
    normalized_eval_score = env.get_normalized_score(eval_score) * 100.0
    
    
    data_state = data.state * state_std + state_mean
    init_state = init_state * state_std + state_mean
    state_list = state_list * state_std + state_mean

    data_size = data.size

    x_seq, y_seq = data_state[:data_size, 0], data_state[:data_size, 1]
    plt.scatter(x_seq, y_seq, label=f"expert {data_size} transitions", s=1, alpha=0.5)

    data_size = state_list.shape[0]
    x_seq, y_seq = state_list[:, 0], state_list[:, 1]
    plt.scatter(x_seq, y_seq, label=f"eval state {data_size} transitions", s=1, alpha=0.5)
    
    data_size = init_state.shape[0]
    x_seq, y_seq = init_state[:, 0], init_state[:, 1]
    plt.scatter(x_seq, y_seq, label=f"start state {data_size} transitions", s=10, alpha=1.0)
    

    plt.legend(loc = "lower right")
    plt.title(f"eval-policy,score : {normalized_eval_score}")
    # plt.xlim(0,7.5)
    # plt.ylim(0,11)
    plt.savefig(imgname+"_eval-policy.png")
    
    print(f"draw policy evaluation >>> {imgname}"+"_eval-policy.png")

    if show_fig:
        plt.show()
    plt.clf()
    


    print(f"[DRAW] Evaluation over {eval_episodes} episodes: "
    f"{eval_score:.3f} , D4RL score: {normalized_eval_score:.3f}")

map_bounds={
    'large':10,
    'medium':None,
    'umaze':None
}

# large only
# def draw_hotmap(env,actor,device,eval_episodes,seed, state_mean, state_std, imgname, show_fig):
#     eval_scores,init_state,state_list = eval_actor(env,actor,device,eval_episodes,seed)
#     init_state = init_state * state_std + state_mean
#     state_list = state_list * state_std + state_mean
    
#     normalized_eval_score = env.get_normalized_score(eval_scores) * 100.0
#     init_state = (init_state * 10).astype(int)
#     state_list = (state_list * 10).astype(int)
#     maze2d = np.zeros((110,110),dtype=float)
#     for i in range(eval_episodes):
#         idx_x = init_state[i][0]
#         idx_y = init_state[i][1]
#         # print(idx_x,idx_y)
#         maze2d[idx_x][idx_y] = max(normalized_eval_score[i], maze2d[idx_x][idx_y])
#         l = i * 800
#         r = (i+1) * 800
#         for j in range(l,r):
#             idx_x = state_list[j][0]
#             idx_y = state_list[j][1]
#             maze2d[idx_x][idx_y] = max(normalized_eval_score[i], maze2d[idx_x][idx_y])
            
#     fig, ax = plt.subplots()
#     im = ax.imshow(maze2d,cmap='viridis',interpolation='nearest',origin='lower')
#     ax.set_title(imgname[:-4])
#     fig.colorbar(im,ax=ax,label='reward')
    
#     plt.savefig(imgname)
    
#     if show_fig:
#         plt.show()
#     plt.clf()
    

# def generate_uniform_points(M, N):
#     # 从 [0, M) 区间均匀采样 N 个点
#     x_coordinates = np.random.uniform(0, M, N)
#     y_coordinates = np.random.uniform(0, M, N)

#     # 将生成的坐标合并为二维数组
#     points = np.column_stack((x_coordinates, y_coordinates))

#     return points


# def hotmap(points, values, imgname,show_fig=False):
#     values = deepcopy(values)
#     x_coords = points[:,1]
#     y_coords = points[:,0]
#     # print(f"x_coords.max {x_coords.max()}, x_coords.min {x_coords.min()} ")
#     x_coords = np.append(x_coords,[10,0])
#     y_coords = np.append(y_coords,[10,0])
#     values = np.append(values,[0,0])

#     grid_size = 100
#     grid, xedges, yedges = np.histogram2d(x_coords, y_coords, bins=grid_size, weights=values)
#     smoothed_grid = gaussian_filter(grid, sigma=1.0)
#     plt.imshow(smoothed_grid, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], cmap='viridis_r',origin='lower')
#     plt.colorbar()
#     plt.title(imgname)
#     plt.savefig(imgname)
#     if show_fig:
#         plt.show()
#     plt.clf()
# large only

# def weighted_scatter(data, weight, state_mean, state_std, save_name):
#     state = data.state

# def calc_weight_for_databuffer_with_policy(policy, buffer,device):
    
#     l,buffer_size = 0,buffer.size
#     weight = torch.Tensor([])
#     while(l < buffer_size):
#         r = min(l+1000,buffer_size)
#         state_i = torch.Tensor(buffer.state[l:r]).to(device)
#         action_i= torch.Tensor(buffer.action[l:r]).to(device)
        
#         weight_i = torch.sum(policy.get_log_density(state_i,action_i),dim=-1).detach().cpu()
#         weight = torch.cat([weight,weight_i.reshape(-1)],dim=0)
        
#         l += 1000
#     return weight

@torch.no_grad()
def draw_evalmap(env,actor,device,eval_episodes,seed, state_mean, state_std, 
                algos_name,save_name,expert_buffer,seed_offset=19260817):
    
    env.seed(seed + seed_offset)
    actor.to(device)
    
    start_point = []
    rewards = []
    
    # to draw confidence density map
    state_all = []
    prob_all = []
    
    
    start_time = time.time()
    for i in range(eval_episodes):
        state,done = env.reset(),False
        # state = env.reset_to_location(location = points[i])
        start_point.append(state)
        
        episode_reward = 0
        while not done:
            action, log_prob = actor.select_action_with_prob(state, device)
            
            state_all.append(state)
            prob_all.append(np.exp(log_prob))
            
            # action_all.append(action)
            
            state, reward, done, _ = env.step(action)
            episode_reward += reward
        rewards.append(episode_reward)
        
        if((i + 1) % (max(int(eval_episodes / 10),1)) == 0):
            print(f'[ eval process ]: {i+1}/{eval_episodes} | Score Mean {(env.get_normalized_score(np.array(rewards))*100.0).mean()}')
    
    end_time = time.time()
    
    print(f" > Elapsed time : {end_time-start_time} seconds.")
        
        
    start_point = np.array(start_point)
    start_point = start_point * state_std + state_mean
    
    state_all = np.array(state_all)
    state_all = state_all * state_std + state_mean
    
    # prob = get_prob(actor, )
    
    
    
    scatter_map = plt.scatter(start_point[:,0],start_point[:,1], label=f"inital state {start_point.shape[0]} transitions", s=5, alpha=0.5)
    
    plt.colorbar(scatter_map,ticks=np.linspace(0, 300, 10))

    plt.legend(loc="lower right")
    plt.title(algos_name+"_inital_state")
    plt.savefig(save_name+"_inital_state.png")
    plt.xlim(0,7.5)
    plt.ylim(0,11)
    plt.clf()
    print(f"draw policy evaluation >>> {save_name}"+"_inital_state.png")
    
    rewards = np.array(rewards)
    normalized_score = env.get_normalized_score(rewards)*100.0
    print("Score Mean : ",normalized_score.mean(),"  |   Median : ",np.median(normalized_score))
    
    mean_score = [0]
    for i in range(normalized_score.shape[0]):
        mean_score.append(normalized_score[:i+1].mean())
    plt.plot(range(len(mean_score)),mean_score)
    plt.title(algos_name + "_mean_score")
    plt.xlabel("Eval Times")
    plt.ylabel("Normalized Score")
    plt.savefig(save_name+"_mean_score.png")
    plt.clf()
    print(f"draw policy evaluation >>> {save_name}"+"_mean_score.png")
    
    data_state = expert_buffer.state * state_std + state_mean
    
    data_size = expert_buffer.size
    x_seq, y_seq = data_state[:data_size, 0], data_state[:data_size, 1]
    plt.scatter(x_seq, y_seq, label=f"expert {data_size} transitions", s=1, alpha=0.5)
    plt.scatter(start_point[:,0],start_point[:,1],c=normalized_score,cmap='viridis_r', s=50, edgecolors='k')
    plt.colorbar(label='Weights')
    plt.xlim(0,7.5)
    plt.ylim(0,11)
    plt.title(algos_name+f'_Score={normalized_score.mean()}')
    plt.savefig(save_name+"_Score.png")
    plt.clf()
    
    print(f"draw policy evaluation >>> {save_name}"+"_Score.png")
    
    import pandas as pd
    
    weight = deepcopy(normalized_score)
    if(weight.min() < 0):weight = weight - weight.min()
    # weight = 1 / weight
    data = {"x":start_point[:,0],"y":start_point[:,1],"z":weight}
    df = pd.DataFrame(data)
    sns.kdeplot(df,x="x",y="y",weights="z",fill=True,cmap="Blues",cbar=True)
    plt.xlim(0,7.5)
    plt.ylim(0,11)
    plt.title(algos_name+f'_Score={normalized_score.mean()}')
    plt.savefig(save_name+"_Score_density.png")
    plt.clf()
    
    print(f"draw policy evaluation >>> {save_name}_Score_density.png")
    
    # weight = deepcopy(prob_all)
    
    
    # plt.scatter(state_all[:,0],state_all[:,1],c=weight,cmap='viridis_r', s=50, edgecolors='k')
    # plt.colorbar(label='Weights')
    # plt.xlim(0,7.5)
    # plt.ylim(0,11)
    # plt.title(algos_name+f'_Score={normalized_score.mean()}')
    # plt.savefig(save_name+"_confidence.png")
    # plt.clf()
    
    # print(f"draw policy evaluation >>> {save_name}"+"_confidence.png")
    
    # data = {"x":state_all[:,0],"y":state_all[:,1],"z":weight}
    # df = pd.DataFrame(data)
    # sns.kdeplot(df,x="x",y="y",weights="z",fill=True,cmap="Blues",cbar=True)
    # plt.xlim(0,7.5)
    # plt.ylim(0,11)
    # plt.title(algos_name+f'_Score={normalized_score.mean()}')
    # plt.savefig(save_name+"_confidence_density.png")
    # plt.clf()
    # print(f"draw policy evaluation >>> {save_name}_confidence_density.png")
    
    
    
    
    # hotmap(start_point, normalized_score, save_name+"_hotmap.png")
    # eval_scores,init_state,state_list = eval_actor(env,actor,device,eval_episodes,seed)
    # init_state = init_state * state_std + state_mean
    # state_list = state_list * state_std + state_mean
    
    # normalized_eval_score = env.get_normalized_score(eval_scores) * 100.0
    # init_state = (init_state * 10).astype(int)
    # state_list = (state_list * 10).astype(int)

# def draw_distribution(buffer, state_mean, state_std,imgname,show_fig=False):
#     draw_meshgrid(buffer, state_mean, state_std, imgname=imgname)

# def draw_oldaug_newaug(old_buffer, new_buffer,state_mean, state_std,
#                        old_weight, new_weight, ratio, 
#                        imgname, show_fig=False):
#     # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    
#     old_state = old_buffer.state * state_std + state_mean
#     new_state = new_buffer.state * state_std + state_mean
    
#     old_data_size = old_buffer.size
#     new_data_size = new_buffer.size
    
#     old_x, old_y = old_state[:old_data_size, 0], old_state[:old_data_size, 1]
#     new_x, new_y = new_state[:new_data_size, 0], new_state[:new_data_size, 1]
#     # plt.scatter(new_x, new_y, cmap='viridis_r', label=f"New Augment Data", s=1, alpha=0.1)    
    
#     plt.scatter(new_x, new_y, label=f"New Augment Data", s=1, alpha=0.2)    
#     plt.scatter(old_x, old_y, label=f"Old Augment Data", s=1, alpha=0.4)
#     plt.legend(loc='lower right')
#     plt.xlim(0,7.5)
#     plt.ylim(0,11)
#     plt.savefig(imgname + '-coverage.png')
#     plt.title("coverage")
#     plt.clf()
    
#     draw_density(new_buffer, state_mean, state_std,
#                 #   new_weight,
#                   imgname=imgname+'-NewAug_Density.png')
    # # 绘制weights分布图
    # old_x, old_y = old_state[:old_data_size, 0], old_state[:old_data_size, 1]
    # scatter_ax1 = ax1.scatter(old_x, old_y, c=old_weight, cmap='viridis_r',label=f"{old_data_size} transitions", s=1, alpha=0.3, edgecolors='face',linewidths=0.5)
    # ax1.legend(loc="lower right")
    # ax1.set_xlim([0,7.5])
    # ax1.set_ylim([0,11])
    # ax1.set_title(f"Old Augment Data, mean weight: {old_weight.mean().item()}")
    
    
    # new_x, new_y = new_state[:new_data_size, 0], new_state[:new_data_size, 1]
    # scatter_ax2 = ax2.scatter(new_x, new_y, c=new_weight, cmap='viridis_r', label=f"{new_data_size} transitions", s=1, alpha=0.3, edgecolors='face',linewidths=0.5)
    # ax2.legend(loc="lower right")
    # ax2.set_xlim([0,7.5])
    # ax2.set_ylim([0,11])
    # ax2.set_title(f"New Augment Data, mean weight: {new_weight.mean().item()}")
    
    # # ax2.title(f"New Augment Data, {new_data_size} trainsitions")
    # plt.suptitle(f"Threshold = {ratio}")

    # # 调整布局
    # plt.tight_layout()
    # plt.xlim(0,7.5)
    # plt.ylim(0,11)
    # cbar = fig.colorbar(scatter_ax2, ax=[ax1,ax2],label='sample_weight')
    # cbar.mappable.set_clim(0, 1)
    # # cbar.set_cl
    # # colorbar = plt.colorbar(label='Weights')
    # # colorbar.set_clim(vmin=0, vmax=1)
    # plt.savefig(imgname)
    
    # if(show_fig):
    #     plt.show()
    # plt.clf()
    # plt.scatter(x_coord, y_coord, c=offline_weight, cmap='viridis_r', s=1,edgecolors='face', linewidths=0.5)
    # plt.scatter(new_x, new_y, c)

# def draw_density(buffer, state_mean, state_std,
#                 #   weight,
#                   imgname,show_fig=False):
#     sample_size = buffer.size
#     if(buffer.size > 100000):
#         sample_size = 100000
#     state,_,_,_,_,_ = buffer.sample(batch_size=sample_size)
#     state = state.cpu().numpy()
#     state = state * state_std + state_mean
#     # new_state = new_buffer.state * state_std + state_mean
    
#     buffer_size = buffer.size
#     # new_data_size = new_buffer.size
#     x, y = state[:, 0], state[:, 1]
#     point = state[:,:2]
#     print("point.shape : ",point.shape)
#     kde = KernelDensity(kernel='epanechnikov', bandwidth=0.05).fit(point)
#     s = kde.score_samples(point)
#     xx,yy = np.meshgrid(np.linspace(0,7.5,200),np.linspace(0,11,200))
#     # print('point.shape : ',point.shape)
#     # print('s.shape : ',s.shape)
#     zz = griddata(points=point, values=s,xi=(xx,yy),method='linear')
#     a = np.linspace(np.nanmin(zz),np.nanmax(zz),20)
#     myc = plt.contourf(xx,yy,zz,levels=a,alpha=0.75,cmap='viridis_r')
#     cbar = plt.colorbar(myc)
    
#     # plt.contourf(X,Y,weight)
    
#     plt.xlim(0,7.5)
#     plt.ylim(0,11)
#     plt.title(f"Density, point={sample_size}")
#     plt.savefig(imgname+"-Density.png")
#     if(show_fig):
#         plt.show()
#     plt.clf()
    
    
    
    

if __name__ == "__main__":
    state_mean = np.array([ 3.7311654e+00, 5.3069911e+00, -1.5191372e-03, -1.8593170e-03])
    state_std = np.array([1.8069382, 2.5668666, 2.4387012, 2.652832])
    
    import argparse
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--eval_episodes',default=10,type=int)
    parser.add_argument('--load_algos', default='ROMI-BCDP',type=str)
    parser.add_argument('--load_seed', default=0,type=int)
    parser.add_argument('--suffix',default="",type=str)
    parser.add_argument('--log_path',default="",type=str)
    

    
    args = parser.parse_args()
    
    log_path = os.path.join('logs','maze2d-large-expert-v1-[5]-maze2d-large-v1')
    
    log_path = os.path.join(log_path,args.load_algos,f"checkpoints-seed[{args.load_seed}]")
    
    if(args.log_path != ""):
        log_path = args.log_path
    
    figure_save_path = os.path.join(log_path,'figure')
    
    if(args.suffix == ""):
        model_path = os.path.join(log_path,args.load_algos)
        figure_name = os.path.join(figure_save_path,args.load_algos)
    else:
        model_path = os.path.join(log_path,args.load_algos+"_"+args.suffix)
        figure_name=os.path.join(figure_save_path,args.load_algos+"_"+args.suffix)
        
        
    
    os.makedirs(figure_save_path, exist_ok=True)

    expert_data = 'maze2d-large-expert-v1'
    offline_data= 'maze2d-large-v1'

    data_e, data_o, env = load_data.get_offline_imitation_data(expert_data, offline_data, 
                                                          expert_num=5, offline_exp=0)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    replay_buffer_e = utils.ReplayBuffer(state_dim, action_dim, 1000000)
    replay_buffer_o = utils.ReplayBuffer(state_dim, action_dim, 1000000)


    replay_buffer_e.convert_D3RL(data_e)
    replay_buffer_o.convert_D3RL(data_o)
    replay_buffer_e.normalize_states(mean=state_mean, std=state_std)
    replay_buffer_o.normalize_states(mean=state_mean, std=state_std)
    # state_mean = np.array([ 3.7311654e+00, 5.3069911e+00, -1.5191372e-03, -1.8593170e-03])
    # state_std = np.array([1.8069382, 2.5668666, 2.4387012, 2.652832])
    
    
    # env = gym.make(env_name)
    env = utils.wrap_env(env,state_mean=state_mean, state_std=state_std)

    max_action = float(env.action_space.high[0])

    seed = 0
    utils.set_seed(seed, env)


    max_action = float(env.action_space.high[0])
    
    old_buffer = replay_buffer_e
    new_buffer = replay_buffer_o
    old_weight = np.zeros_like(replay_buffer_e.reward)
    new_weight = np.zeros_like(replay_buffer_o.reward)
    ratio = 1.0
    imgname = os.path.join('figure','AugData.png')
    
    # draw_oldaug_newaug(old_buffer, new_buffer, state_mean, state_std,
    #                    old_weight, new_weight, ratio,
    #                    imgname, False)
    # kwargs = {
    #     "state_dim": state_dim, 
    #     "action_dim": action_dim, 
    #     "max_action": max_action,
    # }


    # Initialize policy
    eval_episodes = args.eval_episodes
    
    
    # policy = BCDP(**kwargs)
    # print(f"Load Policy Model from {model_path}")
    # policy.load(filename=model_path)
    
    # draw_evalmap(env, policy.actor, device='cpu', eval_episodes=eval_episodes, seed=seed,
    #         state_mean=state_mean, state_std=state_std, 
    #         algos_name = args.load_algos,
    #         save_name = figure_name,
    #         expert_buffer=replay_buffer_e)   
    # exit(0)   