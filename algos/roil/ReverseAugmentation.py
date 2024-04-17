


import numpy as np
from numpy.linalg import norm
from functools import partial
import torch
import d4rl
import copy
import sys
from visual_maze2d import draw_data, draw_density, draw_data_with_weight

from algos.weighted_sample import StateDiscriminator,SADiscriminator
from copy import deepcopy

sys.path.append('..')

import matplotlib.pyplot as plt
import os
import utils

    # l, size = 0, len(state_data)

    # weight = torch.Tensor([])
    # while l < size:             
    #     state_i = state_data[l: min(l+1000,size), :]
    #     state_i = torch.Tensor(state_i).to(device)
    #     weight_i = weight_model.weight_sample(state_i).detach().cpu()
    #     weight = torch.cat([weight, weight_i.reshape(-1)], dim=0)

    #     l += 1000

    # return weight


def calc_weight_for_databuffer_with_policy(policy, buffer,device):
    
    l,buffer_size = 0,buffer.size
    weight = torch.Tensor([])
    while(l < buffer_size):
        r = min(l+1000,buffer_size)
        state_i = torch.Tensor(buffer.state[l:r]).to(device)
        _,_,action_i= policy(state_i)
        action_i = action_i.to(device)
        # action_i= torch.Tensor(action_i).to(device)
        
        weight_i = torch.sum(policy.get_log_density(state_i,action_i),dim=-1).detach().cpu()
        weight = torch.cat([weight,weight_i.reshape(-1)],dim=0)
        
        l += 1000
    return weight
    


def Filter_Sample_State_with_Policy(policy, now_buffer, num_preserved, 
                        state_mean, state_std,
                        weight_sample=False,
                        device='cpu',log_path=None,step=None,detail_info=True):
    # print("nowbuffer.size : ",now_buffer.size)
    weight = calc_weight_for_databuffer_with_policy(policy, now_buffer, device)
    # print("weight.shape : ",weight.shape)
    def print_quantile(array:torch.Tensor):
        info = "[ Info ]: "
        for i in range(11):
            quan = i/10
            info += f"{quan:.1f}: [{array.quantile(quan):.3f}], "
        print(info)
    if(detail_info):
        print(">>> weight info <<<")
        print_quantile(weight)
    
    # sorted_weight,index = weight.sort()
    
    # clip_min = int(now_buffer.size * 0.1)
    # clip_max = int(now_buffer.size * 0.9) # 去掉极端值
    # index = index[clip_min:clip_max]
    # _, index = index.sort()
    # weight = weight[index]
    # indices = indices[:num_preserved] # 取log prob 最小的几个,这些和expert trajectory距离最远
    # select_weight = weight[indices]
    
    # print(">>> select weight info <<<")
    # print_quantile(select_weight)
    
    num_preserve = num_preserved
    if(detail_info):
        print(" num_preserved : ",num_preserve, "   buffer.size : ",now_buffer.size)
    
    # indices = weight_index[:num_preserve]
    # select_weight = weight[indices]
    if(weight_sample):
        if(detail_info):
            print("> [ Use Weight Sample Method ]")
        index = np.array(range(now_buffer.size))
        prob =( 1 / (torch.exp(weight) + 1e-5) ).numpy()
        # prob = (1 / (weight+1e-3)).numpy()
        # prob = np.exp(prob) / sum( np.exp(prob) )
        prob = prob / sum(prob)
        # print("index.shape : ",index.shape)
        # print("prob.shape : ",prob.shape)
        indices = np.random.choice(a=index, size=num_preserve, replace=False, p=prob)
        select_weight = weight[indices]
    else:
        if(detail_info):
            print("> [ Use Hard Select Method ] ")
        select_weight, indices = torch.topk(weight, num_preserve, largest=False)
    
    if(detail_info):
        print('>>> Select buffer weight <<<')
        print_quantile(select_weight)
    # print(f'Selected state weight : Min {select_weight.min()}, Q1 {torch.quantile(select_weight, 0.25).item()}, Med {select_weight.median()}, Q2 {torch.quantile(select_weight, 0.75).item()}, Max {select_weight.max()}, Shape: {select_weight.shape}, Sum: {select_weight.sum()}')
    
    def set_buffer_with_index(buffer, ind):
        if(isinstance(ind,torch.Tensor)):
            ind = ind.cpu().numpy()
        buffer.state = buffer.state[ind]
        buffer.action= buffer.action[ind]
        buffer.next_state= buffer.next_state[ind]
        buffer.reward=buffer.reward[ind]
        buffer.not_done=buffer.not_done[ind]
        buffer.flag=buffer.flag[ind]
        buffer.size = len(ind)
        return buffer
    
    now_buffer = set_buffer_with_index(now_buffer, indices)
    return now_buffer



def model_rollout_with_policy_filter(rbc_policy, fake_env, 
                              dataset_replay_buffer, replay_buffer, state_mean, state_std,
                              rollout_length=3, rollout_batch_size=1000, max_rollout_size=1e6,
                              is_uniform=True, OODdet=None, device='cpu', log_path=None,
                              weight_sample=False,policy=None,detail_draw=True):
    
    if(is_uniform):
        print("Rollout Action with uniform")
    else:
        print("Rollout Action with reverseBC")
    
    every_step_rollout_size_ = max_rollout_size // rollout_length
    
    every_step_rollout_size = every_step_rollout_size_ * 2  # to rollout aug_size transition 
                                                            # *5 means sample enough transition to select, only sample 20% data
    
    
    sample_buffer = deepcopy(dataset_replay_buffer) # start point
    
    init_step = 0

    if(detail_draw):
        print("---------------------------------------")
        print("every_step_rollout_size : ",every_step_rollout_size)
    for i in range(rollout_length):

        if(detail_draw):
            print(f"=========[ Model Rollout Step {i}/{rollout_length} ]=========")
        
        temp_buffer = utils.ReplayBuffer(state_dim=replay_buffer.state.shape[1],
                                    action_dim=replay_buffer.action.shape[1],
                                    max_size = every_step_rollout_size * 2,
                                    device=device)
        
        while(temp_buffer.size < every_step_rollout_size):
            state, action, next_state, reward, not_done, _ = sample_buffer.sample(rollout_batch_size)
            next_observations = state.cpu().data.numpy()
            real_batch_size = next_observations.shape[0]
            
            if( is_uniform ):
                action_size = (real_batch_size, rbc_policy.action_dim)
                actions = np.random.uniform(low=-1,high=1,size=action_size)
            else:
                actions = rbc_policy.select_action(next_observations)
                
            observations, rewards, dones, _ = fake_env.step(next_observations, actions)
            
            for j in range(real_batch_size):

                # try:
                temp_buffer.add(observations[j], actions[j], next_observations[j], reward=0, done=0)
                # except:
                #     print('observations.shape : ',observations.shape)
                #     print('temp_buffer.state.shape : ',temp_buffer.state.shape)
        # print(f"[ Model Rollout Step {i}/{rollout_length} ]")
        if(detail_draw):
            print("Rollout Length : ",temp_buffer.size)
            print('select state from latest rollout transition')
        
        if(log_path and detail_draw):
            draw_data(temp_buffer, state_mean, state_std,os.path.join(log_path,f"Rollout_{i}_onestep_state.png"), show_fig=False)
            draw_density(temp_buffer, state_mean, state_std,os.path.join(log_path,f"Rollout_{i}_onestep_state_density.png"), show_fig=False)
            
            print('draw >>> ',os.path.join(log_path,f"Rollout_{i}_onestep_state.png"))
        
        
        
        # def calculate_action_with_policy(buffer, actor, device='cpu'):
        #     batch_size = 1000
        #     l = 0
        #     actions = []
            
        #     actor.eval()
        #     while(l < buffer.size):
        #         r = min(l+batch_size,buffer.size)
        #         state = torch.FloatTensor(buffer.state[l:r]).to(device)
        #         _,_,action = actor(state)
        #         actions.append(action.detach())
        #         l += batch_size
        #     actor.train()

        #     actions = torch.cat(actions,dim=0)
        #     actions = actions.cpu().numpy()
        #     return actions
        
        # try: # Persudo action label, not useful
        #     temp_buffer.action = calculate_action_with_policy(temp_buffer, policy.actor,device = device)
        # except:
        #     temp_buffer.action = calculate_action_with_policy(temp_buffer, policy.policy,device = device)
        
        if(i >= init_step):# preserve ratio
            
            # try:
            temp_buffer = Filter_Sample_State_with_Policy(policy.actor, temp_buffer, every_step_rollout_size_, state_mean, state_std,
                                                weight_sample=weight_sample, 
                                                device=device,log_path=log_path,step=i,detail_info=detail_draw)
            # except: # special for dwbc
            #     temp_buffer = Filter_Sample_State_with_Policy(policy.policy, temp_buffer, every_step_rollout_size_, state_mean, state_std,
            #                         weight_sample=weight_sample, 
            #                         device=device,log_path=log_path,step=i,detail_info=detail_draw)
            
            
        sample_buffer = deepcopy(temp_buffer)
        sample_buffer.size = min(sample_buffer.size, every_step_rollout_size_)
        replay_buffer = replay_buffer.extend(sample_buffer)
        
        if(detail_draw):
            print("Select Length : ",sample_buffer.size)
            print("Total Rollout Length : ",replay_buffer.size)
        
        if(log_path and detail_draw):
            draw_data(replay_buffer, state_mean, state_std, os.path.join(log_path,f"Rollout_{i}_all_state.png"), show_fig=False)
            draw_density(replay_buffer, state_mean, state_std, os.path.join(log_path,f"Rollout_{i}_all_state_density.png"))
            print('draw >>> ',os.path.join(log_path,f"Rollout_{i}_all_state.png"))
            
            if(i >= init_step ):
                draw_data(sample_buffer, state_mean, state_std, os.path.join(log_path,f"Rollout_{i}_onestep_selected_state.png"), show_fig=False)
                draw_density(sample_buffer, state_mean, state_std, os.path.join(log_path,f"Rollout_{i}_onestep_selected_state_density.png"))
            
                print('draw >>> ',os.path.join(log_path,f"Rollout_{i}_onestep_selected_state.png"))
            
        # draw_data(replay_buffer_e, state_mean, state_std, , show_fig=args.visual_data)
    
    
    draw_data(replay_buffer, state_mean, state_std, log_path+f"/Rollout_all_state.png",show_fig=False)  
    draw_density(replay_buffer, state_mean, state_std, log_path+f"/Rollout_all_state_density.png",show_fig=False)  

    print("draw >>> ",log_path+f"/Rollout_all_state.png")
    
    return replay_buffer




def calc_weight_for_databuffer(state_data, weight_model, 
                               title="weight on data", show_fig=True, no_sort=False,
                               device='cpu'):
    l, size = 0, len(state_data)

    weight = torch.Tensor([])
    while l < size:             
        state_i = state_data[l: min(l+1000,size), :]
        state_i = torch.Tensor(state_i).to(device)
        weight_i = weight_model.weight_sample(state_i).detach().cpu()
        weight = torch.cat([weight, weight_i.reshape(-1)], dim=0)

        l += 1000

    return weight

    if no_sort:
        return weight

    weight, _ = weight.sort()
    print("mean weight on data: ", weight.mean())

    print("weight >= [X]: [X] samples")
    stat_list = []
    for i in range(10):
        stat_list.append((i / 10, (weight >= i / 10).sum().item()))
    print(stat_list)
    
    # if show_fig:
        
        # import matplotlib.pyplot as plt
        # plt.plot(range(len(weight)), weight)
        # plt.savefig(os.path.join(args.log_path, title + ".png"))
        # plt.title(title)
        # plt.show()
    
    return weight


def Filter_Sample_State(prestep_buffer, now_buffer, num_preserved, 
                        state_mean, state_std,
                        weight_sample=False,
                        training_steps=10000,eval_freq=2000,
                        batch_size=256,
                        device='cpu',log_path=None,step=None):
    state_dim = prestep_buffer.state.shape[1]
    
    kwargs = {
        "state_dim": state_dim, 
        "lr": 1e-4, 
        "device": device,
    }
    
    print("kwargs of state learning: ", kwargs)
    
    state_dis_model = StateDiscriminator(**kwargs)
    
    training_iters = 0

    while training_iters < training_steps:

        log_dict = state_dis_model.train(prestep_buffer, now_buffer, iterations=eval_freq, batch_size=batch_size)

        training_iters += eval_freq
        
        print(f"Training iterations: {training_iters}/{training_steps}:")


        dis_loss = np.mean(log_dict["dis_loss"])
        accuracy = np.mean(log_dict["accuracy"])

        print("dis_loss: ", dis_loss, "accuracy: ", accuracy)
        
    weight = calc_weight_for_databuffer(now_buffer.state[:now_buffer.size], state_dis_model,device=device)
    pre_weight = calc_weight_for_databuffer(prestep_buffer.state[:prestep_buffer.size], state_dis_model, device=device)
    
    # print("Weight.shape : ",weight.shape)
    # print("now_buffer.size : ",now_buffer.size)
    
    # now_buffer.flag = weight
    # pre_weight.flag = weight
    
    weight_sort, weight_index = weight.sort()
    preweight_sort,_ = pre_weight.sort()
    
    plt.plot(range(len(weight_sort)), weight_sort, label='now step')
    plt.title(f"Rollout_{step}_step_neg_weight.png")
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(log_path, f"Rollout_{step}_step_neg_weight.png"))
    plt.clf()
    
    now_buffer.flag = weight
    
    draw_data_with_weight(now_buffer, state_mean, state_std, 
                          imgname=os.path.join(log_path, f"Rollout_{step}_neg_weight_dist.png"), show_fig=False)
    # plt.scatter()
    
    plt.plot(range(len(preweight_sort)), preweight_sort, label='last step')
    plt.title(f"Rollout_{step}_step_pos_weight")
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(log_path, f"Rollout_{step}_step_pos_weight.png"))
    plt.clf()
    
    prestep_buffer.flag = pre_weight
    
    draw_data_with_weight(prestep_buffer, state_mean, state_std, 
                    imgname=os.path.join(log_path, f"Rollout_{step}_pos_weight_dist.png"), show_fig=False)
    
    # plt.show()
    def print_quantile(array:torch.Tensor):
        info = "[ Info ]: "
        for i in range(11):
            quan = i/10
            info += f"{quan:.1f}: [{array.quantile(quan):.3f}], "
        print(info)
    print(">>> Rollout buffer (Negetive class) weight <<<")
    print_quantile(weight)
    print(">>> Positive buffer (Positive class) weight <<<")
    print_quantile(pre_weight)
    # print(f'Rollout state weight : Min {weight.min()}, Q1 {torch.quantile(weight, 0.25).item()}, Med {weight.median()}, Q2 {torch.quantile(weight, 0.75).item()}, Max {weight.max()}, Shape: {weight.shape}, Sum: {weight.sum()}')
    # print(f'positive buffer weight : Min {pre_weight.min()}, Q1 {torch.quantile(pre_weight, 0.25).item()}, Med {pre_weight.median()}, Q2 {torch.quantile(pre_weight, 0.75)}, Max {pre_weight.max()}, Shape: {pre_weight.shape}, Sum: {pre_weight.sum()}')
    num_preserve = num_preserved
    print(" num_preserved : ",num_preserve, "   buffer.size : ",now_buffer.size)
    
    # indices = weight_index[:num_preserve]
    # select_weight = weight[indices]
    if(weight_sample):
        print("> [ Use Weight Sample Method ]")
        index = np.array(range(now_buffer.size))
        prob = (1 - weight).numpy()
        # prob = np.exp(prob) / sum( np.exp(prob) )
        prob = prob / sum(prob)
        # print("index.shape : ",index.shape)
        # print("prob.shape : ",prob.shape)
        indices = np.random.choice(a=index, size=num_preserve, replace=False, p=prob)
        select_weight = weight[indices]
    else:
        print("> [ Use Hard Select Method ] ")
        select_weight, indices = torch.topk(weight, num_preserve, largest=False)
    
    print('>>> Select buffer weight <<<')
    print_quantile(select_weight)
    # print(f'Selected state weight : Min {select_weight.min()}, Q1 {torch.quantile(select_weight, 0.25).item()}, Med {select_weight.median()}, Q2 {torch.quantile(select_weight, 0.75).item()}, Max {select_weight.max()}, Shape: {select_weight.shape}, Sum: {select_weight.sum()}')
    
    def set_buffer_with_index(buffer, ind):
        if(isinstance(ind,torch.Tensor)):
            ind = ind.cpu().numpy()
        buffer.state = buffer.state[ind]
        buffer.action= buffer.action[ind]
        buffer.next_state= buffer.next_state[ind]
        buffer.reward=buffer.reward[ind]
        buffer.not_done=buffer.not_done[ind]
        buffer.flag=buffer.flag[ind]
        buffer.size = len(ind)
        return buffer
    
    now_buffer = set_buffer_with_index(now_buffer, indices)
    return now_buffer
        # if args.mode != "debug":
        #     wandb.log({
        #             "dis_loss": dis_loss, 
        #             "accuracy": accuracy, 
        #             },
        #         step=training_iters,
        #     )



def model_rollout_with_filter(rbc_policy, fake_env, 
                              dataset_replay_buffer, replay_buffer, state_mean, state_std,
                              rollout_length=3, rollout_batch_size=1000, max_rollout_size=1e6,
                              ratio=1.0,
                              is_uniform=True, OODdet=None, device='cpu', log_path=None,
                              weight_sample=False,rollout_filter_with_base=False):
    
    every_step_rollout_size_ = max_rollout_size // rollout_length
    
    every_step_rollout_size = int(every_step_rollout_size_ / ratio) * 2 # to rollout aug_size transition 
                                                                        # *2 means sample enough transition to select
    
    
    sample_buffer = deepcopy(dataset_replay_buffer) # start point
    
    init_step = 0
    # temp_buffer = utils.ReplayBuffer(state_dim=sample_buffer.state.shape[1],
    #                                  action_dim=sample_buffer.action.shape[1],
    #                                  max_size = every_step_rollout_size * 2,
    #                                  device=device)
    
    print("---------------------------------------")
    print("every_step_rollout_size : ",every_step_rollout_size)
    for i in range(rollout_length):

        print(f"======[ Model Rollout Step {i}/{rollout_length} ]======")
        
        temp_buffer = utils.ReplayBuffer(state_dim=sample_buffer.state.shape[1],
                                    action_dim=sample_buffer.action.shape[1],
                                    max_size = every_step_rollout_size * 3,
                                    device=device)
        
        while(temp_buffer.size < every_step_rollout_size):
            state, action, next_state, reward, not_done, _ = sample_buffer.sample(rollout_batch_size)
            next_observations = state.cpu().data.numpy()
            
            if( is_uniform ):
                action_size = (rollout_batch_size, rbc_policy.action_dim)
                actions = np.random.uniform(low=-1,high=1,size=action_size)
            else:
                actions = rbc_policy.select_action(next_observations)
                
            observations, rewards, dones, _ = fake_env.step(next_observations, actions)
            
            for j in range(rollout_batch_size):

                try:
                    temp_buffer.add(observations[j], actions[j], next_observations[j], reward=0, done=0)
                except:
                    print('observations.shape : ',observations.shape)
                    print('temp_buffer.state.shape : ',temp_buffer.state.shape)
        # print(f"[ Model Rollout Step {i}/{rollout_length} ]")
        print("Rollout Length : ",temp_buffer.size)
        print('select state from latest rollout transition')
        
        if(log_path):
            draw_data(temp_buffer, state_mean, state_std,os.path.join(log_path,f"Rollout_{i}_onestep_state.png"), show_fig=False)
            draw_density(temp_buffer, state_mean, state_std,os.path.join(log_path,f"Rollout_{i}_onestep_state_density.png"), show_fig=False)
            
            print('draw >>> ',os.path.join(log_path,f"Rollout_{i}_onestep_state.png"))
        
        if(i >= init_step):# preserve ratio
            # if(replay_buffer.size == 0):
            #     temp_buffer = Filter_Sample_State(sample_buffer, temp_buffer, ratio, weight_sample=weight_sample, device=device,log_path=log_path,step=i)
            # else:
            #     temp_buffer = Filter_Sample_State(replay_buffer, temp_buffer, ratio, weight_sample=weight_sample, device=device,log_path=log_path,step=i) 
            training_steps = 1000
            eval_freq = 200
            if(rollout_filter_with_base):
                temp_buffer = Filter_Sample_State(dataset_replay_buffer, temp_buffer, every_step_rollout_size_, state_mean, state_std,
                                                  weight_sample=weight_sample,
                                                  training_steps=training_steps, eval_freq=eval_freq, 
                                                  device=device,log_path=log_path,step=i)
            else:
                temp_buffer = Filter_Sample_State(sample_buffer, temp_buffer, every_step_rollout_size_, state_mean, state_std,
                                                  weight_sample=weight_sample, 
                                                  training_steps=training_steps, eval_freq=eval_freq, 
                                                  device=device,log_path=log_path,step=i)
            
            
        sample_buffer = deepcopy(temp_buffer)
        sample_buffer.size = min(sample_buffer.size, every_step_rollout_size_)
        replay_buffer = replay_buffer.extend(sample_buffer)
        
        print("Select Length : ",sample_buffer.size)
        print("Total Rollout Length : ",replay_buffer.size)
        
        if(log_path):
            draw_data(replay_buffer, state_mean, state_std, os.path.join(log_path,f"Rollout_{i}_all_state.png"), show_fig=False)
            draw_density(replay_buffer, state_mean, state_std, os.path.join(log_path,f"Rollout_{i}_all_state_density.png"))
            print('draw >>> ',os.path.join(log_path,f"Rollout_{i}_all_state.png"))
            
            if(i >= init_step ):
                draw_data(sample_buffer, state_mean, state_std, os.path.join(log_path,f"Rollout_{i}_onestep_selected_state.png"), show_fig=False)
                draw_density(sample_buffer, state_mean, state_std, os.path.join(log_path,f"Rollout_{i}_onestep_selected_state_density.png"))
            
                print('draw >>> ',os.path.join(log_path,f"Rollout_{i}_onestep_selected_state.png"))
            
        # draw_data(replay_buffer_e, state_mean, state_std, , show_fig=args.visual_data)
        
    
    return replay_buffer

# def model_rollout_with_filter(rbc_policy, fake_env, dataset_replay_buffer, replay_buffer, args, rollout_length=3, rollout_batch_size=1000, is_uniform=False, OODdet=None):
#     sample_buffer = deepcopy(dataset_replay_buffer)
#     temp_buffer = utils.ReplayBuffer(state_dim=fake_env.model.state_dim,
#                                      action_dim=fake_env.model.action_dim,
#                                      max_size=args.buffer_size, 
#                                      device=args.device)
#     if( not args.is_forward_rollout):
        
#         for i in range(rollout_length):
#             state, action, next_state, reward, not_done, _ = sample_buffer.sample(rollout_batch_size)
#             next_observations = state.cpu().data.numpy()

#             if( is_uniform ):
#                 action_size = (rollout_batch_size, rbc_policy.action_dim)
#                 actions = np.random.uniform(low=-1,high=1,size=action_size)
#             else:
#                 actions = rbc_policy.select_action(next_observations)
                
#             observations, rewards, dones, _ = fake_env.step(next_observations, actions)
            
#             rollout_batch_size = next_observations.shape[0]
            
#             for j in range(rollout_batch_size):
#                 if i == 0:      # this state-action will lead to expert-state
#                     temp_buffer.add(observations[j], actions[j], next_observations[j], reward=1, done=0)
#                 else:
#                     temp_buffer.add(observations[j], actions[j], next_observations[j], reward=0, done=0)
                    
#             temp_buffer = Filter_Sample_State(sample_buffer,temp_buffer,args)
#             replay_buffer.extend(temp_buffer)
#             sample_buffer = temp_buffer
#             temp_buffer.clear()
#             print('Sample_Buffer_size : ',sample_buffer.size)
#             print('Temp_Buffer_size : ',temp_buffer.size)
                    
#             next_observations = observations
        
            
    # state, action, next_state, reward, weight, not_done = dataset_replay_buffer.sample(rollout_batch_size)


def model_rollout_select(rbc_policy, fake_env, dataset_replay_buffer, replay_buffer, args, rollout_length=3, rollout_batch_size=1000, is_uniform=False, OODdet=None):

    print("Artificial Control! goal position start point is less then 1%")
    
    # state, action, next_state, reward, weight, not_done = dataset_replay_buffer.sample(rollout_batch_size)
    state_list = []
    
    reward_1 = 0
    reward_0 = 0
    while(len(state_list) < rollout_batch_size):
        state, action, next_state, reward, weight, not_done = dataset_replay_buffer.sample(rollout_batch_size)
 
        for i in range(rollout_batch_size):
            if reward[i] < 0.5:
                reward_0 += 1
                state_list.append(state[i])
            else:
                reward_1 += 1
                if(reward_1 >= rollout_batch_size * 0.01):continue
                state_list.append(state[i])
            if(len(state_list) == rollout_batch_size):break

    state = torch.stack(state_list)

        

    if not args.is_forward_rollout:     # reverse augmentation
        
        next_observations = state.cpu().data.numpy()

        rollout_batch_size = rollout_batch_size
        for i in range(rollout_length):

            if is_uniform:
                action_size = (rollout_batch_size, rbc_policy.action_dim)
                actions = np.random.uniform(low=-1, high=1, size=action_size)
            else:
                actions = rbc_policy.select_action(next_observations)
            observations, rewards, dones, _ = fake_env.step(next_observations, actions)
            
            rollout_batch_size = next_observations.shape[0]

            for j in range(rollout_batch_size):

                if i == 0:      # this state-action will lead to expert-state
                    replay_buffer.add(observations[j], actions[j], next_observations[j], reward=1, done=0)
                else:
                    replay_buffer.add(observations[j], actions[j], next_observations[j], reward=0, done=0)


            next_observations = observations


    else:

        raise Exception("Not Implemented")

        observations = state.cpu().data.numpy()
        weights = weight.cpu().data.numpy()
        rollout_batch_size = rollout_batch_size
        for i in range(args.rollout_length):
            if is_uniform:
                action_size = (rollout_batch_size, rbc_policy.action_dim)
                actions = np.random.uniform(low=-1, high=1, size=action_size)
            else:
                actions = rbc_policy.select_action(observations)
            next_observations, rewards, dones, _ = fake_env.step(observations, actions)
            for j in range(rollout_batch_size):
                replay_buffer.add(observations[j], actions[j], next_observations[j], rewards[j], weights[j], dones[j])

            non_dones = ~dones.squeeze(-1)
            if non_dones.sum() == 0:
                print("Model rollout break early")
                break

            observations = next_observations[non_dones]
            weights = weights[non_dones]
            rollout_batch_size = observations.shape[0]

    return replay_buffer

def model_rollout(rbc_policy, fake_env, dataset_replay_buffer, replay_buffer, args, rollout_length=3, rollout_batch_size=1000, is_uniform=False, OODdet=None):


    state, action, next_state, reward, weight, not_done = dataset_replay_buffer.sample(rollout_batch_size)

    if not args.is_forward_rollout:     # reverse augmentation
        
        next_observations = state.cpu().data.numpy()
        weights = weight.cpu().data.numpy()
        rollout_batch_size = rollout_batch_size
        for i in range(rollout_length):

            if is_uniform:
                action_size = (rollout_batch_size, rbc_policy.action_dim)
                actions = np.random.uniform(low=-1, high=1, size=action_size)
            else:
                actions = rbc_policy.select_action(next_observations)
            observations, rewards, dones, _ = fake_env.step(next_observations, actions)


            rollout_batch_size = next_observations.shape[0]

            for j in range(rollout_batch_size):
                # replay_buffer.add(observations[j], actions[j], next_observations[j], rewards[j], weights[j], dones[j])

                if i == 0:      # this state-action will lead to expert-state
                    replay_buffer.add(observations[j], actions[j], next_observations[j], reward=1, done=0)
                else:
                    replay_buffer.add(observations[j], actions[j], next_observations[j], reward=0, done=0)

            # non_dones = ~dones.squeeze(-1)
            # if non_dones.sum() == 0:
            #     print("Model rollout break early")
            #     break

            # next_observations = observations[non_dones]
            # weights = weights[non_dones]
            # rollout_batch_size = next_observations.shape[0]

            next_observations = observations


    else:

        raise Exception("Not Implemented")

        observations = state.cpu().data.numpy()
        weights = weight.cpu().data.numpy()
        rollout_batch_size = rollout_batch_size
        for i in range(args.rollout_length):
            if is_uniform:
                action_size = (rollout_batch_size, rbc_policy.action_dim)
                actions = np.random.uniform(low=-1, high=1, size=action_size)
            else:
                actions = rbc_policy.select_action(observations)
            next_observations, rewards, dones, _ = fake_env.step(observations, actions)
            for j in range(rollout_batch_size):
                replay_buffer.add(observations[j], actions[j], next_observations[j], rewards[j], weights[j], dones[j])

            non_dones = ~dones.squeeze(-1)
            if non_dones.sum() == 0:
                print("Model rollout break early")
                break

            observations = next_observations[non_dones]
            weights = weights[non_dones]
            rollout_batch_size = observations.shape[0]

    return replay_buffer


