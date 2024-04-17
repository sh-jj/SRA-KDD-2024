


from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import asdict, dataclass
import os
import sys
from pathlib import Path
import random
import uuid

from copy import deepcopy

import d4rl
import gym
import numpy as np
import pyrallis
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import time
from datetime import datetime

from torch.distributions import Normal
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import TanhTransform


from datasets import load_data
import utils

import logging
import pandas as pd

import algos.roil.ReverseBC as ReverseBC
import algos.roil as roil

from algos.oil import IQL


from utils.tools import *

TensorBatch = List[torch.Tensor]

TEST_LAST_STEPS = 10




def train_method(args):




    data_e, data_o, env = load_data.get_offline_imitation_data(args.expert_data, args.offline_data, 
                                                          expert_num=args.expert_num, offline_exp=args.offline_exp)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    

    replay_buffer_e = utils.ReplayBuffer(state_dim, action_dim, args.buffer_size, args.device)
    replay_buffer_o = utils.ReplayBuffer(state_dim, action_dim, args.buffer_size, args.device)


    replay_buffer_e.convert_D3RL(data_e)
    replay_buffer_o.convert_D3RL(data_o)



    if args.normalize:

        observations_all = np.concatenate([replay_buffer_e.state, replay_buffer_o.state]).astype(np.float32)

        # print(observations_all.shape)

        state_mean = np.mean(observations_all, 0)
        state_std = np.std(observations_all, 0) + 1e-3
    else:
        state_mean, state_std = 0, 1

    replay_buffer_e.normalize_states(mean=state_mean, std=state_std)
    replay_buffer_o.normalize_states(mean=state_mean, std=state_std)
    

        
    env = utils.wrap_env(env, state_mean=state_mean, state_std=state_std)

    max_action = float(env.action_space.high[0])


    if args.log_path is not None:

        config_path = os.path.join(args.log_path, args.name)

        print(f"Config path: {config_path}")

        os.makedirs(config_path, exist_ok=True)

        with open(os.path.join(config_path, "args.yaml"), "w") as f:
            pyrallis.dump(args, f)


        print(f"Checkpoint path: {args.checkpoints_path}")
        os.makedirs(args.checkpoints_path, exist_ok=True)


    # from visual_maze2d import draw_data
    # draw_data(replay_buffer_e, state_mean, state_std, os.path.join(args.save_file_path, f"{args.name}-exp.png"), show_fig=args.visual_data)
    # draw_data(replay_buffer_o, state_mean, state_std, os.path.join(args.save_file_path, f"{args.name}-off.png"), show_fig=args.visual_data)
    # print("Draw graph >>> ",args.save_file_path)
    
    print("State_dim : ",state_dim)
    print("Action_dim : ",action_dim)
    # print("expert : reward : ")
    # exit(0)
    # Set seeds
    seed = args.seed
    utils.set_seed(seed, env)



    # ===== Create Env Model Ensemble =====

    replay_buffer_all = utils.ReplayBuffer(state_dim, action_dim, args.buffer_size, args.device)
    replay_buffer_all = replay_buffer_all.extend(replay_buffer_e)
    replay_buffer_all = replay_buffer_all.extend(replay_buffer_o)

    print("Load Dynamics Models")

    model_ensemble = roil.DynamicsEnsemble(env, args, replay_buffer=replay_buffer_all, 
                                           reverse=True, 
                                           hidden_sizes=[1024, 1024], base_seed=args.seed, device=args.device)
    

    dynamics_dir = os.path.join(args.log_path, f"checkpoints-seed[{args.seed}]")
    dynamics_path = os.path.join(dynamics_dir, "dynamics.pt")


    if os.path.exists(dynamics_path):
        print(f">>>>> Loading Dynamics ensemble with : {dynamics_path}")

        # dynamics_path = os.path.join(args.checkpoints_path, args.ensemble_checkpoint)
        model_ensemble.load(dynamics_path)
    else:
        print(f">>>>> Training Dynamics ensemble ")
        model_ensemble.train(n_epochs=args.env_n_epochs, log_epoch=True, grad_clip=args.grad_clip, reverse=True)

        # dynamics_path = os.path.join(args.checkpoints_path, "dynamics.pt")
        print(f">>>>> Saving ensemble weights in {dynamics_path}")
        model_ensemble.save(dynamics_path)

    # print("Evaluate Dynamics on whole data")
    # disc, err_all = model_ensemble.compute_threshold(replay_buffer_all)
    # print(f">>>>> Computed Maximum Discrepancy for Ensemble: {model_ensemble.threshold}")
    # print(f">>>>> Error for Ensemble: {err_all.mean()}")

    # err_all_list = model_ensemble.eval(replay_buffer_all)
    # print(f">>>>> Error for Base-Model : {err_all_list}")
    

    # print("Evaluate Dynamics on expert data")
    # disc, err_e = model_ensemble.compute_threshold(replay_buffer_e)
    # print(f">>>>> Computed Maximum Discrepancy for Ensemble: {model_ensemble.threshold}")
    # print(f">>>>> Error for Ensemble: {err_e.mean()}")

    # err_e_list = model_ensemble.eval(replay_buffer_e)
    # print(f">>>>> Error for Base-Model : {err_e_list}")



    print("---------------------------------------")
    print("Build Reverse Policy")

    true_env = roil.OracleEnv(env, state_mean=state_mean, state_std=state_std)
    fake_env = roil.FakeEnv(model=model_ensemble, is_fake_deterministic=True)

 
    if args.model_rollout:
        rbc_policy = ReverseBC.ReverseBC(state_dim, action_dim, max_action, args.device, args.entropy_weight)
        
        train_buffer = replay_buffer_all
        test_buffer = replay_buffer_e


        if args.if_raw_action:
            print("rollout with raw policy")
            roil.test_dynamics(rbc_policy, fake_env, true_env, test_buffer, rollout_batch_size=test_buffer.size,
                                        rollout_length=1, if_raw_action=args.if_raw_action)

        if not args.is_uniform_rollout:
            ckp_file = os.path.join(args.checkpoints_path, "reverse_policy.pt")

            if not os.path.exists(ckp_file):
                # train reverse policy with VAE

        
                print(f"train reverse policy with {args.reverse_policy_max_timesteps:.0f} steps")

                training_iters = 0
                while training_iters < args.reverse_policy_max_timesteps:
                    vae_loss = rbc_policy.train(train_buffer, iterations=int(args.rb_eval_freq), batch_size=args.batch_size)

                    training_iters += args.rb_eval_freq

                    print(f'Train step: {int(training_iters)}    |   VAE training Loss:  {vae_loss}')

                    roil.test_dynamics(rbc_policy, fake_env, true_env, test_buffer, rollout_batch_size=1000,
                                        rollout_length=5, is_uniform=args.is_uniform_rollout)


                    print("---------------------------------------")

                # ckp_file = os.path.join(args.checkpoints_path, "reverse_policy.pt")
                rbc_policy.save(ckp_file)
                # print(f"Save reverse_bc: {training_iters}")
                print("Save reverse_bc >>> ", ckp_file)
            else:

                rbc_policy.load(ckp_file)
                print("Load reverse_bc <<< ", ckp_file)
            

            print("Test dynamics on the expert state: ")
            roil.test_dynamics(rbc_policy, fake_env, true_env, test_buffer, rollout_batch_size=test_buffer.size,
                                        rollout_length=5, is_uniform=args.is_uniform_rollout)
        else:
            print("rollout with uniform policy")
            roil.test_dynamics(rbc_policy, fake_env, true_env, test_buffer, rollout_batch_size=test_buffer.size,
                                        rollout_length=args.rollout_length, is_uniform=args.is_uniform_rollout)
        


    print("---------------------------------------")
    
    print("Reverse Augmentation")
    
    # Set seeds
    seed = args.seed
    utils.set_seed(seed, env)


    aug_replay_buffer = utils.ReplayBuffer(state_dim, action_dim, max_size=args.aug_size, device=args.device)

    if not args.model_rollout_with_policy_filter:# if "raw" in args.reverse_mode:
        model_replay_buffer = utils.ReplayBuffer(state_dim, action_dim, max_size=args.aug_size, device=args.device)

        # if args.reverse_mode == "raw-e" :
        base_buffer = replay_buffer_e
        
        if args.reverse_mode == "raw-all":
            base_buffer = replay_buffer_all

        rollout_batch_size = 2000

        aug_count, filtered_count = 0, 0

        while model_replay_buffer.size < model_replay_buffer.max_size:


            new_buffer = utils.ReplayBuffer(state_dim, action_dim, max_size=int(1e6), device=args.device)

            roil.model_rollout(rbc_policy, fake_env, base_buffer, new_buffer, args, rollout_batch_size=rollout_batch_size,
                                    is_uniform=args.is_uniform_rollout, rollout_length=args.rollout_length)

 

            model_replay_buffer = model_replay_buffer.extend(new_buffer)
                
        
            prints_num = 5
            if  model_replay_buffer.size // (model_replay_buffer.max_size // prints_num) > (model_replay_buffer.size - rollout_batch_size) // (model_replay_buffer.max_size // prints_num):
                
                
                if args.filter_ind:
                    print(f"[Model Rollout] generate {model_replay_buffer.size} / {model_replay_buffer.max_size} transitions, filtered {int(100*filtered_count/aug_count):.2f}%")
                else:
                    print(f"[Model Rollout] generate {model_replay_buffer.size} / {model_replay_buffer.max_size} transitions")


                
                aug_count, filtered_count = 0, 0

        aug_replay_buffer = model_replay_buffer




    print("Finished...")

    

    print("---------------------------------------")


    # actor = Actor(state_dim, action_dim, max_action).to(args.device)
    # actor_optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)


    # Set seeds
    seed = args.seed
    utils.set_seed(seed, env)


    max_action = float(env.action_space.high[0])
    
    kwargs = {
        "state_dim": state_dim, 
        "action_dim": action_dim, 
        "max_action": max_action,
        "lr": args.lr, 
        "wd": args.wd, 
        "discount": args.discount,
        "tau": args.tau,
        "device": args.device,
        # IQL
        "beta": args.beta,
        "iql_tau": args.iql_tau,
        "policy_freq": args.policy_freq,
        "bc_freq": args.bc_freq, 
        # IQL + BC
        "alpha": args.alpha,
    }


    print("---------------------------------------")
    print(f"Training, Expert: {args.expert_data} [{args.expert_num}], Seed: {seed}")



    print("Expert:  {} transitation".format(replay_buffer_e.size))


    print("Reverse-Augmented:  {} transitation".format(aug_replay_buffer.size))

    print(f"Sample ratio of augmented tranistions: {args.model_ratio}")

    print("")

    print("Offline:  {} transitation".format(replay_buffer_o.size))
    print("---------------------------------------")


    # Initialize policy
    policy = IQL(**kwargs)
    
    if(args.policy_load_model != ""):
        # 200,000 step already trained, so need to rollout new transitions.
        
        print(f"Load Policy Model from {args.policy_load_model}")
        policy.load(filename=args.policy_load_model)

    
    # ====================================================
    else:
        print("learn IQL+BC on expert + offline buffer + augmented buffer")

        # Set Reward
        replay_buffer_e.reward = np.ones_like(replay_buffer_e.reward)
        replay_buffer_o.reward = np.zeros_like(replay_buffer_o.reward)
        aug_replay_buffer.reward = np.zeros_like(aug_replay_buffer.reward)

        # wandb_init(asdict(args))


        evaluations = []
        best_d4rl_score = 0


        training_iters = 0
        
        
        offline_buffers = None
        expert_buffers = replay_buffer_e
        if not args.model_rollout_with_policy_filter:
            offline_buffers = utils.BufferEnsemble([(replay_buffer_o, 1. - args.model_ratio), 
                                                (aug_replay_buffer, args.model_ratio), 
                                                ])
        

        current_ratio = args.model_ratio
        offline_start_buffer = utils.ReplayBuffer(state_dim, action_dim,max_size = args.aug_size, device=args.device)
        single_time_aug_size = args.aug_size//((args.max_timesteps - args.warm_up) // args.rollout_freq)
        
        base_buffer = replay_buffer_e
        if args.reverse_mode == "raw-all":
            base_buffer = replay_buffer_all

        rollout_batch_size = 2000
        
        print("Total Augment Data Size: ",args.aug_size)
        print("Single Time Augment Data Size: ",single_time_aug_size)
        
        warmup_dir = os.path.join(args.log_path, f"checkpoints-seed[{args.seed}]",f"warmup_{args.warm_up}")
        print("Warmup_Dir : ",warmup_dir)
        # if(os.path.exists(warmup_dir)):
        try:
            policy.load(warmup_dir)
            training_iters = args.warm_up
            print("---------------------------------------")
            
            print("Escape warm up training, now training_iters = ",args.warm_up)
            # print("---------------------------------------")
        except:
            pass

        
        
        while training_iters < args.max_timesteps:

            if(training_iters == args.warm_up):
                print("---------------------------------------")
                
                print("Save warm up model >>> ",warmup_dir)
                # print("---------------------------------------")
                
                policy.save(warmup_dir)
                seed = args.seed
                utils.set_seed(seed, env)
                
            # print('Train step:', training_iters)
            if args.model_rollout_with_policy_filter and training_iters >= args.warm_up:
                if(offline_buffers == None or training_iters % args.rollout_freq == 0):
                    
                    print("-------------------Model Rollout With Policy-------------------")
                    
                    save_path = os.path.join(args.save_file_path,str(training_iters))
                    os.makedirs(save_path, exist_ok=True)

                    if(args.use_offline):
                        
                        def calc_weight_for_databuffer_with_policy(policy, buffer,device):
        
                            l,buffer_size = 0,buffer.size
                            weight = torch.Tensor([])
                            while(l < buffer_size):
                                r = min(l+1000,buffer_size)
                                state_i = torch.Tensor(buffer.state[l:r]).to(device)
                                _,_,action_i = policy(state_i)
                                action_i = action_i.to(device)


                                weight_i = torch.sum(policy.get_log_density(state_i,action_i),dim=-1).detach().cpu()
                                weight = torch.cat([weight,weight_i.reshape(-1)],dim=0)
                                
                                l += 1000
                            return weight
                        
                        def select_offline_start_point(offline_buffer, expert_buffer, policy, threshold):

                            
                            weight_offline = calc_weight_for_databuffer_with_policy(policy.actor, offline_buffer, args.device)
                            weight_expert  = calc_weight_for_databuffer_with_policy(policy.actor, expert_buffer,  args.device)
                            
                            # sorted_weight_expert, _ = weight_expert.sort()
                            
                            thre = torch.quantile(weight_expert, threshold)
                            
                            index = torch.nonzero(weight_offline > thre).view(-1)
                            
                            if(len(index) > expert_buffer.size * 10):
                                _, index = torch.topk(weight_offline, expert_buffer.size * 10)
                            
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
                            
                            buffer_o = set_buffer_with_index(offline_buffer, index)
                            return buffer_o
                        
                        
                        buffer_o = deepcopy(replay_buffer_o)
                        buffer_o = select_offline_start_point(buffer_o, replay_buffer_e, policy, args.start_point_threshold)
                        

                        if(args.reverse_mode == "raw-all"):
                            real_base_buffer = replay_buffer_all
                        else:
                            real_base_buffer = replay_buffer_e
                        
                        if(offline_start_buffer.size == 0):
                            offline_start_buffer = offline_start_buffer.extend(buffer_o)
                            base_buffer = utils.BufferEnsemble([(real_base_buffer, args.expert_start_ratio,
                                                                 offline_start_buffer, 1 - args.expert_start_ratio)])
                        else:
                            base_buffer = utils.BufferEnsemble([(real_base_buffer, args.expert_start_ratio),
                                                                         (offline_start_buffer, (1 - args.expert_start_ratio)*(1 - args.new_buffer_ratio)),
                                                                         (buffer_o, (1 - args.expert_start_ratio)*args.new_buffer_ratio)])
                            offline_start_buffer = offline_start_buffer.extend(buffer_o)
                        
                        print("[Offline Start] : Real start : ",real_base_buffer.size,"    total_offline_start : ",offline_start_buffer.size,"    New select offline start : ",buffer_o.size)


                    temp_replay_buffer = utils.ReplayBuffer(state_dim, action_dim,max_size = args.aug_size, device=args.device)
                    temp_replay_buffer = roil.model_rollout_with_policy_filter(rbc_policy, fake_env,
                                                                base_buffer, temp_replay_buffer,
                                                                state_mean=state_mean,state_std=state_std,
                                                                rollout_length=args.rollout_length,
                                                                rollout_batch_size=rollout_batch_size,
                                                                max_rollout_size=single_time_aug_size,
                                                                #  ratio=args.rollout_filter_ratio,
                                                                is_uniform=args.is_uniform_rollout,device=args.device,log_path=save_path,
                                                                weight_sample=args.rollout_weighted_sample,policy=policy,
                                                                detail_draw=False)
                    # You cannot directly discard the previous data
                    # This may cause problems with forgetting
                    # One possiable solution is to add a proportional coefficient (default: 0.7)

                    temp_replay_buffer.reward = np.zeros_like(temp_replay_buffer.reward)
                    if(aug_replay_buffer.size == 0):
                        aug_replay_buffer = aug_replay_buffer.extend(temp_replay_buffer)
                        
                        offline_buffers = utils.BufferEnsemble([(replay_buffer_o, 1. - current_ratio), 
                                                            (aug_replay_buffer, current_ratio), 
                                                            ])
                    else:
                        

                        offline_buffers = utils.BufferEnsemble([(replay_buffer_o, 1. - current_ratio), 
                                                            (aug_replay_buffer, current_ratio * (1 - args.new_buffer_ratio)), 
                                                            (temp_replay_buffer, current_ratio * args.new_buffer_ratio)
                                                            ])
                        
                        aug_replay_buffer = aug_replay_buffer.extend(temp_replay_buffer)
                        
                    print("[Policy Rollout] : new data size: ",temp_replay_buffer.size,"    total aug data size: ",aug_replay_buffer.size,"    total data size: ",offline_buffers.size)
                    
                    
                    print("-------------------Model Rollout Finished-------------------")

                
            current_learning_rate = policy.actor_optimizer.state_dict()['param_groups'][0]['lr']

            if args.lr_decay and training_iters == args.max_timesteps / 2:
                for param_group in policy.actor_optimizer.param_groups:
                    param_group['lr'] = current_learning_rate / 10

                for param_group in policy.v_optimizer.param_groups:
                    param_group['lr'] = current_learning_rate / 10

                for param_group in policy.q_optimizer.param_groups:
                    param_group['lr'] = current_learning_rate / 10
                
                print('Decreasing learning rate:', current_learning_rate, " ---> "  , current_learning_rate / 10)

            

            if(training_iters < args.warm_up):
                log_dict = policy.train(expert_buffers, replay_buffer_o, iterations=int(args.eval_freq), batch_size=args.batch_size)
            else:
                log_dict = policy.train(expert_buffers, offline_buffers, iterations=int(args.eval_freq), batch_size=args.batch_size)

            training_iters += args.eval_freq
            print(f"Training iterations: {training_iters}/{args.max_timesteps}:")
            
            if args.reverse_mode == "reduce-ratio" and training_iters % 50000 == 0:

                current_ratio -= 0.2
                print(f"reduce model ratio: {current_ratio + 0.2:.2f}   >>>     {current_ratio:.2f}")
                offline_buffers = utils.BufferEnsemble([(replay_buffer_o, 1. - current_ratio), 
                                            (aug_replay_buffer, current_ratio), 
                                            ])
                
            
            actor_loss = np.mean(log_dict["actor_loss"])
            value_loss = np.mean(log_dict["value_loss"])
            q_loss = np.mean(log_dict["q_loss"])
            actor_loss_iql = np.mean(log_dict["actor_loss_iql"])
            actor_loss_bc = np.mean(log_dict["actor_loss_bc"])
            
            print("actor_loss: ", actor_loss, "    value_loss: ", value_loss)
            print("actor_loss_iql: ", actor_loss_iql, "    actor_loss_bc: ", actor_loss_bc)
            print("q_loss: ",q_loss)


            eval_scores = utils.eval_actor(env, policy.actor, device=args.device, eval_episodes=args.eval_episodes, seed=args.seed)

            normalized_eval_score = env.get_normalized_score(eval_scores) * 100.0
            
            evaluations.append(normalized_eval_score)
            
            print("Reverse QL agent: ")
            print(f"Evaluation over {args.eval_episodes} episodes: "
            f"{eval_scores.mean():.3f} , D4RL score: {normalized_eval_score.mean():.3f}")

            
            print("---------------------------------------------------------")

            if args.mode != "debug":
                wandb.log(
                        {
                        "actor_loss_train": actor_loss, 
                        "critic_loss_train": q_loss, 
                        # "actor_loss_train": train_loss, 
                        # "actor_loss_val": val_loss, 
                        "d4rl_normalized_score": normalized_eval_score.mean(), 
                        # "recorded_d4rl_score": best_d4rl_score
                        },
                        step=training_iters,
                    )

        
        print("finish training")



        # file_suffix = f"_preaug{args.rollout_length}"
        file_suffix=""
        ckp_path = os.path.join(args.checkpoints_path, METHOD+file_suffix)
        policy.save(ckp_path)
        print("save model >>> ", ckp_path)
        

        print("--------------       results     -----------")
        
        results_table = utils.summary_table_with_multi_evaluations(evaluations)

        print("last results")
        print(results_table.tail())


        results_table.to_csv(os.path.join(args.log_path, "all_test_seed{}{}.csv".format(args.seed,file_suffix)))
        
        print("save results >>>", os.path.join(args.log_path, "all_test_seed{}{}.csv".format(args.seed,file_suffix)))

        print("least result mean +/- std: {} +/- {}".format(results_table.iloc[-1]["mean"], results_table.iloc[-1]["std"]))
        

        print("last results")
        print(results_table.tail())

        if args.mode != "debug":

            wandb.log(
                        {"all_evaluations": wandb.Table(dataframe=results_table)},
                    )
            
            wandb.run.summary["least_evaluations_average"] = results_table.iloc[-1]["mean"]
            
    print(f'Test model ... ... ')

    eval_policy(env, policy.actor, device=args.device, eval_episodes=args.eval_episodes, seed=args.seed,
                data=replay_buffer_e, state_mean=state_mean, state_std=state_std, 
                imgname=os.path.join(args.checkpoints_path, f"{args.name}-premodel"), show_fig=args.visual_data)            

    print("---------------------------------------")
 
    if args.mode != "debug":
        wandb.finish()

    return kwargs



import argparse




if __name__ == "__main__":

    METHOD = "SRA-IQL"
    CURRENT_TIME = str(datetime.now()).replace(" ", "-")

    parser = argparse.ArgumentParser()


    # Wandb logging
    parser.add_argument('--project', type=str, default='Model-Based Offline-Imitation-Learning')  
    parser.add_argument('--group', type=str, default='MuJoCo')



    # exp - data

    parser.add_argument('--expert-data', type=str, default='hopper-expert-v2')  
    parser.add_argument('--offline-data', type=str, default='hopper-random-v2')
    parser.add_argument('--expert-num', type=int, default=10, help='number of expert episodes')
    parser.add_argument('--offline-exp', type=int, default=0, help='number of expert episodes in offline data')



    # exp - BCDP

    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--wd', type=float, default=0.005, help='weight decay')
    parser.add_argument('--lr-decay', action='store_true', help='if decay the lr (* 0.1) at the half of training')

    parser.add_argument('--max_timesteps', type=int, default=500000, help='')
    parser.add_argument("--discount", type=float, default=0.99)  # Discount factor

    
    parser.add_argument("--expl_noise", type=float, default=0.1)  
    parser.add_argument("--tau", type=float, default=0.005)  # Target network update rate
    
    parser.add_argument("--beta", default=3.0, type=float)
    parser.add_argument("--iql_tau", default=0.7, type=float)
    
    parser.add_argument("--policy_freq", type=int, default=1)  
    parser.add_argument("--bc_freq", type=int, default=1)
    
    # parser.add_argument("--alpha", type=float, default=2.5)  
    parser.add_argument("--alpha", type=float, default=1.0)  
    parser.add_argument('--normalize', type=bool, default=True, help='Normalize states')

    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for all networks')
    # parser.add_argument('--discount', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--buffer-size', type=int, default=10_000_000, help='Replay buffer size')

    parser.add_argument('--seed', type=int, default=0)


    # Dynamics Model Ensemble Arguments
    parser.add_argument('--env_n_models', type=int, help='Number of dynamics models in ensemble', default=7)
    parser.add_argument('--env_n_epochs', type=int, help='Number of epochs to train models', default=20)
    parser.add_argument('--grad_clip', type=float, help='Max Gradient Norm', default=1.0)
    parser.add_argument('--dynamics_optim', type=str, help='Optimizer to use [sgd, adam]', default='sgd')
    parser.add_argument('--ensemble_checkpoint', type=str, default=None)


    # Reverse Policy Arguments

    parser.add_argument('--model_rollout', type=bool, default=True, help='generate reverse policy by [model: True | random: False]')
    parser.add_argument('--is_uniform_rollout', type=bool, default=False, help='generate reverse policy by [model: False | random: True]')
    parser.add_argument('--if_raw_action', type=bool, default=False, help='use raw policy (only for evaluation)')
    parser.add_argument('--entropy_weight', type=float, default=0.5, help='')
    parser.add_argument('--reverse_policy_max_timesteps', type=int, default=2e5, help=' Max time steps to train reverse policy')
    parser.add_argument('--rb_eval_freq', type=int, default=5e4, help='How often (time steps) we evaluate')

    parser.add_argument('--reverse_policy_checkpoint', type=str, default="reverse_policy.pt")

    # reverse augmentation Arguments
    parser.add_argument('--is_forward_rollout', type=bool, default=False, help='forward / reverse')

    parser.add_argument('--rollout_length', type=int, default=3, help='Length of Model Rollout')

    # parser.add_argument('--aug_size', type=int, default=100000, help='')
    parser.add_argument('--aug_size', type=int, default=int(4e6), help='')

    parser.add_argument('--model_ratio', type=float, default=0.1, help='')

    parser.add_argument('--reverse_mode', type=str, default="raw-e", choices=["raw", "raw-e", "raw-all", "filter-offline"])

    parser.add_argument('--visual-data', action='store_true')

    # exp - Reverse - BCQ

    parser.add_argument('--reverse_QL', type=bool, default=False, help='optimize the reverse action via Q-Learning')
    parser.add_argument("--discount-bcq", default=0.99)  # Discount factor
    parser.add_argument("--tau-bcq", default=0.005)  # Target network update rate
    parser.add_argument("--lmbda-bcq", default=0.75)  # Weighting for clipped double Q-learning in BCQ
    parser.add_argument("--phi-bcq", default=0.05)  # Max perturbation hyper-parameter for BCQ
    

    # exp -testing
    parser.add_argument('--eval-episodes', type=int, default=30, help='How many episodes run during evaluation')
    parser.add_argument('--eval_freq', type=int, default=5000, help='How often (time steps) we evaluate')


    # logging
    parser.add_argument('--log-path', type=str, default='logs')  
    parser.add_argument('--load_model', type=str, default="")

    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--mode', type=str, default='train')
    
    parser.add_argument('--policy_load_model',default="",type=str)
    parser.add_argument("--model_rollout_with_policy_filter", default=True, action='store_true',help='if True, model rollout will selected by pretrained filter')
    parser.add_argument("--rollout_freq",default=10000,type=int,help="Trajectory generation is performed every N(rollout_freq) rounds of policy iterations")
    parser.add_argument("--new_buffer_ratio",default=0.3,type=float,help="The proportion of trajectories captured from newly generated data")
    parser.add_argument("--rollout_weighted_sample",action='store_true', default=True, help='if True, model rollout will selected by filter weight')
    parser.add_argument("--use_offline", default=True, action="store_true",help="if True, will use offline data as start point in training")
    parser.add_argument("--expert_start_ratio",type=float,default=0.9,help="ratio of expert start as start point")
    parser.add_argument("--start_point_threshold",type=float,default=0.8)
    parser.add_argument("--warm_up",type=int,default=200000,help="do offline RL + UDS method when training_iter < warm_up")
    
    args = parser.parse_args()


    exp_name = f"{args.expert_data}-[{args.expert_num}]-{args.offline_data}"
    if('antmaze' in args.expert_data):
        exp_name = f"{args.expert_data}-[{args.expert_num}]-{args.expert_data}"
    if args.offline_exp > 0:
        exp_name = exp_name + f"-{args.offline_exp}-exp-in-offline"

    args.group = exp_name

    args.name = f"{METHOD}-seed[{args.seed}]-time-{CURRENT_TIME}"  
    


    
    if args.is_uniform_rollout:
        reverse_policy_mode = "Uniform"
    else:
        reverse_policy_mode = "ROMI" 


    if args.reverse_mode == "raw-e":
        reverse_state_mode = "E"
        if(args.use_offline):
            reverse_state_mode = f"E[{args.expert_start_ratio}]O[{args.start_point_threshold}]"
    elif args.reverse_mode == "raw-all":
        reverse_state_mode = "ALL"
        
    if(args.model_rollout_with_policy_filter):
        reverse_state_mode=reverse_state_mode + f"-PolicyFilter[{args.new_buffer_ratio},{args.rollout_freq}]"

    args.name = f"{reverse_policy_mode}-{reverse_state_mode}-{args.rollout_length}-{args.model_ratio}-BCDP-seed[{args.seed}]-time-{CURRENT_TIME}"  


    args.log_path = os.path.join("logs", args.group)
    os.makedirs(args.log_path, exist_ok=True)

    args.log_path = os.path.join("logs", args.group, METHOD)
    os.makedirs(args.log_path, exist_ok=True)
    
    args.checkpoints_path = os.path.join(args.log_path, f"checkpoints-seed[{args.seed}]")

    args.save_file_path = os.path.join(args.log_path,args.name)
    os.makedirs(args.save_file_path, exist_ok=True)
    
    log_file = os.path.join(args.log_path, args.mode + "-" + args.name + ".txt")
    
    sys.stdout = utils.TextLogger(filename=log_file)

    if args.mode != "debug":
        utils.wandb_init(vars(args))
        
    if args.mode in ["train", "debug"]:
        train_method(args)
    
    if args.mode != "debug":
        wandb.finish()
    # test(args)

    # sys.stdout.close()