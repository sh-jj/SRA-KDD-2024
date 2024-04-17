import numpy as np
from copy import deepcopy


def test_dynamics(policy, fake_env, true_env, dataset_replay_buffer, rollout_batch_size, rollout_length, is_uniform=False, if_raw_action=False):
    state, action, next_state, reward, weight, not_done = dataset_replay_buffer.sample(rollout_batch_size)
    
    
    obs_err_list = []
    for test_length in range(1, rollout_length + 1):
        
        
        next_obs = deepcopy(state.cpu().data.numpy())
        
        if if_raw_action:
            next_obs = deepcopy(next_state.cpu().data.numpy())

        act_list = []
        next_obs_list = [next_obs]
        # rew_list = []
        rollout_batch_size = next_obs.shape[0]
        try:
            for i in range(test_length):
                if is_uniform:
                    action_size = (rollout_batch_size, policy.action_dim)
                    act = np.random.uniform(low=-1, high=1, size=action_size)
                else:
                    act = policy.select_action(next_obs)
    
    
                if if_raw_action:
                    act = action.cpu().data.numpy()

                # next_obs, rew, dones, _ = fake_env.step(next_obs, act)
                # non_dones = ~dones.squeeze(-1)
                # for j in range(len(next_obs_list)):
                #     next_obs_list[j] = next_obs_list[j][non_dones]
                # for j in range(len(rew_list)):
                #     rew_list[j] = rew_list[j][non_dones]
                # for j in range(len(act_list)):
                #     act_list[j] = act_list[j][non_dones]

                # next_obs = next_obs[non_dones]
                # rew = rew[non_dones]
                # act = act[non_dones]

                next_obs, _, _, _ = fake_env.step(next_obs, act)

                true_obs = state.cpu().data.numpy()
                # print(true_obs)
                # print(next_obs)

                # print("-----------------------------")
                # obs_diff = np.mean(np.mean(np.square(true_obs - next_obs), axis=-1), axis=-1)
                # print("obs_diff", obs_diff)

                # print("(true) state -> next-state:     ", state[:3],   '   ->  ', next_state[:3])

                # print("(fake) state -> next-state:     ", next_obs[:3],   '   ->  ', next_state[:3])

                # print("-----------------------------")


                # xit(0)

                next_obs_list.append(next_obs)
                # rew_list.append(rew)
                act_list.append(act)
                
                rollout_batch_size = next_obs.shape[0]

            obs = deepcopy(next_obs)
            for i in range(test_length - 1, -1, -1):
                
                act = act_list[i]
                obs_fake = next_obs_list[i]
                # rew_fake = rew_list[i]
                obs, rew, dones, _ = true_env.step(obs, act)

            obs_diff = np.mean(np.mean(np.square(obs - obs_fake), axis=-1), axis=-1)

            # print("-----------------------------")
            # print("raw_obs", obs)
            
            # print("mean, std: ", obs.mean(axis=0), obs.std(axis=0))

            # print("fake_obs", obs_fake)

            # print("mean, std: ", obs_fake.mean(axis=0), obs_fake.std(axis=0))

            # print("-----------------------------")

            # rew_diff = np.mean(np.mean(np.square(rew - rew_fake), axis=-1), axis=-1)
            # print('[ Test Model ] Rollout {} | Obs: {} | Rew: {}'.format(test_length, obs_diff, rew_diff))
            print('[ Test Model ] Rollout {} | Obs: {} '.format(test_length, obs_diff))


            obs_err_list.append(obs_diff)

        except:
            break
        
    return obs_err_list
