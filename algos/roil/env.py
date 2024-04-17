import numpy as np




class FakeEnv:
    def __init__(self, model,
                 is_use_reward=True,
                 is_use_oracle_reward=False,
                 is_fake_deterministic=False):
        self.model = model
        # self.config = config
        # self.args = args
        self.is_use_reward = is_use_reward
        self.is_use_oracle_reward = is_use_oracle_reward
        self.is_fake_deterministic = is_fake_deterministic

    '''
        x : [ batch_size, obs_dim + 1 ]
        means : [ num_models, batch_size, obs_dim + 1 ]
        vars : [ num_models, batch_size, obs_dim + 1 ]
    '''

    def get_uncertainty(self, obs, act):
        ensemble_preds = self.model.predict(obs, act)
    
    def step(self, obs, act):
        assert len(obs.shape) == len(act.shape)
        if len(obs.shape) == 1:
            obs = obs[None]
            act = act[None]
            return_single = True
        else:
            return_single = False

        ensemble_preds = self.model.predict(obs, act)

        num_models, batch_size, _ = ensemble_preds.shape

        ensemble_model_means = ensemble_preds.mean(dim=0, keepdim=True).expand(num_models, -1, -1)
        ensemble_model_stds = ensemble_preds.std(dim=0, keepdim=True).expand(num_models, -1, -1)

        ensemble_model_means = ensemble_model_means.cpu().numpy()
        ensemble_model_stds = ensemble_model_stds.cpu().numpy()

        ensemble_model_means += obs


        if self.is_fake_deterministic:
            ensemble_samples = ensemble_model_means
        else:
            ensemble_samples = ensemble_model_means + np.random.normal(size=ensemble_model_means.shape) * ensemble_model_stds

        if not self.is_fake_deterministic:
            #### choose one model from ensemble
            num_models, batch_size, _ = ensemble_model_means.shape
            # model_inds = self.model.random_inds(batch_size)
            model_inds = np.random.choice(num_models, size=batch_size)
            batch_inds = np.arange(0, batch_size)
            samples = ensemble_samples[model_inds, batch_inds]
            ####
        else:
            samples = np.mean(ensemble_samples, axis=0)

        next_obs = samples

        # rewards, next_obs = samples[:,:1], samples[:,1:]
        
        # if not self.args.is_forward_rollout:
        #     terminals = self.config.termination_fn(next_obs, act, obs)
        #     if self.is_use_oracle_reward:
        #         rewards = self.config.reward_fn(next_obs, act, obs)
        # else:
        #     terminals = self.config.termination_fn(obs, act, next_obs)
        #     if self.is_use_oracle_reward:
        #         rewards = self.config.reward_fn(obs, act, next_obs)

        # penalized_rewards = rewards

        if return_single:
            next_obs = next_obs[0]
            # penalized_rewards = penalized_rewards[0]
            terminals = terminals[0]

        return next_obs, None, None, None
    
        # return next_obs, penalized_rewards, terminals, None




class OracleEnv:
    def __init__(self, environment, state_mean = 0, state_std = 1):
        self.env = environment

        self.state_mean = state_mean
        self.state_std = state_std

        print(self.env.spec.id.split("-")[0])
        task_name = self.env.spec.id
        if task_name[:6] == 'maze2d':
            domain = 'maze2d'
            test_model_length = 2
        elif task_name[:7] == 'antmaze':
            domain = 'antmaze'
            test_model_length = 15
        elif task_name[:11] == 'halfcheetah':
            domain = 'halfcheetah'
            test_model_length = 13
        elif task_name[:6] == 'hopper':
            domain = 'hopper'
            test_model_length = 5
        elif task_name[:8] == 'walker2d':
            domain = 'walker2d'
            test_model_length = 13
        elif task_name[:4] == 'ant-':
            domain = 'ant'
            test_model_length = 13
        else:
            test_model_length = environment.observation_space.high.shape[0]
            # raise NotImplementedError

        if task_name[:6] == 'maze2d' or task_name[:7] == 'antmaze':
            test_padding = 0
        elif task_name[:6] == 'hopper' or task_name[:11] == 'halfcheetah' or task_name[:3] == 'ant' or task_name[:8] == 'walker2d':
            test_padding = 1
        else:
            test_padding = 0
            # raise NotImplementedError
    
        self.test_model_length = test_model_length
        self.test_padding = test_padding

    def step(self, obs, act, deterministic=False):
        assert len(obs.shape) == len(act.shape)
        
        self.env.reset()

        batchsize = obs.shape[0]
        next_obs_list = []
        rew_list = []
        tem_list = []
        # print('self.env.model.nq: ', self.env.model.nq)
        for i in range(batchsize):



            # qpos = obs[i][:self.test_model_length]
            # qvel = obs[i][self.test_model_length:]

            raw_obs = obs[i] * self.state_std + self.state_mean

            qpos = raw_obs[:self.test_model_length]
            qvel = raw_obs[self.test_model_length:]

            if self.test_padding > 0:
                qpos = np.concatenate([[0,], qpos], axis=0)

            
            self.env.set_state(qpos, qvel)


            next_observation, reward, terminal, _ = self.env.step(act[i])
            next_obs_list.append(next_observation)
            rew_list.append([reward])
            tem_list.append([terminal])
            ### next qpos
            next_qpos = next_observation[:2]
        next_obs = np.array(next_obs_list)
        rewards = np.array(rew_list)
        terminals = np.array(tem_list)

        return next_obs, rewards, terminals, {}
