import argparse
from shihs.tools import *
import utils
import os

parser = argparse.ArgumentParser()

parser.add_argument('--eval_episodes',default=10,type=int)
parser.add_argument('--load_algos', default='ROMI-BCDP',type=str)
parser.add_argument('--load_seed', default=0,type=int)
parser.add_argument('--suffix',default="",type=str)
parser.add_argument('--log_path',default="",type=str)
parser.add_argument('--device',default="cuda",type=str)
parser.add_argument('--buffer-size', type=int, default=10_000_000, help='Replay buffer size')

# parser.add_argument('--seed',default=0,type=int)



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

replay_buffer_e = utils.ReplayBuffer(state_dim, action_dim, args.buffer_size, args.device)
replay_buffer_o = utils.ReplayBuffer(state_dim, action_dim, args.buffer_size, args.device)


replay_buffer_e.convert_D3RL(data_e)
replay_buffer_o.convert_D3RL(data_o)

observations_all = np.concatenate([replay_buffer_e.state, replay_buffer_o.state]).astype(np.float32)

# print(observations_all.shape)

state_mean = np.mean(observations_all, 0)
state_std = np.std(observations_all, 0) + 1e-3
    
replay_buffer_e.normalize_states(mean=state_mean, std=state_std)
replay_buffer_o.normalize_states(mean=state_mean, std=state_std)




# env = gym.make(env_name)
env = utils.wrap_env(env,state_mean=state_mean, state_std=state_std)

max_action = float(env.action_space.high[0])

seed = args.load_seed
utils.set_seed(seed, env)


max_action = float(env.action_space.high[0])

kwargs = {
    "state_dim": state_dim, 
    "action_dim": action_dim, 
    "max_action": max_action,
    "device": args.device
}


# Initialize policy
# load_model = "shihs/model/bcdp-rollout[10]-PG[20w]/ROMI-BCDP"
eval_episodes = args.eval_episodes


policy = BCDP(**kwargs)
if(args.load_algos == "dwbc"):
    
    from algos.dwbc import DWBC
    kwargs = {
        "state_dim": state_dim, 
        "action_dim": action_dim, 
        # "max_action": max_action,
        "device": args.device
    }
    policy = DWBC(**kwargs)
    
print(f"Load Policy Model from {model_path}")
policy.load(filename=model_path)


if(args.load_algos == "dwbc"):
    draw_evalmap(env, policy.policy, device=args.device, eval_episodes=eval_episodes, seed=seed,
        state_mean=state_mean, state_std=state_std, 
        algos_name = args.load_algos,
        save_name = figure_name,
        expert_buffer=replay_buffer_e) 

else:
    draw_evalmap(env, policy.actor, device=args.device, eval_episodes=eval_episodes, seed=seed,
        state_mean=state_mean, state_std=state_std, 
        algos_name = args.load_algos,
        save_name = figure_name,
        expert_buffer=replay_buffer_e)   