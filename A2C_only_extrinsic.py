import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from Environment_wrappers.wrappers import ResizeAndGrayscale

from Agents.neural_network import ActorCritic
from Logger.my_logger import A2CLogger

#from stable_baselines3.a2c.a2c import A2C
#from stable_baselines3.ppo.ppo import PPO
from stable_baselines3.dqn.dqn import DQN
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv

from Config import ENV_CFG, A2C_CFG
from Config.all_hyperparams_dict import HYPERPARAMS

import wandb

import torch
import random
import numpy as np

if ENV_CFG.SEED != -1:
    torch.manual_seed(ENV_CFG.SEED)
    random.seed(ENV_CFG.SEED)
    np.random.seed(ENV_CFG.SEED)
    torch.backends.cudnn.benchmark = False
    #torch.use_deterministic_algorithms(mode = True)

run = wandb.init(project = "MarioExtrinsic", config = HYPERPARAMS)

env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, ENV_CFG.ALL_ACTION_SPACE)
action_dim = env.action_space.n

test_env = MaxAndSkipEnv(env, skip=1)
test_env = ResizeAndGrayscale(test_env, ENV_CFG.RESIZED_SIZE, ENV_CFG.TEMPORAL_CHANNELS)
test_env = ExtrinsicWrapper(test_env)

env = MaxAndSkipEnv(env, skip=ENV_CFG.ACTION_SKIP)
env = ResizeAndGrayscale(env, ENV_CFG.RESIZED_SIZE, ENV_CFG.TEMPORAL_CHANNELS)
env = ExtrinsicWrapper(env)

policy_kwargs = dict(
    features_extractor_class=ActorCritic,
    net_arch=[dict(pi=[A2C_CFG.POLICY_NEURONS], vf=[A2C_CFG.VALUE_NEURONS])]
)

policy_kwargs_DQN = dict(
    features_extractor_class=ActorCritic,
    net_arch=[A2C_CFG.POLICY_NEURONS]
)
#cnn_policy = CnnPolicy()
log_path = 'sb3_logs'
format = 'tensorboard'
#model = A2C("CnnPolicy", env, verbose=1, learning_rate=A2C_CFG.LR, use_rms_prop=A2C_CFG.RMS_PROP, 
#            policy_kwargs=policy_kwargs, n_steps=A2C_CFG.NUM_STEPS, seed=ENV_CFG.SEED, 
#            max_grad_norm=A2C_CFG.MAX_GRAD_NORM, gamma=A2C_CFG.GAMMA, vf_coef=A2C_CFG.VALUE_LOSS_COEF,
#            ent_coef=A2C_CFG.ENTROPY_COEF, gae_lambda=A2C_CFG.GAE_LAMBDA)

model = DQN("CnnPolicy", env, verbose=1, policy_kwargs=policy_kwargs_DQN, seed=ENV_CFG.SEED,
            buffer_size=int(1e5))
model.set_logger(A2cLogger(log_path, format))

for i in range(100):
    model.learn(total_timesteps=10_000) # 10_000
    print("Train step ended!")

    buffer = []
    obs = test_env.reset()
    length_of_episode = 1_000
    for i in range(length_of_episode):
        action, _states = model.predict(obs, deterministic=True)
        #torch_obs = torch.from_numpy(obs).unsqueeze(0).to(ENV_CFG.DEVICE)
        #log_proba = model.policy.evaluate_actions(torch_obs, torch.tensor(action).to(ENV_CFG.DEVICE))[1]
        #print(torch.exp(log_proba))
        
        obs, reward, done, info = test_env.step(action.item())

        buffer.append(np.expand_dims(obs[:, :, 0], 0))
        if done or i == length_of_episode-1:
            video_sequence = np.stack(buffer, axis = 0)
            print(video_sequence.shape)
            wandb.log({"Test video": wandb.Video(video_sequence, fps = ENV_CFG.FPS*6, format = "gif")})
            print("Test video sent")
            break

env.close()
