from torch import optim
import gym_super_mario_bros
import gym
from nes_py.wrappers import JoypadSpace
from Environment_wrappers.wrappers import ResizeAndGrayscale, IntrinsicWrapper

from ICM.ICM import ICM
from ICM.ICM_buffer import ICMBuffer
from Agents.neural_network import ActorCritic
from Logger.custom_sb3_loggers import A2cLogger

from stable_baselines3.a2c.a2c import A2C
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv
from Logger.custom_sb3_loggers import A2cLogger

from Config import ENV_CFG, ICM_CFG, A2C_CFG
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

run = wandb.init(project = "Mario", config = HYPERPARAMS)


env = gym.make('SuperMarioBros-v0')
env.seed(ENV_CFG.SEED) # New
env = JoypadSpace(env, ENV_CFG.ALL_ACTION_SPACE)
action_dim = env.action_space.n
icm = ICM(action_dim, temporal_channels = ENV_CFG.TEMPORAL_CHANNELS, 
    hidden_layer_neurons=ICM_CFG.HIDDEN_LAYERS, eta = ICM_CFG.ETA, feature_map_qty=ICM_CFG.FMAP_QTY
    ).to(ENV_CFG.DEVICE).train()
icm_optimizer = optim.Adam(icm.parameters(), lr=ICM_CFG.LR)
icm_buffer = ICMBuffer(ICM_CFG.BATCH_SIZE, ICM_CFG.BUFFER_SIZE)

env = MaxAndSkipEnv(env, skip=ENV_CFG.ACTION_SKIP)
env = ResizeAndGrayscale(env, ENV_CFG.RESIZED_SIZE, ENV_CFG.TEMPORAL_CHANNELS)
env = IntrinsicWrapper(env, ENV_CFG.ACTION_SPACE_SIZE, motivation_model = icm, optimizer_of_model = icm_optimizer, 
                       buffer = icm_buffer, motivation_only = True, beta_coef=ICM_CFG.BETA,
                       fps = ENV_CFG.FPS)

policy_kwargs = dict(
    features_extractor_class=ActorCritic,
    net_arch=[dict(pi=[A2C_CFG.POLICY_NEURONS], vf=[A2C_CFG.VALUE_NEURONS])]
)

#cnn_policy = CnnPolicy()
log_path = 'sb3_logs'
format = 'tensorboard'
model = A2C("CnnPolicy", env, verbose=1, learning_rate=A2C_CFG.LR, use_rms_prop=A2C_CFG.RMS_PROP, 
            policy_kwargs=policy_kwargs, n_steps=A2C_CFG.NUM_STEPS, seed=ENV_CFG.SEED, 
            max_grad_norm=A2C_CFG.MAX_GRAD_NORM, gamma=A2C_CFG.GAMMA, vf_coef=A2C_CFG.VALUE_LOSS_COEF,
            ent_coef=A2C_CFG.ENTROPY_COEF, gae_lambda=A2C_CFG.GAE_LAMBDA)
model.set_logger(A2cLogger(log_path, format))

for i in range(100):
    model.learn(total_timesteps=10_000)
    
    print("Train step ended!")

    buffer = []
    obs = env.reset()
    length_of_episode = 1000
    for i in range(length_of_episode):
        action, _states = model.predict(obs, deterministic=True)
        
        obs, reward, done, info = env.step(action.item())

        buffer.append(np.expand_dims(obs[:, :, 0], 0))
        if done or i == length_of_episode-1:
            video_sequence = np.stack(buffer, axis = 0)
            print(video_sequence.shape)
            wandb.log({"Test video": wandb.Video(video_sequence, fps = ENV_CFG.FPS, format = "gif")})
            print("Test video sent")
            break

env.close()