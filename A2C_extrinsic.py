from environments.crossing_limited import FourRoomsEnvLimited
from minigrid_wrappers.imagedirection import RGBImgObsDirectionWrapper
from minigrid_wrappers.movementactions import MovementActions
from gym.wrappers import TimeLimit

import torch
import random
import gym
import numpy as np
from stable_baselines3.common.atari_wrappers import WarpFrame
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.utils import set_random_seed
from loggers.logger_callback import LoggerCallback
from loggers.eval_callback import LoggerEvalCallback
from loggers.a2c_logger import A2CLogger
from loggers.global_counter import GlobalCounter
from loggers.prepare_maps import prepare_maps
from stable_baselines_intrinsic.intrinsic_a2c import intrinsic_A2C
from icm_mine.icm import ICM

from config import log_config
from config.compressed_config import environment_config, a2c_config, icm_config, hyperparameters

if environment_config.SEED != -1:
    torch.manual_seed(environment_config.SEED)
    random.seed(environment_config.SEED)
    np.random.seed(environment_config.SEED)

def make_env(env_id, grid_size, rank, seed=0):
    def _init(agent_y_pos=1, agent_x_pos=1):
        env = gym.make(env_id, grid_size=grid_size, agent_pos=(agent_y_pos, agent_x_pos), goal_pos=(grid_size-2, grid_size-2))
        env.seed(seed + rank)
        env = RGBImgObsDirectionWrapper(env, grid_size=grid_size, restriction=None)
        env = MovementActions(env)
        env = TimeLimit(env, 50)
        env = WarpFrame(env, width=42, height=42)
        return env
    set_random_seed(seed)
    return _init

if __name__=="__main__":
    parallel_envs = a2c_config.NUM_AGENTS # 20
    grid_size = 16
    env_id = "MiniGrid-FourRoomsEnvLimited-v0" # SuperMarioBros
    prepare_maps(make_env(env_id, grid_size, 0, 0))
    global_counter = GlobalCounter()
    icm = ICM(4, environment_config.TEMPORAL_CHANNELS, 
              icm_config.INVERSE_SCALE, icm_config.FORWARD_SCALE, use_softmax=False, 
              hidden_layer_neurons=icm_config.HIDDEN_LAYERS, eta=icm_config.ETA, 
              feature_map_qty=icm_config.FMAP_QTY)\
                .to(environment_config.DEVICE)

    # Eval and train environments
    env = SubprocVecEnv([make_env(env_id, grid_size, i) for i in range(parallel_envs)], start_method="forkserver")

    eval_env = SubprocVecEnv([make_env(env_id, grid_size, 256)])
    
    policy_kwargs = a2c_config.POLICY_KWARGS

    model = intrinsic_A2C(policy="CnnPolicy", env=env, motivation_model=icm, motivation_lr=icm_config.LR, 
                          motivation_grad_norm=icm_config.GRAD_NORM, intrinsic_reward_coef=icm_config.INTRINSIC_REWARD_COEF,
                          warmup_steps=icm_config.WARMUP, global_counter=global_counter, learning_rate=a2c_config.LR, 
                          n_steps=a2c_config.NUM_STEPS, gamma=a2c_config.GAMMA, gae_lambda=a2c_config.GAE_LAMBDA, 
                          ent_coef=a2c_config.ENTROPY_COEF, vf_coef=a2c_config.VALUE_LOSS_COEF, max_grad_norm=a2c_config.MAX_GRAD_NORM, 
                          use_rms_prop=a2c_config.RMS_PROP, verbose=1, policy_kwargs=policy_kwargs, seed=environment_config.SEED)


    model.set_logger(A2CLogger(log_config.LOSS_LOG_FREQUENCY/a2c_config.NUM_STEPS, None, "stdout", global_counter = global_counter))
    model.learn(total_timesteps=float(1e8), callback=[LoggerCallback(0, "Minigrid report", hyperparameters.HYPERPARAMS,
                                                                     global_counter = global_counter,
                                                                     quantity_of_agents = a2c_config.NUM_AGENTS, grid_size=grid_size,  
                                                                     log_frequency = log_config.AGENT_LOG_FREQUENCY,
                                                                     video_submission_frequency=log_config.VIDEO_SUBMISSION_FREQUENCY,
                                                                     device=environment_config.DEVICE)])
