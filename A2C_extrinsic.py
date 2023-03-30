from environments.fast_grid import GridWorld, FOUR_ROOMS_OBSTACLES
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

def make_env(seed=environment_config.SEED):
    def _init(agent_y_pos=1, agent_x_pos=1):
        grid_size = FOUR_ROOMS_OBSTACLES.shape
        env = GridWorld(grid_size=grid_size,
                       obstacle_mask=FOUR_ROOMS_OBSTACLES,
                       agent_pos=(agent_y_pos, agent_x_pos),
                       goal_pos=(grid_size[0]-2, grid_size[1]-2), pixel_size=8, time_limit=50,
                       color_map=dict(floor=.0, obstacle=.43, agent=.98, target=.8),
                       const_punish=0.02*0.9, terminal_decay=1.,
                       warp_size=(42, 42), beautiful=True)
        return env
    set_random_seed(seed)
    return _init

if __name__=="__main__":
    print(FOUR_ROOMS_OBSTACLES.shape)
    parallel_envs = a2c_config.NUM_AGENTS # 20
    grid_size = FOUR_ROOMS_OBSTACLES.shape[0]
    env_id = "MiniGrid-FastGridFourRooms-v0" # SuperMarioBros
    prepare_maps(make_env())
    global_counter = GlobalCounter()
    icm = ICM(4, environment_config.TEMPORAL_CHANNELS, 
              icm_config.INVERSE_SCALE, icm_config.FORWARD_SCALE, use_softmax=False, 
              hidden_layer_neurons=icm_config.HIDDEN_LAYERS, eta=icm_config.ETA, 
              feature_map_qty=icm_config.FMAP_QTY)\
                .to(environment_config.MOTIVATION_DEVICE)
    
    # Eval and train environments
    env = SubprocVecEnv([make_env() for i in range(parallel_envs)], start_method="forkserver")

    
    policy_kwargs = a2c_config.POLICY_KWARGS

    model = intrinsic_A2C(policy="CnnPolicy", env=env, motivation_model=icm, motivation_lr=icm_config.LR, 
                          motivation_grad_norm=icm_config.GRAD_NORM, intrinsic_reward_coef=icm_config.INTRINSIC_REWARD_COEF,
                          warmup_steps=icm_config.WARMUP, global_counter=global_counter, learning_rate=a2c_config.LR, 
                          n_steps=a2c_config.NUM_STEPS, gamma=a2c_config.GAMMA, gae_lambda=a2c_config.GAE_LAMBDA, 
                          ent_coef=a2c_config.ENTROPY_COEF, vf_coef=a2c_config.VALUE_LOSS_COEF, max_grad_norm=a2c_config.MAX_GRAD_NORM, 
                          use_rms_prop=a2c_config.RMS_PROP, verbose=1, policy_kwargs=policy_kwargs, seed=environment_config.SEED,
                          device=environment_config.MODEL_DEVICE, motivation_device=environment_config.MOTIVATION_DEVICE)


    model.set_logger(A2CLogger(log_config.LOSS_LOG_FREQUENCY/a2c_config.NUM_STEPS, None, "stdout", global_counter = global_counter))
    model.learn(total_timesteps=float(1e8), callback=[LoggerCallback(0, "Minigrid report", hyperparameters.HYPERPARAMS,
                                                                     global_counter=global_counter,
                                                                     quantity_of_agents=a2c_config.NUM_AGENTS, grid_size=grid_size,
                                                                     log_frequency=log_config.AGENT_LOG_FREQUENCY,
                                                                     video_submission_frequency=log_config.VIDEO_SUBMISSION_FREQUENCY,
                                                                     device=environment_config.MOTIVATION_DEVICE)])
