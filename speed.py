from environments.crossing_limited import FourRoomsEnvLimited
from minigrid_wrappers.imagedirection import RGBImgObsDirectionWrapper
from minigrid_wrappers.movementactions import MovementActions
from gym.wrappers import TimeLimit

import time
import wandb
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
    wandb.init(project="Speed evaluation")
    parallel_envs = a2c_config.NUM_AGENTS # 20
    grid_size = 16
    env_id = "MiniGrid-FourRoomsEnvLimited-v0" # SuperMarioBros
    prepare_maps(make_env(env_id, grid_size, 0, 0))
    global_counter = GlobalCounter()

    env = SubprocVecEnv([make_env(env_id, grid_size, i) for i in range(parallel_envs)], start_method="forkserver")

    eval_env = SubprocVecEnv([make_env(env_id, grid_size, 256)])

    current_time = time.perf_counter()
    for i in range(50_000):
        if i == 0:
            env.reset()
        actions = np.random.randint(0, 4, (parallel_envs))
        new_obs, rewards, dones, infos = env.step(actions)
        if i%2_000==0 and i>0:
            new_time = time.perf_counter()
            wandb.log({"Elapsed time": new_time-current_time}, step=i)
            current_time = new_time
