import numpy as np
import torch
import random
import gym
import os
import vizdoom
from vizdoom import gym_wrapper
from stable_baselines3.common.atari_wrappers import WarpFrame
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.utils import set_random_seed
from loggers.logger_callback import LoggerCallback
from loggers.eval_callback import LoggerEvalCallback
from loggers.a2c_logger import A2CLogger
from loggers.global_counter import GlobalCounter
from stable_baselines_intrinsic.intrinsic_a2c_doom import intrinsic_A2C
from icm_mine.icm import ICM

from wrappers.LastAndSkipEnv import LastAndSkipEnv

from config import log_config
from config.compressed_config import environment_config, a2c_config, icm_config, hyperparameters
from agents.neural_network_ext import ActorCritic
import envpool
from envpool_to_sb3.vec_adapter import VecAdapter
import envpool_to_sb3
import wandb
import time

def make_env(env_id, rank, seed=0):
    def _init():
        env = gym.make(env_id)
        env.seed(seed + rank)
        env = WarpFrame(env, width=42, height=42)
        env = LastAndSkipEnv(env, skip=environment_config.ACTION_SKIP)
        return env
    set_random_seed(seed)
    return _init

if __name__=="__main__":
    wandb.init(project="Speed evaluation")
    parallel_envs = 6
    envpool_env_id = "MyWayHome-v1" # SuperMarioBros
    global_counter = GlobalCounter()

    env = envpool.make(envpool_env_id, env_type="gym", num_envs=parallel_envs, seed=environment_config.SEED,
                       img_height = environment_config.RESIZED_SIZE[0], img_width = environment_config.RESIZED_SIZE[1],
                       stack_num=4, frame_skip=4, use_combined_action=True, 
                       cfg_path="/home/dvasilev/doom_icm/mario_icm/custom_my_way_home.cfg",
                       wad_path="/home/dvasilev/doom_icm/mario_icm/maps/my_way_home_dense.wad",
                       reward_config={"ARMOR": [0.01, 0.]})
    env.spec.id = envpool_env_id
    env = VecAdapter(env)

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
