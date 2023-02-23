import torch
import random
import numpy as np
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv
from stable_baselines3.common.atari_wrappers import WarpFrame
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.a2c.a2c import A2C

from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros

from config import environment_config, a2c_config

if environment_config.SEED != -1:
    torch.manual_seed(environment_config.SEED)
    random.seed(environment_config.SEED)
    np.random.seed(environment_config.SEED)
    #torch.backends.cudnn.benchmark = False
    #torch.use_deterministic_algorithms(mode = True)

def atari_wrapper(env, clip_reward = True):
    """
    Utility function for multiprocessed env.
    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    env = JoypadSpace(env, environment_config.ALL_ACTION_SPACE)
    env = WarpFrame(env, width=42, height=42)
    env = MaxAndSkipEnv(env, skip=environment_config.ACTION_SKIP)
    return env

if __name__=="__main__":
    parallel_envs = a2c_config.NUM_AGENTS # 20
    
    # Eval and train environments
    env = make_vec_env("SuperMarioBros-v0", n_envs=parallel_envs, seed=0, 
                       vec_env_cls=SubprocVecEnv, wrapper_class=atari_wrapper)
    env = VecFrameStack(env, n_stack = 4)
    
    
    model = A2C("CnnPolicy", env,
                verbose=1, learning_rate=a2c_config.LR, use_rms_prop=a2c_config.RMS_PROP, 
                n_steps=a2c_config.NUM_STEPS, seed=environment_config.SEED, 
                max_grad_norm=a2c_config.MAX_GRAD_NORM, gamma=a2c_config.GAMMA, vf_coef=a2c_config.VALUE_LOSS_COEF,
                ent_coef=a2c_config.ENTROPY_COEF, gae_lambda=a2c_config.GAE_LAMBDA)

    log_path = 'sb3_logs'
    format = 'tensorboard'

    model.learn(total_timesteps=float(1e8))
