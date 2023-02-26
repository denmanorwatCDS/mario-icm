import torch
import random
import gym
import numpy as np
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv
from stable_baselines3.common.atari_wrappers import WarpFrame
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.a2c.a2c import A2C
from stable_baselines3.common.utils import set_random_seed

from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros

from config import environment_config, a2c_config
from agents.copied_feature_extractor import CustomCNN, NatureCNN

if environment_config.SEED != -1:
    torch.manual_seed(environment_config.SEED)
    random.seed(environment_config.SEED)
    np.random.seed(environment_config.SEED)

def make_env(env_id, rank, seed=0):
    def _init():
        env = gym.make(env_id)
        env.seed(seed + rank)
        #env = JoypadSpace(env, environment_config.ALL_ACTION_SPACE)
        env = WarpFrame(env, width=42, height=42)
        env = MaxAndSkipEnv(env, skip=environment_config.ACTION_SKIP)
        return env
    set_random_seed(seed)
    return _init


if __name__=="__main__":
    parallel_envs = a2c_config.NUM_AGENTS # 20
    env_id = "SpaceInvaders-v4"
    # Eval and train environments
    env = SubprocVecEnv([make_env(env_id, i) for i in range(parallel_envs)], start_method="forkserver")
    env = VecFrameStack(env, n_stack = 4)
    
    policy_kwargs = {"features_extractor_class": CustomCNN, 
                     "net_arch": [dict(pi=[a2c_config.POLICY_NEURONS], vf=[a2c_config.VALUE_NEURONS])]}

    model = A2C("CnnPolicy", env,
                verbose=1, learning_rate=a2c_config.LR, use_rms_prop=a2c_config.RMS_PROP, policy_kwargs=policy_kwargs,
                n_steps=a2c_config.NUM_STEPS, seed=environment_config.SEED, 
                max_grad_norm=a2c_config.MAX_GRAD_NORM, gamma=a2c_config.GAMMA, vf_coef=a2c_config.VALUE_LOSS_COEF,
                ent_coef=a2c_config.ENTROPY_COEF, gae_lambda=a2c_config.GAE_LAMBDA)

    model.learn(total_timesteps=float(1e8))
