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
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike
from loggers.logger_callback import LoggerCallback
from loggers.eval_callback import LoggerEvalCallback
from loggers.a2c_logger import A2CLogger
from loggers.global_counter import GlobalCounter


from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros

from config import environment_config, a2c_config
from agents.neural_network_ext import ActorCritic
from agents.copied_feature_extractor import CustomCNN, NatureCNN

if environment_config.SEED != -1:
    torch.manual_seed(environment_config.SEED)
    random.seed(environment_config.SEED)
    np.random.seed(environment_config.SEED)

def make_env(env_id, rank, seed=0):
    def _init():
        env = gym.make(env_id)
        env.seed(seed + rank)
        env = JoypadSpace(env, environment_config.ALL_ACTION_SPACE)
        env = WarpFrame(env, width=42, height=42)
        env = MaxAndSkipEnv(env, skip=environment_config.ACTION_SKIP)
        return env
    set_random_seed(seed)
    return _init


if __name__=="__main__":
    parallel_envs = a2c_config.NUM_AGENTS # 20
    env_id = "SuperMarioBros-1-1-v0" # SuperMarioBros
    global_counter = GlobalCounter()

    # Eval and train environments
    env = SubprocVecEnv([make_env(env_id, i) for i in range(parallel_envs)], start_method="forkserver")
    env = VecFrameStack(env, n_stack = 4)

    eval_env = SubprocVecEnv([make_env(env_id, 256)])
    eval_env = VecFrameStack(eval_env, n_stack = 4)
    
    policy_kwargs = dict(optimizer_class=RMSpropTFLike, optimizer_kwargs=dict(eps=1e-5))

    model = A2C("CnnPolicy", env,
                verbose=1, policy_kwargs=policy_kwargs,
                seed=environment_config.SEED, vf_coef=0.25, ent_coef=0.01)

    model.set_logger(A2CLogger(None, "stdout", global_counter = global_counter))
    model.learn(total_timesteps=float(1e8), callback=[LoggerCallback(0, "Extrinsic A2C", None, global_counter = global_counter), 
                                                      LoggerEvalCallback(eval_env=eval_env, eval_freq=20_000, 
                                                                         global_counter=global_counter)])
