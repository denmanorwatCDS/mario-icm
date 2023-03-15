from gym_minigrid.envs import FourRoomsEnv
from minigrid_wrappers.imagedirection import RGBImgObsDirectionWrapper, FullyObsDirectionWrapper
from minigrid_wrappers.movementactions import MovementActions
from gym.wrappers import TimeLimit

import torch
import random
import gym
import numpy as np
from stable_baselines3.common.atari_wrappers import WarpFrame
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike
from loggers.logger_callback import LoggerCallback
from loggers.eval_callback import LoggerEvalCallback
from loggers.a2c_logger import A2CLogger
from loggers.global_counter import GlobalCounter
from stable_baselines_intrinsic.intrinsic_a2c import intrinsic_A2C
from icm_mine.icm import ICM

from nes_py.wrappers import JoypadSpace
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv
from wrappers.LastAndSkipEnv import LastAndSkipEnv
import gym_super_mario_bros

from config import log_config
from config.compressed_config import environment_config, a2c_config, icm_config, hyperparameters
from agents.neural_network_ext import ActorCritic
from environments.empty_limited import EmptyEnvLimited

if environment_config.SEED != -1:
    torch.manual_seed(environment_config.SEED)
    random.seed(environment_config.SEED)
    np.random.seed(environment_config.SEED)

def make_env(env_id, rank, seed=0):
    def _init():
        env = gym.make(env_id)
        env.seed(seed + rank)
        env = FullyObsDirectionWrapper(env)
        env = RGBImgObsDirectionWrapper(env)
        env = MovementActions(env)
        env = TimeLimit(env, 100)
        env = WarpFrame(env, width=42, height=42)
        return env
    set_random_seed(seed)
    return _init

if __name__=="__main__":
    parallel_envs = 20 # 20
    env_id = "MiniGrid-EmptyEnvLimited-v0" # SuperMarioBros
    global_counter = GlobalCounter()
    icm = ICM(4, environment_config.TEMPORAL_CHANNELS, 
              icm_config.INVERSE_SCALE, icm_config.FORWARD_SCALE, use_softmax=False, 
              hidden_layer_neurons=icm_config.HIDDEN_LAYERS, eta=icm_config.ETA, 
              feature_map_qty=icm_config.FMAP_QTY)\
                .to("cuda:0" if torch.cuda.is_available() else "cpu")

    # Eval and train environments
    env = SubprocVecEnv([make_env(env_id, i) for i in range(parallel_envs)], start_method="forkserver")
    env = VecFrameStack(env, n_stack = environment_config.TEMPORAL_CHANNELS)

    eval_env = SubprocVecEnv([make_env(env_id, 256)])
    eval_env = VecFrameStack(eval_env, n_stack = environment_config.TEMPORAL_CHANNELS)
    
    policy_kwargs = a2c_config.POLICY_KWARGS

    model = intrinsic_A2C(policy="CnnPolicy", env=env, motivation_model=icm, motivation_lr=icm_config.LR, 
                          motivation_grad_norm=icm_config.GRAD_NORM, intrinsic_reward_coef=icm_config.INTRINSIC_REWARD_COEF,
                          warmup_steps=icm_config.WARMUP, global_counter=global_counter, learning_rate=a2c_config.LR, 
                          n_steps=a2c_config.NUM_STEPS, gamma=a2c_config.GAMMA, gae_lambda=a2c_config.GAE_LAMBDA, 
                          ent_coef=a2c_config.ENTROPY_COEF, vf_coef=a2c_config.VALUE_LOSS_COEF, max_grad_norm=a2c_config.MAX_GRAD_NORM, 
                          use_rms_prop=a2c_config.RMS_PROP, verbose=1, policy_kwargs=policy_kwargs, seed=environment_config.SEED)


    model.set_logger(A2CLogger(log_config.LOSS_LOG_FREQUENCY, None, "stdout", global_counter = global_counter))
    model.learn(total_timesteps=float(1e8), callback=[LoggerCallback(log_config.AGENT_LOG_FREQUENCY, 0, "Minigrid A2C", 
                                                                     hyperparameters.HYPERPARAMS, global_counter = global_counter, 
                                                                     num_agents = a2c_config.NUM_AGENTS), 
                                                      LoggerEvalCallback(eval_env=eval_env, eval_freq=20_000, 
                                                                         global_counter=global_counter)])
