import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import gym
import stable_baselines3
import torch
import multiprocessing as mp

from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv
from stable_baselines3.common.env_checker import check_env

from Callbacks.my_callback import CustomCallback
from Callbacks.my_eval_callback import CustomEvalCallback
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike
from Config import ENV_CFG, ICM_CFG, A2C_CFG
from Environment_wrappers.wrappers import ResizeAndGrayscale, IntrinsicWrapper
from Environment_wrappers.Async_vec_env import asyncronous_make_vec_env
from Logger.my_logger import A2CLogger
from Agents.neural_network import ActorCritic
from Environment_wrappers.TensorConvertion import TensorConvertionManager

from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
import wandb

from ICM.Motivation_process import MotivationProcess
from ICM.ICM import ICM
from ICM.ICM_buffer import ICMBuffer
from torch import optim

def atari_wrapper(env, rank, connections_to_model, clip_reward = True):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    env = JoypadSpace(env, ENV_CFG.ALL_ACTION_SPACE)
    env = MaxAndSkipEnv(env, skip=ENV_CFG.ACTION_SKIP)
    env = ResizeAndGrayscale(env, ENV_CFG.RESIZED_SIZE, ENV_CFG.TEMPORAL_CHANNELS)
    print(rank)
    print(connections_to_model[rank])
    if icm is not None:
        env = IntrinsicWrapper(env, ENV_CFG.ACTION_SPACE_SIZE, 
        connection_to_model=connections_to_model[rank], motivation_only=True,
        beta_coef=ICM_CFG.BETA, fps=ENV_CFG.FPS)
    return env

if __name__=="__main__":
    wandb.init(project = "Cartpole")
    parallel_envs = A2C_CFG.NUM_PROCESSES

    pipes = [mp.Pipe() for i in range(A2C_CFG.NUM_PROCESSES)]
    pipe_ends1 = [pipe[0] for pipe in pipes]
    pipe_ends2 = [pipe[1] for pipe in pipes]

    icm = ICM(ENV_CFG.ACTION_SPACE_SIZE, temporal_channels = ENV_CFG.TEMPORAL_CHANNELS,
    hidden_layer_neurons=ICM_CFG.HIDDEN_LAYERS, eta = ICM_CFG.ETA, feature_map_qty=ICM_CFG.FMAP_QTY
    ).to(ENV_CFG.DEVICE).train()
    icm_optimizer = optim.Adam(globals()["icm"].parameters(), lr=ICM_CFG.LR)
    icm_model = MotivationProcess(icm, icm_optimizer, pipe_ends1)
    icm_model.start()
    # Parallel environments
    # Add test environment! For comparability with stable-baselines3 baselines
    # env = SubprocVecEnv([make_env("SpaceInvadersNoFrameskip-v4", seed = i) for i in range(parallel_envs)])
    
    # Add, analogous to seed, different pipe_ends for different processes
    env = asyncronous_make_vec_env("SuperMarioBros-v0", n_envs=parallel_envs, seed=0, 
                       vec_env_cls=SubprocVecEnv, wrapper_class=atari_wrapper,
                       wrapper_kwargs={"connections_to_model": pipe_ends2})
    print(env.reset().shape)
    #print(env.step(torch.ones(20)).shape)
    #env = VecNormalize(env, norm_reward=True, training=True)
    #env = VecFrameStack(env, n_stack=ENV_CFG.TEMPORAL_CHANNELS) - WARNING with this all has been working

    #eval_env = SubprocVecEnv([make_env("SpaceInvadersNoFrameskip-v4", clip_reward=False, seed = i) for i in range(1)])
    
    # eval_env = make_vec_env("SuperMarioBros-v0", n_envs=1, seed=1001, 
    #                        vec_env_cls=SubprocVecEnv, wrapper_class=atari_wrapper, 
    #                        wrapper_kwargs={"clip_reward": False})
    
    #eval_env = VecNormalize(eval_env, norm_reward=False, training=False)
    #eval_env = VecFrameStack(eval_env, n_stack=4) - WARNING with this all has been working
    callback = CustomCallback(parallel_envs=parallel_envs, action_space_size=env.action_space.n)
    
    #eval_callback =\
    #    CustomEvalCallback(eval_env=eval_env, eval_freq=5000, parallel_envs=1, 
    #                       action_space_size=eval_env.action_space.n)

    policy_kwargs = {"optimizer_class": RMSpropTFLike, "optimizer_kwargs": {"eps": 1e-05},
                     "features_extractor_class": ActorCritic, 
                     "net_arch": [dict(pi=[A2C_CFG.POLICY_NEURONS], vf=[A2C_CFG.VALUE_NEURONS])]}
    model = A2C("CnnPolicy", env, verbose=1, ent_coef=0.01, vf_coef=0.25, policy_kwargs=policy_kwargs)

    log_path = 'sb3_logs'
    format = 'tensorboard'
    model.set_logger(A2CLogger(log_path, format))

    model.learn(total_timesteps=float(1e8), callback=[callback])
