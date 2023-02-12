import os
#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import gym
import stable_baselines3
import torch

from stable_baselines3.a2c.a2c import A2C
from stable_baselines_motivation.A2C_with_ICM import A2CWithMotivation
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv

from Callbacks.my_callback import CustomCallback
from Callbacks.my_eval_callback import CustomEvalCallback
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike
from Config import ENV_CFG, ICM_CFG, A2C_CFG
from Config.all_hyperparams_dict import HYPERPARAMS
from Environment_wrappers.wrappers import ResizeAndGrayscale
from Logger.my_logger import A2CLogger
from Agents.neural_network import ActorCritic

from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros

from ICM.ICM import ICM
from ICM.ICM_buffer import ICMBuffer
from torch import optim
import faulthandler
import signal



def atari_wrapper(env, clip_reward = True):
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

    return env

if __name__=="__main__":
    print(os.getpid()) # 2084532
    faulthandler.enable()
    faulthandler.register(signal.SIGUSR1.value)
    parallel_envs = A2C_CFG.NUM_AGENTS # 20
    
    # Parallel environments
    icm = ICM(ENV_CFG.ACTION_SPACE_SIZE, temporal_channels = ENV_CFG.TEMPORAL_CHANNELS,
    hidden_layer_neurons=ICM_CFG.HIDDEN_LAYERS, eta = ICM_CFG.ETA, feature_map_qty=ICM_CFG.FMAP_QTY
    ).to(ENV_CFG.DEVICE).train()

    # ICM
    icm_optimizer = optim.Adam(globals()["icm"].parameters(), lr=ICM_CFG.LR)
    icm_buffer = ICMBuffer(save_latest=ICM_CFG.SAVE_LATEST, buffer_size=ICM_CFG.BUFFER_SIZE, sample_size=ICM_CFG.BATCH_SIZE)
    
    # Eval and train environments
    env = make_vec_env("SuperMarioBros-v0", n_envs=parallel_envs, seed=0, 
                       vec_env_cls=SubprocVecEnv, wrapper_class=atari_wrapper)
    
    eval_env = make_vec_env("SuperMarioBros-v0", n_envs=1, seed=1001, 
                            vec_env_cls=SubprocVecEnv, wrapper_class=atari_wrapper, 
                            wrapper_kwargs={"clip_reward": False})
    
    callback = CustomCallback(parallel_envs=parallel_envs, action_space_size=env.action_space.n, 
                              HYPERPARAMS=HYPERPARAMS)
    eval_callback =\
        CustomEvalCallback(eval_env=eval_env, eval_freq=50000, parallel_envs=1, 
                           action_space_size=eval_env.action_space.n)

    # A2C parameters
    policy_kwargs = {"features_extractor_class": ActorCritic, 
                     "net_arch": [dict(pi=[A2C_CFG.POLICY_NEURONS], vf=[A2C_CFG.VALUE_NEURONS])]}
    
    model = A2C("CnnPolicy", env,
                verbose=1, learning_rate=A2C_CFG.LR, use_rms_prop=A2C_CFG.RMS_PROP, 
                policy_kwargs=policy_kwargs, n_steps=A2C_CFG.NUM_STEPS, seed=ENV_CFG.SEED, 
                max_grad_norm=A2C_CFG.MAX_GRAD_NORM, gamma=A2C_CFG.GAMMA, vf_coef=A2C_CFG.VALUE_LOSS_COEF,
                ent_coef=A2C_CFG.ENTROPY_COEF, gae_lambda=A2C_CFG.GAE_LAMBDA)
    """
    model = A2CWithMotivation("CnnPolicy", env, forward_backward_motivation=icm, motivation_buffer=icm_buffer,
                motivation_optim=icm_optimizer, action_space_size=ENV_CFG.ACTION_SPACE_SIZE,
                beta=ICM_CFG.BETA, reward_type="Extrinsic",
                verbose=1, learning_rate=A2C_CFG.LR, use_rms_prop=A2C_CFG.RMS_PROP, 
                policy_kwargs=policy_kwargs, n_steps=A2C_CFG.NUM_STEPS, seed=ENV_CFG.SEED, 
                max_grad_norm=A2C_CFG.MAX_GRAD_NORM, gamma=A2C_CFG.GAMMA, vf_coef=A2C_CFG.VALUE_LOSS_COEF,
                ent_coef=A2C_CFG.ENTROPY_COEF, gae_lambda=A2C_CFG.GAE_LAMBDA)
    """
    log_path = 'sb3_logs'
    format = 'tensorboard'
    model.set_logger(A2CLogger(log_path, format, num_workers=parallel_envs))

    model.learn(total_timesteps=float(1e8), callback=[callback, eval_callback])
