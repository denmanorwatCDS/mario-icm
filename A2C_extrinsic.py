import numpy as np
import torch
import random
import gym
import os
import vizdoom
from loggers.logger_callback import LoggerCallback
from loggers.eval_callback import LoggerEvalCallback
from loggers.a2c_logger import A2CLogger
from loggers.global_counter import GlobalCounter
from stable_baselines_intrinsic.intrinsic_a2c_doom import intrinsic_A2C
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv
from icm_mine.icm import ICM
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.atari_wrappers import AtariWrapper

from config import log_config
from config.compressed_config import environment_config, a2c_config, icm_config, hyperparameters

from doom_samples.utils.wrapper import ObservationWrapper
from stable_baselines3.common.env_util import make_vec_env
from vizdoom.gym_wrapper.base_gym_env import VizdoomEnv
from gym.wrappers import FrameStack

if environment_config.SEED != -1:
    torch.manual_seed(environment_config.SEED)
    random.seed(environment_config.SEED)
    np.random.seed(environment_config.SEED)


def prepare_env(seed, rank):
    def wrap_env():
        env = VizdoomEnv("/home/dvasilev/doom_icm/mario_icm/custom_my_way_home.cfg", frame_skip=4)
        env.reset(seed=seed+rank)
        env = Monitor(env, filename=None)
        env = ObservationWrapper(env)
        return env

    set_random_seed(seed)
    return wrap_env

if __name__=="__main__":
    parallel_envs = 20 # 20
    envpool_env_id = "VizdoomCustom-v1" # SuperMarioBros
    global_counter = GlobalCounter()
    print(vizdoom.scenarios_path)

    # Eval and train environments
    env = SubprocVecEnv([prepare_env(0, i) for i in range(parallel_envs)])
    
    print(env.action_space.n)
    icm = ICM(env.action_space.n, environment_config.TEMPORAL_CHANNELS, 
              icm_config.INVERSE_SCALE, icm_config.FORWARD_SCALE, use_softmax=False, 
              hidden_layer_neurons=icm_config.HIDDEN_LAYERS, eta=icm_config.ETA, 
              feature_map_qty=icm_config.FMAP_QTY)\
                .to("cuda:0" if torch.cuda.is_available() else "cpu")

    policy_kwargs = a2c_config.POLICY_KWARGS

    model = intrinsic_A2C(policy="CnnPolicy", env=env, motivation_model=icm, motivation_lr=icm_config.LR,
                          motivation_grad_norm=icm_config.GRAD_NORM, intrinsic_reward_coef=icm_config.INTRINSIC_REWARD_COEF,
                          warmup_steps=icm_config.WARMUP, global_counter=global_counter, learning_rate=a2c_config.LR,
                          n_steps=a2c_config.NUM_STEPS, gamma=a2c_config.GAMMA, gae_lambda=a2c_config.GAE_LAMBDA,
                          ent_coef=a2c_config.ENTROPY_COEF, vf_coef=a2c_config.VALUE_LOSS_COEF, max_grad_norm=a2c_config.MAX_GRAD_NORM,
                          use_rms_prop=a2c_config.RMS_PROP, verbose=1, policy_kwargs=policy_kwargs,
                          device=environment_config.MODEL_DEVICE, motivation_device=environment_config.MOTIVATION_DEVICE)


    model.set_logger(A2CLogger(log_config.LOSS_LOG_FREQUENCY/a2c_config.NUM_STEPS, None, "stdout", global_counter = global_counter))
    model.learn(total_timesteps=float(1e8), callback=[LoggerCallback(0, "Doom report", hyperparameters.HYPERPARAMS,
                                                                     global_counter = global_counter,
                                                                     quantity_of_agents = a2c_config.NUM_AGENTS,
                                                                     log_frequency = log_config.AGENT_LOG_FREQUENCY,
                                                                     video_submission_frequency=log_config.VIDEO_SUBMISSION_FREQUENCY,
                                                                     device=environment_config.MOTIVATION_DEVICE,
                                                                     fps=environment_config.FPS)])
