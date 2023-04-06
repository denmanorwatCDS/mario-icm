import numpy as np
import torch
import random
import gym
import os
import vizdoom
from stable_baselines3.common.atari_wrappers import WarpFrame
from stable_baselines3.common.utils import set_random_seed
from loggers.logger_callback import LoggerCallback
from loggers.eval_callback import LoggerEvalCallback
from loggers.a2c_logger import A2CLogger
from loggers.global_counter import GlobalCounter
from stable_baselines_intrinsic.intrinsic_a2c_doom import intrinsic_A2C
from stable_baselines_intrinsic.intrinsic_ppo_doom import Intrinsic_PPO
from icm_mine.icm import ICM

from wrappers.LastAndSkipEnv import LastAndSkipEnv

from config import log_config
from config.compressed_config import environment_config, a2c_config, icm_config, hyperparameters
from agents.neural_network_ext import ActorCritic
import envpool
from envpool_to_sb3.vec_adapter import VecAdapter
import envpool_to_sb3

os.environ["CUDA_LAUNCH_BLOCKING"] = '1'
if environment_config.SEED != -1:
    torch.manual_seed(environment_config.SEED)
    random.seed(environment_config.SEED)
    np.random.seed(environment_config.SEED)

if __name__=="__main__":
    parallel_envs = 20 # 20
    envpool_env_id = "VizdoomCustom-v1" # SuperMarioBros
    global_counter = GlobalCounter()
    print(vizdoom.scenarios_path)

    # Eval and train environments
    env = envpool.make(envpool_env_id, env_type="gym", num_envs=parallel_envs, seed=environment_config.SEED,
                       img_height=environment_config.RESIZED_SIZE[0], img_width=environment_config.RESIZED_SIZE[1],
                       stack_num=environment_config.TEMPORAL_CHANNELS, frame_skip=environment_config.ACTION_SKIP,
                       use_combined_action=True,
                       cfg_path="/home/dvasilev/doom_icm/mario_icm/custom_my_way_home.cfg",
                       wad_path="/home/dvasilev/doom_icm/mario_icm/maps/my_way_home_dense.wad",
                       reward_config={"ARMOR": [0., 0.]})
    env.spec.id = envpool_env_id
    env = VecAdapter(env)
    
    print(env.action_space.n)
    icm = ICM(env.action_space.n, environment_config.TEMPORAL_CHANNELS, 
              icm_config.INVERSE_SCALE, icm_config.FORWARD_SCALE, use_softmax=False, 
              hidden_layer_neurons=icm_config.HIDDEN_LAYERS, eta=icm_config.ETA, 
              feature_map_qty=icm_config.FMAP_QTY)\
                .to("cuda:0" if torch.cuda.is_available() else "cpu")

    policy_kwargs = a2c_config.POLICY_KWARGS

    model = intrinsic_A2C(policy="CnnPolicy", env=env, motivation_model=icm,
                          global_counter=global_counter, motivation_lr=icm_config.LR,
                          motivation_grad_norm=icm_config.GRAD_NORM, intrinsic_reward_coef=icm_config.INTRINSIC_REWARD_COEF,
                          warmup_steps=0,
                          policy_kwargs=policy_kwargs, seed=environment_config.SEED,
                          device=environment_config.MODEL_DEVICE, motivation_device=environment_config.MOTIVATION_DEVICE)


    model.set_logger(A2CLogger(log_config.LOSS_LOG_FREQUENCY/a2c_config.NUM_STEPS, None, "stdout", global_counter = global_counter))
    model.learn(total_timesteps=float(1e8), callback=[LoggerCallback(0, "Doom report", hyperparameters.HYPERPARAMS,
                                                                     global_counter = global_counter,
                                                                     quantity_of_agents = a2c_config.NUM_AGENTS,
                                                                     log_frequency = log_config.AGENT_LOG_FREQUENCY,
                                                                     video_submission_frequency=log_config.VIDEO_SUBMISSION_FREQUENCY,
                                                                     device=environment_config.MOTIVATION_DEVICE,
                                                                     fps=environment_config.FPS)])
