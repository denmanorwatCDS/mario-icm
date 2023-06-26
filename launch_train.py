import os
import numpy as np
import torch
import random
import vizdoom
from loggers.logger_callback import LoggerCallback
from loggers.agent_logger import AgentLogger
from loggers.global_counter import GlobalCounter
from stable_baselines_intrinsic.intrinsic_ppo_doom import intrinsic_PPO
from stable_baselines_intrinsic.intrinsic_a2c_doom import intrinsic_A2C
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv
from icm_mine.icm import ICM
from icm_mine.icm import InverseModel, ForwardModel, FeatureExtractor
from stable_baselines3.common.monitor import Monitor

from mario_icm.config import ppo_config, a2c_config, environment_config, icm_config, log_config

from MuJoCo_custom.EgocentricCylinderMaze import EgocentricCylinderMazeEnv
from MuJoCo_custom.EgocentricAntMaze import EgocentricAntMazeEnv
from MuJoCo_custom.ObservationWrapper import ObservationWrapper as MuJoCoObsWrapper

from mario_icm.ViZDoom.ViZDoom_continuous_support.ViZDoomEnv_gymnasium import VizdoomEnv
from mario_icm.ViZDoom.utils.wrapper_gymnasium import ObservationWrapper
from gymnasium.wrappers import TimeLimit
import random
from time import sleep

import wandb

def prepare_env(seed, rank, discrete = True, vizdoom = True, ant = False, hard = False):
    def wrap_env():
        # Important! Else XML bugs could occur
        sleep_time = rank/5
        sleep(sleep_time)
        # Important block ends. If XML bug returns - try increase sleep time.
        if not vizdoom:
            print("ViZDoom == False => launching MuJoCo")
            assert discrete is False, "MuJoCo could be only in continuous action space"
            if ant == False:
                print("Ant == False => launching cylinder maze")
                env = EgocentricCylinderMazeEnv(render_mode="rgb_array", continuing_task=False, hard=hard)
            else:
                print("Ant == True => launching ant maze")
                assert hard is False, "In AntMaze there is no hard mode"
                env = EgocentricAntMazeEnv(render_mode="rgb_array", continuing_task=False)
            env.reset(seed=seed+rank)
            env = TimeLimit(env, max_episode_steps=1000)
            env = MuJoCoObsWrapper(env)
            return env

        assert ant is False, "In ViZDoom there is no ant's"
        assert hard is False, "In ViZDoom there is no hard mode"
        if discrete:
            env = VizdoomEnv("/home/dvasilev/doom_icm/mario_icm/ViZDoom/custom_my_way_home_discrete.cfg", frame_skip=4)
        else:
            env = VizdoomEnv("/home/dvasilev/doom_icm/mario_icm/ViZDoom/custom_my_way_home_continuous.cfg", frame_skip=4)
        env.reset(seed=seed+rank)
        env = Monitor(env, filename=None)
        env = ObservationWrapper(env)
        return env

    set_random_seed(seed)
    return wrap_env

sweep_configuration = {
    'method': 'grid',
    'name': 'Doom sparse sweep',
    'parameters':
    {
        'entropy': {'values': [0]}, # 1e-02, 1e-03
        'seed': {'values': [110, 220, 330]},
        'intrinsic_reward_coef': {"values": [icm_config.INTRINSIC_REWARD_COEF, 0.]},
        'discrete': {"values": [False, True]},
        'use_ppo': {"values": [True, False]}
     }
}

def main(config = None):
    wandb.init()
    if config is None:
        config = wandb.config
    if wandb.config.discrete == False and wandb.config.intrinsic_reward_coef == 1:
        return None
    elif wandb.config.discrete == False and wandb.config.intrinsic_reward_coef == 0 and wandb.config.seed <= 220:
        return None
     
    torch.manual_seed(config.seed) #wandb.config.seed
    random.seed(config.seed) # wandb.config.seed
    np.random.seed(config.seed) # wandb.config.seed
    parallel_envs = ppo_config.NUM_AGENTS  # 20
    discrete = config.discrete
    vizdoom = True
    use_ppo = config.use_ppo
    hard = False
    ant = False
    global_counter = GlobalCounter()

    # Eval and train environments
    env = SubprocVecEnv([prepare_env(config.seed, i, discrete, vizdoom=vizdoom, hard=hard, ant=ant) for i in range(parallel_envs)],
                        start_method="spawn") # wandb.config.seed

    action_space = env.action_space
    obs_shape = (1, 42, 42)
    feature_extractor = FeatureExtractor(obs_shape, batch_norm=False, skip_conn=False, consecutive_convs=1,
                                         total_blocks=4, feature_map_size=32)
    input_shape = feature_extractor.simple_encoder.get_output_shape()
    state_dim = torch.tensor(input_shape).prod()
    inverse_model = InverseModel(input_shape, action_space, discrete=discrete, pde=False, apply_bounder=False,
                                 group=False, bottleneck=False, fc_qty=2)
    forward_model = ForwardModel(state_dim, action_space, discrete=discrete, hidden_layer_neurons=256)
    icm = ICM(inv_scale=icm_config.INVERSE_SCALE, forward_scale=icm_config.FORWARD_SCALE,
              feature_extractor=feature_extractor, forward_model=forward_model, inverse_model=inverse_model,
              eta=icm_config.ETA).to("cuda:0" if torch.cuda.is_available() else "cpu")

    policy_kwargs = ppo_config.POLICY_KWARGS

    if use_ppo:
        model = intrinsic_PPO(policy="CnnPolicy", env=env, motivation_model=icm, motivation_lr=icm_config.LR,
                          motivation_grad_norm=icm_config.GRAD_NORM,
                          intrinsic_reward_coef=config.intrinsic_reward_coef, extrinsic_reward_coef=icm_config.EXTRINSIC_REWARD_COEF,
                          warmup_steps=icm_config.WARMUP, global_counter=global_counter, # wandb.config.a2c_real_lr
                          n_steps=ppo_config.NUM_STEPS, icm_n_steps=ppo_config.ICM_NUM_STEPS, n_epochs=ppo_config.EPOCHS,
                          batch_size=ppo_config.BATCH_SIZE, gae_lambda=ppo_config.GAE_LAMBDA,
                          ent_coef=ppo_config.ENTROPY_COEF, 
                          verbose=1, policy_kwargs=policy_kwargs,
                          device=environment_config.MODEL_DEVICE,
                          motivation_device=environment_config.MOTIVATION_DEVICE, seed=config.seed) # wandb.config.seed
    else:
        model = intrinsic_A2C(policy="CnnPolicy", env=env, motivation_model=icm, motivation_lr=icm_config.LR,
                              motivation_grad_norm=icm_config.GRAD_NORM,
                              intrinsic_reward_coef=icm_config.INTRINSIC_REWARD_COEF,
                              extrinsic_reward_coef=icm_config.EXTRINSIC_REWARD_COEF,
                              warmup_steps=icm_config.WARMUP, global_counter=global_counter,
                              learning_rate=a2c_config.LR,  # wandb.config.a2c_real_lr
                              n_steps=a2c_config.NUM_STEPS, gamma=a2c_config.GAMMA, gae_lambda=a2c_config.GAE_LAMBDA,
                              ent_coef=a2c_config.ENTROPY_COEF, vf_coef=a2c_config.VALUE_LOSS_COEF,
                              # wandb.config.entropy
                              max_grad_norm=a2c_config.MAX_GRAD_NORM,
                              use_rms_prop=a2c_config.RMS_PROP, verbose=1, policy_kwargs=policy_kwargs,
                              device=environment_config.MODEL_DEVICE,
                              motivation_device=environment_config.MOTIVATION_DEVICE, seed=config.seed)

    model.set_logger(
        AgentLogger(log_config.AGENT_LOG_FREQUENCY, None, "stdout", global_counter=global_counter))
    #1e7, 5e6
    model.learn(total_timesteps=float(2e7), callback=[LoggerCallback(0, "Doom report",
                                                                     global_counter=global_counter,
                                                                     quantity_of_agents=ppo_config.NUM_AGENTS,
                                                                     log_frequency=log_config.AGENT_LOG_FREQUENCY,
                                                                     video_submission_frequency=log_config.VIDEO_SUBMISSION_FREQUENCY,
                                                                     device=environment_config.MOTIVATION_DEVICE,
                                                                     fps=environment_config.FPS)])
    env.close()

if __name__=="__main__":
    # For ssh purposes
    os.environ["MUJOCO_GL"]="egl"
    #
    class Mock():
        def __init__(self):
            self.seed = 100
            self.entropy = 0.
            self.intrinsic_reward_coef = 1.

    config = Mock()
    sweep_id = wandb.sweep(sweep=sweep_configuration, project='ViZDoom_ICM+PPO')
    #main(config)
    wandb.agent(sweep_id, function=main, count=24)
