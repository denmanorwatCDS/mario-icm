import gym.spaces
import numpy as np
import torch
import random
import vizdoom
from loggers.logger_callback import LoggerCallback
from loggers.a2c_logger import A2CLogger
from loggers.global_counter import GlobalCounter
from stable_baselines_intrinsic.intrinsic_a2c_doom import intrinsic_A2C
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack
from icm_mine.icm import ICM
from stable_baselines3.common.monitor import Monitor

from mario_icm.config import a2c_config, environment_config, hyperparameters, icm_config, log_config

from mario_icm.ViZDoom.utils.wrapper import ObservationWrapper
from mario_icm.ViZDoom.ViZDoom_continuous_support.ViZDoomEnv import VizdoomEnv
import wandb

def prepare_env(seed, rank):
    def wrap_env():
        env = VizdoomEnv("/home/dvasilev/doom_icm/mario_icm/ViZDoom/custom_my_way_home.cfg", frame_skip=1)
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
        'entropy': {'values': [1e-03]}, # 1e-02, 1e-03
        'seed': {'values': [10, 100, 1000, 2000, 3000]},
        'a2c_real_lr': {"values": [0.0001]},
        'frame_stack': {"values": [1]}# 0.0001, 0.0007
     }
}

def main():
    wandb.init()
    torch.manual_seed(10) #wandb.config.seed
    random.seed(10) # wandb.config.seed
    np.random.seed(10) # wandb.config.seed
    parallel_envs = 20  # 20
    envpool_env_id = "VizdoomCustom-v1"
    global_counter = GlobalCounter()
    print(vizdoom.scenarios_path)

    # Eval and train environments
    env = SubprocVecEnv([prepare_env(10, i) for i in range(parallel_envs)]) # wandb.config.seed
    env = VecFrameStack(env, 1) # wandb.config.frame_stack

    action_space = env.action_space
    icm = ICM(action_space, 1, # wandb.config.frame_stack
              icm_config.INVERSE_SCALE, icm_config.FORWARD_SCALE,
              hidden_layer_neurons=icm_config.HIDDEN_LAYERS, eta=icm_config.ETA,
              feature_map_qty=icm_config.FMAP_QTY, discrete=False) \
        .to("cuda:0" if torch.cuda.is_available() else "cpu")

    policy_kwargs = a2c_config.POLICY_KWARGS

    model = intrinsic_A2C(policy="CnnPolicy", env=env, motivation_model=icm, motivation_lr=icm_config.LR,
                          motivation_grad_norm=icm_config.GRAD_NORM,
                          intrinsic_reward_coef=icm_config.INTRINSIC_REWARD_COEF,
                          warmup_steps=icm_config.WARMUP, global_counter=global_counter, learning_rate=0.0001, # wandb.config.a2c_real_lr
                          n_steps=a2c_config.NUM_STEPS, gamma=a2c_config.GAMMA, gae_lambda=a2c_config.GAE_LAMBDA,
                          ent_coef=0.001, vf_coef=a2c_config.VALUE_LOSS_COEF, # wandb.config.entropy
                          max_grad_norm=a2c_config.MAX_GRAD_NORM,
                          use_rms_prop=a2c_config.RMS_PROP, verbose=1, policy_kwargs=policy_kwargs,
                          device=environment_config.MODEL_DEVICE,
                          motivation_device=environment_config.MOTIVATION_DEVICE, seed=10) # wandb.config.seed

    model.set_logger(
        A2CLogger(log_config.LOSS_LOG_FREQUENCY / a2c_config.NUM_STEPS, None, "stdout", global_counter=global_counter))
    model.learn(total_timesteps=float(2e7), callback=[LoggerCallback(0, "Doom report", hyperparameters.HYPERPARAMS,
                                                                     global_counter=global_counter,
                                                                     quantity_of_agents=a2c_config.NUM_AGENTS,
                                                                     log_frequency=log_config.AGENT_LOG_FREQUENCY,
                                                                     video_submission_frequency=log_config.VIDEO_SUBMISSION_FREQUENCY,
                                                                     device=environment_config.MOTIVATION_DEVICE,
                                                                     fps=environment_config.FPS)])
    import gc
    gc.collect()
    torch.cuda.empty_cache()

if __name__=="__main__":
    sweep_id = wandb.sweep(sweep=sweep_configuration, project='Doom-sparse-sweep')
    main()
    #wandb.agent(sweep_id, function=main, count=12)
