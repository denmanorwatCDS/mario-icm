import sys
if 'gymnasium' in sys.modules:
    import gymnasium
    if gymnasium.__version__=='0.28.1':
        import gymnasium as gym

import numpy as np
import torch
import random
import vizdoom
from loggers.logger_callback import LoggerCallback
from loggers.a2c_logger import A2CLogger
from loggers.global_counter import GlobalCounter
from stable_baselines3.ppo.ppo import PPO
from stable_baselines_intrinsic.intrinsic_ppo_doom import intrinsic_PPO
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack
from icm_mine.icm import ICM
from stable_baselines3.common.monitor import Monitor

from mario_icm.config import a2c_config, environment_config, hyperparameters, icm_config, log_config

from mario_icm.ViZDoom.utils.wrapper import ObservationWrapper
if vizdoom.__version__=="1.1.13":
    from mario_icm.ViZDoom.ViZDoom_continuous_support.ViZDoomEnv_gym import VizdoomEnv
elif vizdoom.__version__ == "1.2.0":
    from mario_icm.ViZDoom.ViZDoom_continuous_support.ViZDoomEnv_gymnasium import VizdoomEnv
import wandb

def prepare_env(seed, rank, discrete = True):
    def wrap_env():
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
        'seed': {'values': [100, 200, 300]},
        'intrinsic_reward_coef': {"values": [1.]}
     }
}

def main(config = None):
    wandb.init()
    if config is None:
        config = wandb.config
    torch.manual_seed(config.seed) #wandb.config.seed
    random.seed(config.seed) # wandb.config.seed
    np.random.seed(config.seed) # wandb.config.seed
    parallel_envs = 20  # 20
    discrete = False
    envpool_env_id = "VizdoomCustom-v1"
    global_counter = GlobalCounter()
    print(vizdoom.scenarios_path)

    # Eval and train environments
    env = SubprocVecEnv([prepare_env(config.seed, i, discrete) for i in range(parallel_envs)]) # wandb.config.seed
    #env = VecFrameStack(env, 1) # wandb.config.frame_stack

    action_space = env.action_space
    obs_shape = (1, 42, 42)
    icm = ICM(action_space, obs_shape, inv_scale=icm_config.INVERSE_SCALE, forward_scale=icm_config.FORWARD_SCALE,
              hidden_layer_neurons=icm_config.HIDDEN_LAYERS,
              discrete=discrete, pde=False, freeze_grad=False,
              eta=0.02, apply_bounder=False, pde_regulizer=0.,
              inverse_bottleneck=False, inverse_group=False, inverse_fc_qty=2,
              feature_skip_conn=False, feature_consecutive_convs=1, feature_batch_norm=False, feature_total_blocks=4) \
        .to("cuda:0" if torch.cuda.is_available() else "cpu")
    """
    icm = ICM(action_space.n, 1, 0.8, 0.2, False, 256, 0.2, 32).to("cuda:0" if torch.cuda.is_available() else "cpu")
    """
    policy_kwargs = a2c_config.POLICY_KWARGS

    #n_steps = 2000, icm_n_steps=20
    model = PPO(policy="CnnPolicy", env=env)
    """
    model = intrinsic_PPO(policy="CnnPolicy", env=env, motivation_model=icm, motivation_lr=icm_config.LR,
                          motivation_grad_norm=icm_config.GRAD_NORM,
                          intrinsic_reward_coef=config.intrinsic_reward_coef, extrinsic_reward_coef=10.,
                          warmup_steps=icm_config.WARMUP, global_counter=global_counter, # wandb.config.a2c_real_lr
                          n_steps=2000, icm_n_steps=20, n_epochs=5, batch_size=200, gae_lambda=0.95,
                          ent_coef=config.entropy, 
                          verbose=1, policy_kwargs=policy_kwargs,
                          device=environment_config.MODEL_DEVICE,
                          motivation_device=environment_config.MOTIVATION_DEVICE, seed=config.seed) # wandb.config.seed
    """
    model.set_logger(
        A2CLogger(1, None, "stdout", global_counter=global_counter))
    model.learn(total_timesteps=float(1e7), callback=[LoggerCallback(0, "Doom report", hyperparameters.HYPERPARAMS,
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

    class Mock():
        def __init__(self):
            self.seed = 100
            self.entropy = 0.
            self.intrinsic_reward_coef = 0.

    config = Mock()
    sweep_id = wandb.sweep(sweep=sweep_configuration, project='ViZDoom_ICM+PPO')
    #main()
    wandb.agent(sweep_id, function=main, count=10)
