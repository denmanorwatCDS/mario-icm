import gym
import numpy as np
import wandb

from stable_baselines3.a2c.a2c import A2C
from Callbacks.my_callback_ext import CustomCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from Logger.my_logger import A2CLogger

def make_env(env_id, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = gym.make(env_id)
        env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init

if __name__ == "__main__":
    num_cpu = 4  # Number of processes to use
    HYPERPARAMS = {"num_cpu": num_cpu}
    wandb.init(project="Deadlock_exploration")
    env_id = "CartPole-v1"
    # Create the vectorized environment
    env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])
    callback = CustomCallback(parallel_envs=num_cpu, action_space_size=env.action_space.n, project_name="Deadlock_exploration",
                              HYPERPARAMS=HYPERPARAMS)

    # Stable Baselines provides you with make_vec_env() helper
    # which does exactly the previous steps for you.
    # You can choose between `DummyVecEnv` (usually faster) and `SubprocVecEnv`
    # env = make_vec_env(env_id, n_envs=num_cpu, seed=0, vec_env_cls=SubprocVecEnv)

    log_path = 'sb3_logs'
    format = 'tensorboard'
    model = A2C("MlpPolicy", env, verbose=1)
    model.set_logger(A2CLogger(log_path, format, num_workers=num_cpu))
    model.learn(total_timesteps=25_000)

#    obs = env.reset()
#    for _ in range(1000):
#        action, _states = model.predict(obs)
#        obs, rewards, dones, info = env.step(action)
#        env.render()
