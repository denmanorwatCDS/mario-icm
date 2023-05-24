import numpy as np
import gym

import wandb
from pathlib import Path
from mario_icm.ViZDoom.utils.wrapper import ObservationWrapper
from mario_icm.ViZDoom.ViZDoom_continuous_support.ViZDoomEnv import VizdoomEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack

def prepare_folders(quantity, is_test = False):
    folder = "/home/dvasilev/doom_dataset/no_action_repeat/train"
    if is_test:
        folder = "/home/dvasilev/doom_dataset/no_action_repeat/test"
    for i in range(quantity):
        first_frame_folder = folder + "/" + str(i) + "/" + "start_frames"
        second_frame_folder = folder + "/" + str(i) + "/" + "end_frames"
        Path(first_frame_folder).mkdir(exist_ok=True, parents=True)
        Path(second_frame_folder).mkdir(exist_ok=True, parents=True)


def save_observations(current_iter, obs, new_obs, is_test = False):
    folder = "/home/dvasilev/doom_dataset/no_action_repeat/train"
    if is_test:
        folder = "/home/dvasilev/doom_dataset/no_action_repeat/test"
    for i in range(len(obs)):
        first_frame_folder = folder + "/" + str(i) + "/" + "start_frames"
        second_frame_folder = folder + "/" + str(i) + "/" + "end_frames"
        np.save(first_frame_folder + "/" + str(current_iter), obs[i])
        np.save(second_frame_folder + "/" + str(current_iter), new_obs[i])


def save_action_array(action_array, is_test = False):
    folder = "/home/dvasilev/doom_dataset/no_action_repeat/train"
    if is_test:
        folder = "/home/dvasilev/doom_dataset/no_action_repeat/test"
    for i in range(action_array.shape[1]):
        np.save(folder + "/" + str(i) + "/" + "actions", np.array(action_array[:, i]))


def prepare_env(seed, rank):
    def wrap_env():
        env = VizdoomEnv("/home/dvasilev/doom_icm/mario_icm/ViZDoom/custom_my_way_home.cfg", frame_skip=10)
        env.reset(seed=seed+rank)
        env = Monitor(env, filename=None)
        env = ObservationWrapper(env)
        return env
    set_random_seed(seed)
    return wrap_env


def prepare_sampler(env, parallel_envs):
    discrete = True
    low, high = 0, 3
    if type(env.action_space) == gym.spaces.Box:
        discrete = False
        low, high = env.action_space.low, env.action_space.high
    if discrete:
        def sampler():
            return np.random.randint(low, high, size = parallel_envs)
    else:
        def sampler():
            return np.random.uniform(low, high, size=(parallel_envs, 2)).astype(np.float32)
    return sampler

if __name__ == "__main__":
    parallel_envs = 20
    is_test = False
    env = SubprocVecEnv([prepare_env(10, i) for i in range(parallel_envs)])
    env = VecFrameStack(env, n_stack=4)
    wandb.init("EnvPool test")
    prepare_folders(parallel_envs, is_test=is_test)
    action_array = []
    obs = env.reset()
    obs = obs.transpose(0, 3, 1, 2)
    i = 0
    video = []

    action_sampler = prepare_sampler(env, parallel_envs)

    while i < 50_000:
        video.append(obs[0, 0:1, :, :])
        actions = action_sampler()

        new_obs, rewards, dones, info = env.step(actions)
        new_obs = new_obs.transpose(0, 3, 1, 2)
        if not np.any(dones):
            save_observations(i, obs, new_obs, is_test=is_test)
            low_bounds, high_bounds = env.observation_space.low, env.observation_space.high
            action_array.append(actions)
            i += 1
        else:
            video_array = np.array(video)
            wandb.log({"Video": wandb.Video(video_array, fps=120)})
            video = []
        obs = new_obs

    action_array = np.array(action_array)
    save_action_array(action_array, is_test=is_test)