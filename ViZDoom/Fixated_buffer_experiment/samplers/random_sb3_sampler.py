import numpy
import numpy as np

import wandb
from envpool_to_sb3.vec_adapter import VecAdapter
from pathlib import Path
from doom_samples.utils.wrapper import ObservationWrapper
from stable_baselines3.common.env_util import make_vec_env
from doom_samples.custom_VizDoomEnv import CustomVizDoomEnv
from gym.wrappers import FrameStack
from vizdoom.gym_wrapper.base_gym_env import VizdoomEnv

def prepare_folders(quantity, is_test = False):
    folder = "/home/dvasilev/doom_dataset/no_action_repeat/train"
    if is_test:
        folder = "/home/dvasilev/doom_dataset/no_action_repeat/test"
    for i in range(quantity):
        first_frame_folder = folder + "/" + str(i) + "/" + "start_frames"
        second_frame_folder = folder + "/" + str(i) + "/" + "end_frames"
        Path(first_frame_folder).mkdir(exist_ok=True, parents=True)
        Path(second_frame_folder).mkdir(exist_ok=True, parents=True)

def grayscale_obs(obs):
    gray_temporal_obs = []
    for i in range(4):
        gray_temporal_obs.append(0.2989 * obs[i*3:i*3+1, :, :] + 0.5870 * obs[i*3+1:i*3+2, :, :] + 0.1140 * obs[i*3+2:i*3+3, :, :])
    return numpy.concatenate(gray_temporal_obs).astype(np.float32)

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

def wrap_env(env):
    env = VizdoomEnv("/home/dvasilev/doom_icm/mario_icm/custom_my_way_home.cfg", frame_skip=1)
    env = ObservationWrapper(env)
    env = FrameStack(env, 4)
    return env

parallel_envs = 20
env = make_vec_env("VizdoomMyWayHome-v0", n_envs=parallel_envs, wrapper_class=wrap_env)
wandb.init("EnvPool test")
prepare_folders(parallel_envs)
action_array = []
obs = env.reset()
i = 0
video = []
while i < 50_000:
    video.append(obs[0, 0:1])
    actions = np.random.randint(0, 3, parallel_envs)
    new_obs, rewards, dones, info = env.step(actions)
    if not np.any(dones):
        save_observations(i, obs, new_obs)
        action_array.append(actions)
        i += 1
    else:
        video_array = np.array(video)
        wandb.log({"Video": wandb.Video(video_array, fps=120)})
        video = []
    obs = new_obs

action_array = np.array(action_array)
save_action_array(action_array)
