import numpy
import numpy as np


from config.compressed_config import environment_config
import envpool
from envpool_to_sb3.vec_adapter import VecAdapter
from pathlib import Path
import wandb

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
        np.save(first_frame_folder + "/" + str(current_iter), grayscale_obs(obs[i]))
        np.save(second_frame_folder + "/" + str(current_iter), grayscale_obs(new_obs[i]))

def save_action_array(action_array, is_test = False):
    folder = "/home/dvasilev/doom_dataset/no_action_repeat/train"
    if is_test:
        folder = "/home/dvasilev/doom_dataset/no_action_repeat/test"
    for i in range(action_array.shape[1]):
        np.save(folder + "/" + str(i) + "/" + "actions", np.array(action_array[:, i]))

def discrete_to_continuous_action(action):
    multipliers = np.array([5, 10, 5])
    quantity_of_agents = action.shape[0]
    action_OHE = np.zeros((quantity_of_agents, 3))
    action_OHE[np.arange(quantity_of_agents), action] = 1
    return action_OHE * multipliers


envpool_env_id = "VizdoomCustom-v1"
parallel_envs = 20
env = envpool.make(envpool_env_id, env_type="gym", num_envs=parallel_envs, seed=environment_config.SEED,
                   img_height=environment_config.RESIZED_SIZE[0], img_width=environment_config.RESIZED_SIZE[1],
                   stack_num=4, frame_skip=1,
                   cfg_path="/mario_icm/ViZDoom/custom_my_way_home.cfg",
                   wad_path="/mario_icm/ViZDoom/maps/my_way_home_dense.wad",
                   reward_config={"ARMOR": [0., 0.]})
wandb.init("EnvPool test")
prepare_folders(parallel_envs)
env.spec.id = envpool_env_id
env = VecAdapter(env)
action_array = []
obs = env.reset()
i = 0
video = []
while i < 50_000:
    vid_obs = grayscale_obs(obs)[0, 0:1, :, :]
    print(vid_obs.shape)
    video.append(vid_obs)
    actions = np.random.randint(0, 3, parallel_envs)
    actions = discrete_to_continuous_action(actions)
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