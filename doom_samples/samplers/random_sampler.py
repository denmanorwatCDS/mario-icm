from doom_samples.custom_VizDoomEnv import CustomVizDoomEnv
import numpy as np
from gym.wrappers import FrameStack
import wandb
from doom_samples.utils.wrapper import ObservationWrapper
from pathlib import Path
import vizdoom.gym_wrapper  # noqa

def gather_data(iterations, env, name, is_test = False):
    folder = "/home/dvasilev/doom_dataset/no_action_repeat/train"+"/"+name
    if is_test:
        folder = "/home/dvasilev/doom_dataset/no_action_repeat/test"+"/"+name
    first_frame_folder = folder + "/" + "start_frames"
    second_frame_folder = folder + "/" + "end_frames"
    Path(first_frame_folder).mkdir(exist_ok=True, parents=True)
    Path(second_frame_folder).mkdir(exist_ok=True, parents=True)
    video, actions = [], []
    done = True
    for i in range(iterations):
        if done:
            if len(video) > 0:
                video_array = np.array(video)
                wandb.log({"Video": wandb.Video(video_array, fps=120)})
                video = []
            obs, _ = env.reset()
            obs = np.asarray(obs)
            video.append(obs[0:1])
        action = int(np.random.randint(0, 3, 1))
        actions.append(action)
        new_obs, reward, terminated, truncated, info = env.step(action)
        video.append(new_obs[0:1])
        np.save(first_frame_folder + "/" + str(i), obs)
        np.save(second_frame_folder + "/" + str(i), new_obs)
        obs = new_obs
        done = terminated or truncated
    np.save(folder+"/"+"actions", np.array(actions))

if __name__ == "__main__":
    wandb.init("vizdoom_dataset_sanity_check")
    env = CustomVizDoomEnv("VizdoomMyWayHome-v0")
    env = ObservationWrapper(env)
    env = FrameStack(env, 4, lz4_compress=False)
    #for i in range(20):
    #    gather_data(50_000, env, name=str(i), is_test=False)
    gather_data(10_000, env, name="0", is_test=True)
