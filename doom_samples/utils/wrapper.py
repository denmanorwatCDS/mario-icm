import gym
import numpy as np
import cv2

class ObservationWrapper(gym.Wrapper):
    """
    ViZDoom environments return dictionaries as observations, containing
    the main image as well other info.
    The image is also too large for normal training.
    This wrapper replaces the dictionary observation space with a simple
    Box space (i.e., only the RGB image), and also resizes the image to a
    smaller size.
    NOTE: Ideally, you should set the image size to smaller in the scenario files
          for faster running of ViZDoom. This can really impact performance,
          and this code is pretty slow because of this!
    """

    def __init__(self, env, shape=(42, 42)):
        super().__init__(env)

        # Create new observation space with the new shape
        new_shape = (shape[0], shape[1], 1)
        self.observation_space = gym.spaces.Box(0, 255, shape=new_shape, dtype=np.uint8)

    def process_obs(self, obs):
        obs = obs["rgb"]
        obs = cv2.resize(obs, (42, 42))
        obs = 0.2989 * obs[:, :, 0:1] + 0.5870 * obs[:, :, 1:2] + 0.1140 * obs[:, :, 2:]
        obs = obs.astype(np.uint8).squeeze()
        return obs
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = self.process_obs(obs)
        return obs, reward, done, False, info

    def reset(self):
        obs = super().reset()
        return self.process_obs(obs), None