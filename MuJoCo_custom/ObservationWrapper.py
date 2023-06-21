import gymnasium as gym
import cv2
import numpy as np


class ObservationWrapper(gym.ObservationWrapper):
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

    def __init__(self, env):
        super().__init__(env)

        # Create new observation space with the new shape
        self.observation_space = env.observation_space["observation"]
        self.observation_space.dtype = np.uint8

    def observation(self, observation):
        obs = observation["observation"]

        return obs