import torchvision.transforms as T
import gym
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image
import wandb

from Config.ENV_CFG import DEVICE

class ResizeAndGrayscale(gym.ObservationWrapper):
    def __init__(self, env, new_image_size, max_temporal_channels):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(shape=new_image_size + (max_temporal_channels,), low=0, high=255, dtype=np.float32)
        self.new_image_size = new_image_size
        self.max_temporal_channels = max_temporal_channels
        self.observation_buffer = []
        self.is_empty = True


    def observation(self, observation):
        observation_compressed = T.ToPILImage()(observation)
        observation_compressed = T.Grayscale()(observation_compressed)
        observation_compressed = T.Resize(size = self.new_image_size)(observation_compressed)
        observation_compressed = np.array(observation_compressed)
        self.__update_buffer(observation_compressed)
        observation_in_time = self.__get_compressed_observation()

        return observation_in_time


    def __update_buffer(self, observation):
        if self.is_empty:
            self.observation_buffer =\
            [np.expand_dims(observation.copy(), axis=2) for i in range(self.max_temporal_channels)]
            self.is_empty = False
        del self.observation_buffer[0]
        self.observation_buffer.append(np.expand_dims(observation.copy(), axis=2))


    def __get_compressed_observation(self):
        return np.concatenate(self.observation_buffer, axis = 2)


    def reset_buffer(self):
        self.observation_buffer = []
        self.is_empty = True