import sys
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.transforms import Grayscale

if 'gymnasium' in sys.modules:
    import gymnasium as gym
elif 'gym' in sys.modules:
    import gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class ActorCritic(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256): # modified action_space->action_space_size
        super(ActorCritic, self).__init__(observation_space, features_dim)
        self.gray = Grayscale(1)
        num_inputs = observation_space.shape[0]
        self.conv1 = nn.Conv2d(num_inputs, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)

        self.linear = nn.Linear(32 * 3 * 3, features_dim) # Hardcoded image size: bad practice

    def forward(self, inputs):
        x = F.elu(self.conv1(inputs))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        x = x.reshape(-1, 32 * 3 * 3) # Hardcoded image size: bad practice
        return F.elu(self.linear(x))