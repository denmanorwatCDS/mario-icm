import torch
from torch import nn
from torch.nn import functional as F

import gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class ActorCritic(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256): # modified action_space->action_space_size
        super(ActorCritic, self).__init__(observation_space, features_dim)
        num_inputs = observation_space.shape[0]
        self.hx, self.cx = None, None
        self.conv1 = nn.Conv2d(num_inputs, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)

        self.linear = nn.Linear(32 * 3 * 3, features_dim) # Hardcoded image size: bad practice

    def forward(self, inputs):
        #print(inputs.shape)
        #inputs = torch.permute(inputs, dims = (0, 3, 1, 2))/255.0
        x = F.elu(self.conv1(inputs))
        #print(x.shape)
        x = F.elu(self.conv2(x))
        #print(x.shape)
        x = F.elu(self.conv3(x))
        #print(x.shape)
        x = F.elu(self.conv4(x))
        #print(x.shape)

        x = x.reshape(-1, 32 * 3 * 3) # Hardcoded image size: bad practice

        return F.elu(self.linear(x))