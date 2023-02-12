import torch.nn.functional as F
import torch
from torch import nn
from Config.ENV_CFG import DEVICE
from torch import nn


class Predictor(nn.Module):
    def __init__(self, action_dim, state_dim, hidden_layer_neurons):
        super(Predictor, self).__init__()
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.simple_state_predictor = torch.nn.Sequential(nn.Linear(state_dim + action_dim, hidden_layer_neurons),
                                                          nn.ReLU(),
                                                          nn.Linear(hidden_layer_neurons, state_dim))


    def forward(self, state, action):
        action = F.one_hot(action, num_classes = self.action_dim).squeeze()

        concat_info = torch.cat((state, action), dim = 1)
        predicted_state = self.simple_state_predictor(concat_info)
        return predicted_state


#TODO: add 256 to config
class SimpleinverseNet(nn.Module):
    def __init__(self, state_dim, action_classes, hidden_layer_neurons):
        super(SimpleinverseNet, self).__init__()
        self.simple_classifier = torch.nn.Sequential(nn.Linear(2*state_dim, hidden_layer_neurons),
                                                     nn.ReLU(),
                                                     nn.Linear(hidden_layer_neurons, action_classes))


    def forward(self, previous_state, next_state):
        cat_state = torch.cat([previous_state, next_state], 1)

        #TODO: Authors of the book added softmax here
        processed_state = self.simple_classifier(cat_state)
        processed_state = F.softmax(processed_state)
        return processed_state


class SimplefeatureNet(nn.Module):
    def __init__(self, temporal_channels, feature_map_qty):
        super(SimplefeatureNet, self).__init__()
        # TODO No normalization. Maybe, they are not needed because of temporal channels
        self.simple_encoder =\
        nn.Sequential(nn.Conv2d(in_channels = temporal_channels, 
                                out_channels = feature_map_qty, kernel_size = 3, stride = 2, 
                                padding = 1),
                      nn.ELU(),
                      nn.Conv2d(in_channels = feature_map_qty, out_channels = feature_map_qty, 
                                kernel_size = 3, stride = 2, padding = 1),
                      nn.ELU(),
                      nn.Conv2d(in_channels = feature_map_qty, out_channels = feature_map_qty, 
                                kernel_size = 3, stride = 2, padding = 1),
                      nn.ELU(),
                      nn.Conv2d(in_channels = feature_map_qty, out_channels = feature_map_qty, 
                                kernel_size = 3, stride = 2, padding = 1),
                      nn.ELU(),
                      nn.Flatten(start_dim=1)) # 32*3*3
        self.state_dim = 32*3*3



    def forward(self, x):
        x = F.normalize(x)
        y = self.simple_encoder(x) #size N, 288
        return y


class ICM(nn.Module):
    def __init__(self, action_dim, temporal_channels,
                hidden_layer_neurons=256,  eta=1/2, feature_map_qty=32):
        super(ICM, self).__init__()
        self.eta = eta
        self.feature = SimplefeatureNet(temporal_channels, feature_map_qty).train()

        self.action_dim = action_dim
        self.state_dim = self.feature.state_dim
        self.inverse_net = SimpleinverseNet(self.state_dim, self.action_dim, hidden_layer_neurons).train()
        self.forward_net = Predictor(self.action_dim, self.state_dim, hidden_layer_neurons).train()


    def forward(self, observation, action, next_observation):
        # It is neccesary to NOT learn encoder when predicting future states
        # Encoder only learns when it guesses action by pair s_t & s_{t+1}
        # print("Obervation shape: {}".format(observation.shape))
        state = self.feature(observation)
        # print("Latent state shape: {}".format(state.shape))
        next_state = self.feature(next_observation)
        action_logits = self.inverse_net(state, next_state)
        # TODO: change to detach, maybe torch.no_grad() isn't needed
        with torch.no_grad():
            const_state = self.feature(observation)
            const_next_state = self.feature(next_observation)
        predicted_state = self.forward_net(const_state, action.detach())
        return action_logits, predicted_state, const_next_state


    def intrinsic_reward(self, observation, action, next_observation):
        intrinsic_reward = 0
        if type(action) == int:
            action = torch.nn.functional.one_hot(torch.tensor(action), self.action_dim).\
            unsqueeze(dim = 0).to(DEVICE)
        with torch.no_grad():
            predicted_state =\
                self.forward_net(self.feature(observation), action)
            real_state = self.feature(next_observation)
            intrinsic_reward =\
                self.eta*((predicted_state-real_state)**2).sum(dim=1).cpu().detach().numpy()
        return intrinsic_reward
