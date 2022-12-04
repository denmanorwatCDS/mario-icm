import torch.nn.functional as F
import torch
from torch import nn
from Config.environment_config import DEVICE

class Predictor(nn.Module):
    def __init__(self, action_dim, state_dim):
        super(Predictor, self).__init__()
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.simple_state_predictor = torch.nn.Sequential(nn.Linear(state_dim + action_dim, 2*(state_dim + action_dim)),
                                                          nn.ReLU(),
                                                          nn.Linear(2*(state_dim + action_dim), 2*(state_dim + action_dim)),
                                                          nn.ReLU(),
                                                          nn.Linear(2*(state_dim + action_dim), state_dim))

    def forward(self, state, action):
        #print("Quantity of classes: {}".format(self.action_dim))
        #print("Action shape before: {}".format(action.shape))
        action = F.one_hot(action, num_classes = self.action_dim)
        #print("State shape: {}".format(state.shape))
        #print("Action shape after: {}".format(action.shape))
        concat_info = torch.cat((state, action), dim = 1)
        predicted_state = self.simple_state_predictor(concat_info)
        return predicted_state

class SimpleinverseNet(nn.Module):
    def __init__(self, state_dim, action_classes):
        super(SimpleinverseNet, self).__init__()
        self.simple_classifier = torch.nn.Sequential(nn.Linear(2*state_dim, 4*state_dim),
                                                     nn.ReLU(),
                                                     nn.Linear(4*state_dim, 4*state_dim),
                                                     nn.ReLU(),
                                                     nn.Linear(4*state_dim, action_classes))

    def __call__(self, previous_state, next_state):
        cat_state = torch.cat([previous_state, next_state], 1)
        return self.simple_classifier(cat_state)

class SimplefeatureNet(nn.Module):
    def __init__(self, state_dim, temporal_channels):
        super(SimplefeatureNet, self).__init__()
        self.simple_encoder =\
        nn.Sequential(nn.Conv2d(in_channels = temporal_channels, 
                                out_channels = 32, kernel_size = 3, stride = 2, 
                                padding = 1),
                      nn.ELU(),
                      nn.Conv2d(in_channels = 32, out_channels = 32, 
                                kernel_size = 3, stride = 2, padding = 1),
                      nn.ELU(),
                      nn.Conv2d(in_channels = 32, out_channels = 32, 
                                kernel_size = 3, stride = 2, padding = 1),
                      nn.ELU(),
                      nn.Conv2d(in_channels = 32, out_channels = 32, 
                                kernel_size = 3, stride = 2, padding = 1),
                      nn.ELU(),
                      nn.Flatten())
        self.state_dim = state_dim
        self.LSTM = nn.LSTM(32 * 3 * 3, state_dim, batch_first = True) # Batch first was here. 32*3*3

    def init_LSTM(self):
        self.h_n = torch.zeros(1, self.state_dim).to(DEVICE)
        self.c_n = torch.zeros(1, self.state_dim).to(DEVICE)
    
    def __call__(self, observation):
        embeddings = self.simple_encoder(observation)
        output, _ = self.LSTM(embeddings, (self.h_n, self.c_n))
        self.h_n, self.c_n = _

        return output

class ICM(nn.Module):
    def __init__(self, action_dim, state_dim, temporal_channels, eta = 1/2):
        super(ICM, self).__init__()
        self.eta = eta
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.feature = SimplefeatureNet(state_dim, temporal_channels).to(DEVICE)
        self.inverse_net = SimpleinverseNet(state_dim, action_dim).to(DEVICE)
        self.feature.init_LSTM()
        self.forward_net = Predictor(action_dim, state_dim)


    def forward(self, observation, action, next_observation):
        # It is neccesary to NOT learn encoder when predicting future states
        # Encoder only learns when it guesses action by pair s_t & s_{t+1}
        state = self.feature(observation)
        next_state = self.feature(next_observation) 
        action_logits = self.inverse_net(state, next_state)
        action_probabilities = torch.softmax(action_logits, 1)
        with torch.no_grad():
            const_state = self.feature(observation)
            const_next_state = self.feature(next_observation)
        predicted_state = self.forward_net(const_state, action)
        return action_probabilities, predicted_state, const_next_state


    def intrinsic_reward(self, observation, action, next_observation):
        intrinsic_reward = 0
        if type(action) == int:
            action = torch.nn.functional.one_hot(torch.tensor(action), 6).\
            unsqueeze(dim = 0).to(DEVICE)
        with torch.no_grad():
            predicted_state =\
                self.forward_net(self.feature(observation), action)
            real_state = self.feature(next_observation)
            intrinsic_reward =\
                self.eta*((predicted_state-real_state)**2).sum().cpu().detach().numpy()
        return intrinsic_reward
