import torch.nn.functional as F
import torch
from torch import nn
from torchvision.transforms import Grayscale

DEVICE = torch.device("cpu")


class Predictor(nn.Module):
    def __init__(self, action_dim, state_dim, hidden_layer_neurons):
        super(Predictor, self).__init__()
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.simple_state_predictor = torch.nn.Sequential(nn.Linear(state_dim + action_dim, hidden_layer_neurons),
                                                          nn.ReLU(),
                                                          nn.Linear(hidden_layer_neurons, state_dim))


    def forward(self, state, action):
        action = F.one_hot(action, num_classes=self.action_dim).squeeze()

        concat_info = torch.cat((state, action), dim=1)
        predicted_state = self.simple_state_predictor(concat_info)
        return predicted_state


class SimpleinverseNet(nn.Module):
    def __init__(self, state_dim, action_classes, hidden_layer_neurons):
        super(SimpleinverseNet, self).__init__()
        self.simple_classifier = torch.nn.Sequential(nn.Linear(2*state_dim, hidden_layer_neurons),
                                                     nn.ReLU(),
                                                     nn.Linear(hidden_layer_neurons, action_classes))


    def forward(self, previous_state, next_state):
        cat_state = torch.cat([previous_state, next_state], 1)

        action_representation = self.simple_classifier(cat_state)
        return action_representation


class SimplefeatureNet(nn.Module):
    def __init__(self, temporal_channels, feature_map_qty):
        super(SimplefeatureNet, self).__init__()
        self.simple_encoder =\
        nn.Sequential(nn.Conv2d(in_channels=temporal_channels,
                                out_channels=feature_map_qty, kernel_size=3, stride=2,
                                padding=1),
                      nn.ELU(),
                      nn.Conv2d(in_channels=feature_map_qty, out_channels=feature_map_qty,
                                kernel_size=3, stride=2, padding=1),
                      nn.ELU(),
                      nn.Conv2d(in_channels=feature_map_qty, out_channels=feature_map_qty,
                                kernel_size=3, stride=2, padding=1),
                      nn.ELU(),
                      nn.Conv2d(in_channels=feature_map_qty, out_channels=feature_map_qty,
                                kernel_size=3, stride=2, padding=1),
                      nn.ELU(),
                      nn.Flatten(start_dim=1)) # 32*3*3
        self.state_dim = 32*3*3


    def forward(self, x):
        # WARNING Normalize
        x = x/255.
        y = self.simple_encoder(x) #size N, 288
        return y


class ICM(nn.Module):
    def __init__(self, action_dim, temporal_channels, inv_scale, forward_scale,
                hidden_layer_neurons, eta, feature_map_qty, discrete=True,
                predict_delta=False, subtract_ema_reward=False, ema_gamma=0.9, freeze_grad=False):
        super(ICM, self).__init__()
        self.eta = eta
        self.inv_scale = inv_scale
        self.forward_scale = forward_scale
        self.feature = SimplefeatureNet(temporal_channels, feature_map_qty).train()
        self.discrete = discrete

        self.freeze_grad = freeze_grad
        self.predict_delta = predict_delta
        self.subtract_ema_reward = subtract_ema_reward
        if subtract_ema_reward:
            self.ema_gamma = ema_gamma
            self.EMA_reward = 0

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
        #with torch.no_grad():
        const_state = self.feature(observation)
        const_next_state = self.feature(next_observation)
        if self.freeze_grad:
            const_state, const_next_state = const_state.detach(), const_next_state.detach()
        predicted_state = self.forward_net(const_state, action.detach())

        if self.predict_delta:
            predicted_state += const_state.detach()

        return action_logits, predicted_state, const_next_state


    def get_losses(self, observation, action, next_observation):
        predicted_actions, predicted_states, next_states =\
            self(observation, action, next_observation)
        inverse_pred_err = self.calculate_inverse_loss(predicted_actions, action)
        # WARNING: Pathak had 1/2, authors of the book hand't!
        self.raw_forward_loss = ((next_states-predicted_states)**2).sum(dim = 1).mean()
        forward_pred_err =\
            self.forward_scale*self.raw_forward_loss
        self.raw_forward_loss = self.raw_forward_loss.detach().cpu()
        return forward_pred_err, inverse_pred_err

    def calculate_inverse_loss(self, predicted_actions, actions):
        if self.discrete:
            loss = nn.CrossEntropyLoss()
            actions = F.one_hot(actions.flatten(), num_classes=self.action_dim).detach()
        else:
            loss = nn.MSELoss()
        self.raw_inverse_loss = loss(predicted_actions, actions.argmax(dim=1)).mean()
        inverse_pred_err = self.inv_scale * self.raw_inverse_loss
        self.raw_inverse_loss = self.raw_inverse_loss.detach().cpu()
        return inverse_pred_err

    def get_icm_loss(self, observation, action, next_observation):
        forward_loss, inverse_loss = self.get_losses(observation, action, next_observation)
        self.forward_loss = forward_loss.detach().cpu()
        self.inverse_loss = inverse_loss.detach().cpu()
        return forward_loss + inverse_loss

    def intrinsic_reward(self, observation, action, next_observation):
        intrinsic_reward = 0
        if type(action) == int:
            action = torch.nn.functional.one_hot(torch.tensor(action), self.action_dim).\
            unsqueeze(dim=0).to(DEVICE)
        with torch.no_grad():
            predicted_state =\
                self.forward_net(self.feature(observation), action)
            real_state = self.feature(next_observation)
            intrinsic_reward =\
                self.eta*((predicted_state-real_state)**2).sum(dim=1).cpu().detach().numpy()

        if self.subtract_ema_reward:
            self.EMA_reward = self.ema_gamma*self.EMA_reward + (1-self.ema_gamma)*intrinsic_reward.mean()
            intrinsic_reward -= self.EMA_reward
        return intrinsic_reward


    def get_action_prediction_metric(self, observation, next_observation, actions):
        with torch.no_grad():
            latent_obs, latent_next_obs = self.feature(observation), self.feature(next_observation)
            actions_representation = self.inverse_net(latent_obs, latent_next_obs)
            if self.discrete:
                # Mean probability of right action
                probabilities = F.softmax(actions_representation)
                metric = probabilities[torch.arange(0, actions.shape[-1]), actions].mean().item()
            else:
                # MSE of predicted actions vs preformed actions
                print("20=={}".format(((actions_representation-actions)**2).sum(axis=1).shape))
                metric = ((actions_representation-actions)**2).sum(axis=1).mean()
            return metric
