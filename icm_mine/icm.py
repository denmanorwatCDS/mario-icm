import torch.nn.functional as F
import torch
from mario_icm.icm_mine.architecture_constructors import FeatureNetwork, InverseNetwork
from torch import nn
import numpy as np

DEVICE = torch.device("cpu")

class BoundsTranslator(torch.nn.Module):
    def __init__(self, bounds):
        super().__init__()
        self.mean = (bounds[:, 1] + bounds[:, 0])/2
        self.range = (bounds[:, 1] - bounds[:, 0])/2
        self.register_buffer("Mean", self.mean, persistent=True)
        self.register_buffer("Range", self.range, persistent=True)
        self.bounder = torch.nn.Tanh()

    def forward(self, input):
        return self.bounder(input)*self.get_buffer("Range")+self.get_buffer("Mean")

class Identity(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input

class Conv2dOrIdentity(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, identity=False):
        super().__init__()
        if identity:
            self.module = Identity()
        else:
            self.module = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                          kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, input):
        return self.module(input)

def get_covariance_matrix(stds):
    action_space = stds.shape[1]
    device = stds.get_device()
    cov_matricies = torch.eye(action_space).repeat(stds.shape[0], 1, 1).to(device)
    cov_matricies[:, :, 0][cov_matricies[:, :, 0] == 1] *= stds[:, 0]
    cov_matricies[:, :, 1][cov_matricies[:, :, 1] == 1] *= stds[:, 1]
    return cov_matricies

def prepare_gauss_density_loss(regulizer_coef = 0.):
    def gauss_density_loss(params, ground_truth):
        batch_size, action_size = ground_truth.shape
        means = params[:, :action_size]
        stds = params[:, action_size:]
        stds = torch.exp(stds)
        covariance = get_covariance_matrix(stds)
        inv_covariance = torch.linalg.inv(covariance)
        determinant = torch.linalg.det(covariance)

        bias = -action_size/2*(torch.tensor(2*torch.pi).log())
        log_det = -1/2*(determinant).log()
        value = -1/2*((ground_truth-means).reshape(-1, 1, action_size))@inv_covariance@((ground_truth-means).reshape(-1, action_size, 1))
        value = value.flatten()
        regulizer = (stds**2).sum(axis=1).mean()**(1/2)*regulizer_coef*(-1)
        return -(bias+log_det+value+regulizer)
    return gauss_density_loss

def mse_loss(input, target):
    return ((input-target)**2).sum(dim=1)


class ForwardModel(nn.Module):
    def __init__(self, state_dim, action_space, hidden_layer_neurons, discrete):
        super(ForwardModel, self).__init__()
        self.discrete = discrete
        if discrete:
            self.action_dim = action_space.n
        else:
            self.action_dim = action_space.shape[0]
        self.state_dim = state_dim
        self.simple_state_predictor = torch.nn.Sequential(nn.Linear(state_dim + self.action_dim, hidden_layer_neurons),
                                                          nn.ReLU(),
                                                          nn.Linear(hidden_layer_neurons, state_dim))

    def forward(self, state, action):
        if self.discrete:
            action = F.one_hot(action, num_classes=self.action_dim).squeeze()
        batch_size = state.shape[0]
        state = state.reshape(batch_size, -1)
        concat_info = torch.cat((state, action), dim=1)
        predicted_state = self.simple_state_predictor(concat_info)
        return predicted_state


class InverseModel(nn.Module):
    def __init__(self, input_shape, action_space, discrete=True, pde=False, apply_bounder=True,
                 group=False, bottleneck=False, fc_qty=1):
        super(InverseModel, self).__init__()

        self.pde = pde
        self.discrete = discrete

        if discrete:
            action_classes = action_space.n
        else:
            action_classes = action_space.shape[0]
            bounds = np.concatenate((action_space.low.reshape(-1, 1), action_space.high.reshape(-1, 1)), axis=1)

        output_features = action_classes
        if pde:
            output_features = action_classes*2
        self.simple_classifier = InverseNetwork(group=group, bottleneck=bottleneck, input_shape=input_shape,
                                              output_features=output_features, fc_qty=fc_qty)
        if not discrete and apply_bounder:
            if pde:
                bounds = np.concatenate((np.tile(bounds[0], (2, 1)), np.tile(bounds[1], (2, 1))), axis=0)
            bounds = torch.from_numpy(bounds)
            self.simple_classifier.append(BoundsTranslator(bounds))


    def forward(self, previous_state, next_state):
        cat_state = torch.cat([previous_state, next_state], 1)

        action_representation = self.simple_classifier(cat_state)
        return action_representation


class FeatureExtractor(nn.Module):
    def __init__(self, obs_shape, batch_norm=False, skip_conn=False, consecutive_convs=1, activation=nn.ELU,
                 total_blocks=4, feature_map_size=32):
        super(FeatureExtractor, self).__init__()
        self.simple_encoder = FeatureNetwork(obs_shape, batch_norm=batch_norm, skip_conn=skip_conn,
                                             consecutive_convs=consecutive_convs, activation=activation,
                                             total_blocks=total_blocks, feature_map_size=feature_map_size)

    def forward(self, x):
        # WARNING Normalize
        x = x/255.
        y = self.simple_encoder(x) #size N, 288
        return y

    def get_latent(self, x):
        y = self.forward(x)
        batch_size = y.shape[0]
        return y.reshape(batch_size, -1)


class ICM(nn.Module):
    def __init__(self, inv_scale, forward_scale,
                 feature_extractor, inverse_model, forward_model,
                 eta,
                 predict_delta=False, subtract_ema_reward=False,
                 std_regulizer=0., ema_gamma=0.9, freeze_grad=False,):
        super(ICM, self).__init__()

        self.eta = eta
        self.inv_scale = inv_scale
        self.forward_scale = forward_scale
        self.std_regulizer = std_regulizer

        self.freeze_grad = freeze_grad
        self.predict_delta = predict_delta
        self.subtract_ema_reward = subtract_ema_reward
        if subtract_ema_reward:
            self.ema_gamma = ema_gamma
            self.EMA_reward = 0

        self.inverse_net = inverse_model
        self.forward_net = forward_model
        self.feature_extractor = feature_extractor
        self.action_dim = forward_model.action_dim

    def forward(self, observation, action, next_observation):
        # It is neccesary to NOT learn encoder when predicting future states 
        # Encoder only learns when it guesses action by pair s_t & s_{t+1}
        state = self.feature_extractor(observation)
        next_state = self.feature_extractor(next_observation)
        action_logits = self.inverse_net(state, next_state)
        # TODO: change to detach, maybe torch.no_grad() isn't needed
        # with torch.no_grad():
        const_state = self.feature_extractor.get_latent(observation)
        const_next_state = self.feature_extractor.get_latent(next_observation)
        if self.freeze_grad:
            const_state, const_next_state = const_state.detach(), const_next_state.detach()
        predicted_state = self.forward_net(const_state, action.detach())

        if self.predict_delta:
            predicted_state += const_state.detach()

        return action_logits, predicted_state, const_next_state

    def get_losses(self, observation, action, next_observation):
        predicted_actions, predicted_states, next_states =\
            self(observation, action, next_observation)
        inverse_pred_err = self.calculate_inverse_loss(predicted_actions, action).mean()
        self.raw_forward_loss = ((next_states-predicted_states)**2).sum(dim = 1).mean()
        forward_pred_err =\
            self.forward_scale*self.raw_forward_loss
        self.raw_forward_loss = self.raw_forward_loss.detach().cpu()
        return forward_pred_err, inverse_pred_err

    def calculate_inverse_loss(self, predicted, actions):
        if self.inverse_net.discrete:
            loss = nn.CrossEntropyLoss()
            actions = F.one_hot(actions.flatten(), num_classes=self.action_dim).detach()
            actions = actions.argmax(dim=1)
        elif self.inverse_net.pde:
            loss = prepare_gauss_density_loss(regulizer_coef=self.std_regulizer)
        else:
            loss = mse_loss
        self.raw_inverse_loss = loss(predicted, actions)
        inverse_pred_loss = self.inv_scale * self.raw_inverse_loss
        self.raw_inverse_loss = self.raw_inverse_loss.detach().cpu().mean()
        return inverse_pred_loss

    def get_icm_loss(self, observation, action, next_observation):
        forward_loss, inverse_loss = self.get_losses(observation, action, next_observation)
        self.forward_loss = forward_loss.detach().cpu()
        self.inverse_loss = inverse_loss.detach().cpu()
        return forward_loss + inverse_loss

    def intrinsic_reward(self, observation, action, next_observation):
        if self.inverse_net.discrete:
            action = torch.nn.functional.one_hot(torch.tensor(action), self.action_dim).\
            unsqueeze(dim=0).to(DEVICE)
        with torch.no_grad():
            predicted_state =\
                self.forward_net(self.feature_extractor(observation), action)
            real_state = torch.flatten(self.feature_extractor(next_observation), start_dim=1)
            intrinsic_reward =\
                self.eta*((predicted_state-real_state)**2).sum(dim=1).cpu().detach().numpy()

        if self.subtract_ema_reward:
            self.EMA_reward = self.ema_gamma*self.EMA_reward + (1-self.ema_gamma)*intrinsic_reward.mean()
            intrinsic_reward -= self.EMA_reward
        return intrinsic_reward

    def get_action_prediction_metric(self, observation, next_observation, actions, prefix = ""):
        with torch.no_grad():
            latent_obs, latent_next_obs = self.feature_extractor(observation), self.feature_extractor(next_observation)
            output = self.inverse_net(latent_obs, latent_next_obs)
            metric_names = []
            metric_values = []
            prefix = prefix + "/" if prefix != "" else prefix
            if self.inverse_net.discrete:
                # Mean probability of right action
                probabilities = F.softmax(output)
                metric = probabilities[torch.arange(0, actions.shape[-1]), actions].mean().item()
                metric_names.append(prefix+"Probability of right action")
                metric_values.append(metric)
            else:
                # MSE of predicted actions vs preformed actions
                if self.inverse_net.pde:
                    actions_representation = output[:, :self.action_dim].detach().cpu().numpy()
                    stds = output[:, self.action_dim:].detach().cpu().numpy().mean()
                    metric_names.append(prefix+"Mean mean of predicted gaussian")
                    metric_names.append(prefix+"Mean std of predicted gaussian")
                    metric_names.append(prefix+"Distance to right action")
                    metric_values.append(actions_representation.mean())
                    metric_values.append(stds)
                else:
                    actions_representation = output.detach().cpu().numpy()
                    metric_names.append(prefix+"Distance to right action")
                metric = ((((actions_representation-actions)**2).sum(axis=1))**(1/2)).mean()
                metric_values.append(metric)
            return dict(zip(metric_names, metric_values))

    def get_grad_norm(self):
        total_norm = 0
        for p in self.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        return float(total_norm)
