import torch.nn.functional as F
import torch
from torch import nn
import numpy as np

DEVICE = torch.device("cpu")

class Boundify(torch.nn.Module):
    def __init__(self, bounds, bounder):
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

class ConditionalConv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, identity=False):
        super().__init__()
        if identity:
            self.module = Identity()
        else:
            self.module = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                          kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, input):
        return self.module(input)

def params_to_cov_matrix(stds):
    action_space = stds.shape[1]
    device = stds.get_device()
    cov_matricies = torch.eye(action_space).repeat(stds.shape[0], 1, 1).to(device)
    cov_matricies[:, :, 0][cov_matricies[:, :, 0] == 1] *= stds[:, 0]
    cov_matricies[:, :, 1][cov_matricies[:, :, 1] == 1] *= stds[:, 1]
    return cov_matricies

def prepare_multivariate_loss(regulizer_coef = 0.):
    def multivariate_loss(params, ground_truth):
        batch_size, action_size = ground_truth.shape
        means = params[:, :action_size]
        stds = params[:, action_size:]
        stds = torch.exp(stds)
        covariance = params_to_cov_matrix(stds)
        inv_covariance = torch.linalg.inv(covariance)
        determinant = torch.linalg.det(covariance)

        bias = -action_size/2*(torch.tensor(2*torch.pi).log())
        log_det = -1/2*(determinant).log()
        value = -1/2*((ground_truth-means).reshape(-1, 1, action_size))@inv_covariance@((ground_truth-means).reshape(-1, action_size, 1))
        value = value.flatten()
        regulizer = (stds**2).sum(axis=1).mean()**(1/2)*regulizer_coef*(-1)
        return -(bias+log_det+value+regulizer)
    return multivariate_loss

def mse_loss(input, target):
    return ((input-target)**2).sum(dim=1)


class Predictor(nn.Module):
    def __init__(self, action_space, state_dim, hidden_layer_neurons, discrete):
        super(Predictor, self).__init__()
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

        concat_info = torch.cat((state, action), dim=1)
        predicted_state = self.simple_state_predictor(concat_info)
        return predicted_state


class SimpleinverseNet(nn.Module):
    def __init__(self, state_dim, action_space, hidden_layer_neurons, discrete=True, pde=False, apply_bounder=True,
                 additional_fc_layers=0):
        super(SimpleinverseNet, self).__init__()
        if discrete:
            action_classes = action_space.n
        else:
            print(action_space)
            action_classes = action_space.shape[0]
            bounds = np.concatenate((action_space.low.reshape(-1, 1), action_space.high.reshape(-1, 1)), axis=1)

        self.simple_classifier = torch.nn.Sequential(nn.Linear(2*state_dim, hidden_layer_neurons),
                                                     nn.ReLU())
        for i in range(additional_fc_layers):
            self.simple_classifier.append(torch.nn.Linear(hidden_layer_neurons, hidden_layer_neurons//2))
            self.simple_classifier.append(torch.nn.ReLU())
            hidden_layer_neurons = hidden_layer_neurons//2

        if not pde:
            self.simple_classifier.append(nn.Linear(hidden_layer_neurons, action_classes))
        if pde:
            # Vector of means and deviations
            mean_and_std_predictor = nn.Linear(hidden_layer_neurons, 2*action_classes)
            mean_and_std_predictor.bias = torch.nn.Parameter(torch.tensor([0., 0., 0., 0.], requires_grad=True))
            self.simple_classifier.append(mean_and_std_predictor)
        if not discrete and apply_bounder:
            if pde:
                bounds = np.concatenate((np.tile(bounds[0], (2, 1)), np.tile(bounds[1], (2, 1))), axis=0)
                bounds[2:, 0] = 0
                bounds[2:, 1] = 10
            bounds = torch.from_numpy(bounds)
            self.simple_classifier.append(Boundify(bounds))
        

        self.discrete = discrete


    def forward(self, previous_state, next_state):
        cat_state = torch.cat([previous_state, next_state], 1)

        action_representation = self.simple_classifier(cat_state)
        return action_representation


class SimplefeatureNet(nn.Module):
    def __init__(self, obs_shape, feature_map_qty, additional_layers=0):
        super(SimplefeatureNet, self).__init__()
        assert additional_layers in [0, 1, 2, 3, 4], "Expected number of additional layers in range from 0 to 4"
        self.simple_encoder =\
        nn.Sequential(ConditionalConv2d(in_channels=obs_shape[0],
                                out_channels=feature_map_qty, kernel_size=3, stride=1,
                                padding=1, identity=additional_layers <= 0),
                      nn.Conv2d(in_channels=feature_map_qty if additional_layers>0 else obs_shape[0],
                                out_channels=feature_map_qty, kernel_size=3, stride=2,
                                padding=1),
                      nn.ELU(),
                      ConditionalConv2d(in_channels=feature_map_qty,
                                        out_channels=feature_map_qty, kernel_size=3, stride=1,
                                        padding=1, identity=additional_layers <= 1),
                      nn.Conv2d(in_channels=feature_map_qty, out_channels=feature_map_qty,
                                kernel_size=3, stride=2, padding=1),
                      nn.ELU(),
                      ConditionalConv2d(in_channels=feature_map_qty,
                                        out_channels=feature_map_qty, kernel_size=3, stride=1,
                                        padding=1, identity=additional_layers <= 2),
                      nn.Conv2d(in_channels=feature_map_qty, out_channels=feature_map_qty,
                                kernel_size=3, stride=2, padding=1),
                      nn.ELU(),
                      ConditionalConv2d(in_channels=feature_map_qty,
                                        out_channels=feature_map_qty, kernel_size=3, stride=1,
                                        padding=1, identity=additional_layers <= 3),
                      nn.Conv2d(in_channels=feature_map_qty, out_channels=feature_map_qty,
                                kernel_size=3, stride=2, padding=1),
                      nn.ELU(),
                      nn.Flatten(start_dim=1)) # 32*3*3
        with torch.no_grad():
            output = self.simple_encoder(torch.zeros(obs_shape))
            self.state_dim = int(torch.tensor(output.shape).prod())


    def forward(self, x):
        # WARNING Normalize
        x = x/255.
        y = self.simple_encoder(x) #size N, 288
        return y


class ICM(nn.Module):
    def __init__(self, action_space, obs_shape, inv_scale, forward_scale,
                hidden_layer_neurons, eta, feature_map_qty, discrete=True,
                predict_delta=False, subtract_ema_reward=False, apply_bounder=True,
                pde=False, pde_regulizer=0., ema_gamma=0.9, freeze_grad=False,
                additional_conv_layer=0, additional_fc_layers=0):
        super(ICM, self).__init__()
        self.eta = eta
        self.inv_scale = inv_scale
        self.forward_scale = forward_scale
        self.feature = SimplefeatureNet(obs_shape, feature_map_qty, additional_layers=additional_conv_layer).train()
        self.pde_regulizer = pde_regulizer
        self.discrete = discrete
        self.pde = pde

        self.freeze_grad = freeze_grad
        self.predict_delta = predict_delta
        self.subtract_ema_reward = subtract_ema_reward
        if subtract_ema_reward:
            self.ema_gamma = ema_gamma
            self.EMA_reward = 0

        self.state_dim = self.feature.state_dim
        self.inverse_net = SimpleinverseNet(self.state_dim, action_space, hidden_layer_neurons, discrete, 
                                            apply_bounder=apply_bounder, pde=pde,
                                            additional_fc_layers=additional_fc_layers).train()
        self.forward_net = Predictor(action_space, self.state_dim, hidden_layer_neurons, discrete).train()

        if discrete:
            self.action_dim = action_space.n
        else:
            self.action_dim = action_space.shape[0]


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
        inverse_pred_err = self.calculate_inverse_loss(predicted_actions, action).mean()
        # WARNING: Pathak had 1/2, authors of the book hand't!
        self.raw_forward_loss = ((next_states-predicted_states)**2).sum(dim = 1).mean()
        forward_pred_err =\
            self.forward_scale*self.raw_forward_loss
        self.raw_forward_loss = self.raw_forward_loss.detach().cpu()
        return forward_pred_err, inverse_pred_err

    def calculate_inverse_loss(self, predicted, actions):
        if self.discrete:
            loss = nn.CrossEntropyLoss()
            actions = F.one_hot(actions.flatten(), num_classes=self.action_dim).detach()
            actions = actions.argmax(dim=1)
        elif self.pde:
            loss = prepare_multivariate_loss(regulizer_coef=self.pde_regulizer)
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


    def get_action_prediction_metric(self, observation, next_observation, actions, prefix = ""):
        with torch.no_grad():
            latent_obs, latent_next_obs = self.feature(observation), self.feature(next_observation)
            output = self.inverse_net(latent_obs, latent_next_obs)
            metric_names = []
            metric_values = []
            prefix = prefix + "/" if prefix != "" else prefix
            if self.discrete:
                # Mean probability of right action
                probabilities = F.softmax(output)
                metric = probabilities[torch.arange(0, actions.shape[-1]), actions].mean().item()
                metric_names.append(prefix+"Probability of right action")
                metric_values.append(metric)
            else:
                # MSE of predicted actions vs preformed actions
                if self.pde:
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

    def _debug_params(self):
        for name, param in self.named_parameters():
            print("Name: \n{}".format(name))
            print("Params: \n{}".format(param))
            print("Gradient: \n{}".format(param.grad))
