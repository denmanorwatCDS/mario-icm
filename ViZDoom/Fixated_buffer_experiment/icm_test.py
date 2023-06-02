import torch.optim

from mario_icm.ViZDoom.Fixated_buffer_experiment.custom_dataset.stable_buffer import PairedImageDataset, MultiAgentDataset
from torch.utils.data import DataLoader
from mario_icm.icm_mine.icm import ICM
from mario_icm.ViZDoom.ViZDoom_continuous_support.ViZDoomEnv_gym import VizdoomEnv
from sklearn.neighbors import KernelDensity
import numpy as np
import wandb
import matplotlib.pyplot as plt
import os

def visualise_ten_pictures(prev_obs, actions, next_obs):
    previous_frame = prev_obs[:10, 3:, :, :]
    next_frame = next_obs[:10, 3:, :, :]
    actions = actions[:10]
    titles = ["act1: {:.5f}".format(actions[i, 0]) + "; act2: {:.5f}".format(actions[i, 1]) for i in range(10)]
    fig = plt.figure(figsize=(50, 50))

    for i in range(10):
        sub = fig.add_subplot(5, 2, i + 1)
        divider = torch.zeros((1, 42, 1)).cuda()+255.
        image = torch.cat((previous_frame[i], divider, next_frame[i]), dim=2).squeeze()
        image = image.detach().cpu().numpy()
        sub.imshow(image, cmap="gray")
        sub.set_title(titles[i])
    fig.savefig("/home/dvasilev/doom_icm/mario_icm/fixated_buffer_image_logs/examples")
    plt.close(fig)
    plt.cla()

def one_batch_train(train_dataloader, ICM, optim):
    start_frames, end_frames, actions = next(iter(train_dataloader))
    start_frames, end_frames, actions = start_frames.to("cuda:0"), end_frames.to("cuda:0"), actions.to("cuda:0")
    for i in range(5_000):
        forward_loss, inverse_loss = ICM.get_losses(start_frames, actions, end_frames)
        icm_loss = forward_loss + inverse_loss
        optim.zero_grad()
        icm_loss.backward()
        optim.step()
        wandb.log({"Forward loss": forward_loss.cpu().detach(),
                   "Inverse loss": inverse_loss.cpu().detach(),
                   "ICM loss": icm_loss.cpu().detach()})


def slice_train(train_dataloader, ICM, optim, test_dataloader, config):
    counter = 0
    while counter < config.learning_steps:
        for start_frames, end_frames, actions in train_dataloader:
            start_frames, end_frames, actions = start_frames.to("cuda:0"), end_frames.to("cuda:0"), actions.to("cuda:0")
            forward_loss, inverse_loss = ICM.get_losses(start_frames, actions, end_frames)
            icm_loss = inverse_loss+forward_loss
            optim.zero_grad()
            icm_loss.backward()
            grad_norm = ICM.get_grad_norm()
            optim.step()
            metrics = ICM.get_action_prediction_metric(start_frames, end_frames, actions.cpu().numpy())

            logs = {"Forward loss": forward_loss.cpu().detach(),
               "Inverse loss": inverse_loss.cpu().detach(),
               "ICM loss": icm_loss.cpu().detach(),
               "Gradient norm": grad_norm}
            if counter%1000 == 0:
                visualise_ten_pictures(start_frames, actions, end_frames)
                predicted_actions, _, _ = ICM.forward(start_frames, actions, end_frames)
                if config.discrete == False:
                    plot_hist(actions, predicted_actions, np.array([[-1, 1], [-1, 1]]))
                loss = ICM.calculate_inverse_loss(predicted_actions, actions).detach().cpu()
                if config.discrete == False:
                    plot_loss(loss, actions)
                logs.update(get_image_metrics())
                test_model(ICM, test_dataloader)
            logs.update(metrics)
            wandb.log(logs)
            counter += 1

def test_model(model, test_dataloader):
    with torch.no_grad():
        for start_frames, end_frames, actions in test_dataloader:
            start_frames, end_frames, actions = start_frames.to("cuda:0"), end_frames.to("cuda:0"), actions.to("cuda:0")
            forward_loss, inverse_loss = model.get_losses(start_frames, actions, end_frames)
            icm_loss = inverse_loss + forward_loss
            metrics = model.get_action_prediction_metric(start_frames, end_frames, actions.cpu().numpy(), prefix="test")
            logs = {"test/Forward loss": forward_loss.cpu().detach(),
                "test/Inverse loss test": inverse_loss.cpu().detach(),
                "test/ICM loss test": icm_loss.cpu().detach()}
            logs.update(metrics)
            wandb.log(logs)


def plot_hist(gr_actions, pred_actions, bounds):
    if type(gr_actions) != np.ndarray:
        gr_actions = gr_actions.detach().cpu().numpy()
    if type(pred_actions) != np.ndarray:
        pred_actions = pred_actions.detach().cpu().numpy()
    for i in range(gr_actions.shape[-1]):
        gr_action_dim = gr_actions[:, i:i+1]
        pred_actions_dim = pred_actions[:, i:i+1]
        action_bounds = bounds[i]
        kde_gr = KernelDensity(bandwidth=0.05).fit(gr_action_dim)
        kde_pred = KernelDensity(bandwidth=0.05).fit(pred_actions_dim)
        data_points = np.linspace(action_bounds[0], action_bounds[1], num=(action_bounds[1]-action_bounds[0])*250).reshape(-1, 1)
        probabilities_gr = np.exp(kde_gr.score_samples(data_points))
        probabilities_pred = np.exp(kde_pred.score_samples(data_points))
        plt.plot(data_points, probabilities_gr, label="Ground truth")
        plt.plot(data_points, probabilities_pred, label="Predicted")
        plt.legend()
        plt.savefig("/home/dvasilev/doom_icm/mario_icm/fixated_buffer_image_logs/"
                    "Probabilities of actions; action index: {}".format(i))
        plt.cla()


def plot_loss(loss, actions):
    if type(actions) != np.ndarray:
        actions = actions.detach().cpu().numpy()
        loss = loss.detach().cpu().numpy()
    for i in range(actions.shape[1]):
        action_by_dim = actions[:, i]
        pack = [[action, loss] for action, loss in sorted(zip(action_by_dim, loss), key=lambda pair: pair[0])]
        pack = np.array(pack)
        plt.plot(pack[:, 0], pack[:, 1])
        plt.savefig("/home/dvasilev/doom_icm/mario_icm/fixated_buffer_image_logs/"
                f"Loss per action_{i}")
        plt.cla()

def get_image_metrics():
    folder = "/home/dvasilev/doom_icm/mario_icm/fixated_buffer_image_logs/"
    image_paths = os.listdir(folder)
    wandb_images = {}
    for image_path in image_paths:
        wandb_images[image_path] = wandb.Image(folder+"/"+image_path)
    return wandb_images


sweep_configuration = {
    "method": "grid",
    "parameters": {
        "hidden_layer_neurons": {"values": [128]},
        "inverse_fc_qty": {"values": [2]},
        "feature_skip_conn": {"values": [False]},
        "feature_consecutive_convs": {"values": [1]},
        "feature_batch_norm": {"values": [False]},
        "feature_total_blocks": {"values": [6]},
        "pde": {"values": [False]},
        "freeze_grad": {"values": [False]},
        "inverse_bottleneck": {"values": [True]},
        "inverse_group": {"values": [True]},
        "apply_bounder": {"values": [False]},
        "discrete": {"values": [False]},
        "learning_steps": {"values": [7_000]},
        "batch_size": {"values": [500]},
        "lr": {"values": [0.001]},
        "dataset_size": {"values": [1_000_000]},
        "fmap_size": {"values": [32]},
    }
}

def main(config=None):
    import numpy as np
    import random
    torch.manual_seed(1)
    random.seed(1)
    np.random.seed(1)
    wandb.init()
    if config is None:
        print("Config is none")
        config = wandb.config

    train, test = MultiAgentDataset([str(i) for i in range(20)], False, length=config.dataset_size), PairedImageDataset(True)

    train_dataloader = DataLoader(train, batch_size=config.batch_size, shuffle=False)
    action_space = VizdoomEnv("/home/dvasilev/doom_icm/mario_icm/ViZDoom/custom_my_way_home.cfg", frame_skip=1).action_space
    obs_shape = (4, 42, 42)
    """
    icm = ICM_Old(action_space.n, 4, 0.8, 0.2, use_softmax=False, hidden_layer_neurons=256, eta=0.2, feature_map_qty=32)
    """
    icm = ICM(action_space, obs_shape, inv_scale=0.8, forward_scale=0.2, hidden_layer_neurons=config.hidden_layer_neurons,
              discrete=config.discrete, pde=config.pde, freeze_grad=config.freeze_grad,
              eta=0.02, apply_bounder=config.apply_bounder, pde_regulizer=0.,
              inverse_bottleneck=config.inverse_bottleneck, inverse_group=config.inverse_group, inverse_fc_qty=config.inverse_fc_qty,
              feature_skip_conn=config.feature_skip_conn, feature_consecutive_convs=config.feature_consecutive_convs, 
              feature_batch_norm=config.feature_batch_norm, feature_total_blocks=config.feature_total_blocks)
    icm = icm.to("cuda:0")

    optim = torch.optim.Adam(icm.parameters(), lr=config.lr)
    test_dataloader = DataLoader(test, batch_size=config.batch_size, shuffle=False)
    slice_train(train_dataloader, icm, optim, test_dataloader, config)


if __name__ == "__main__":

    class Mock():
        def __init__(self):
            self.batch_size = 500
            self.lr = 0.001
            self.dataset_size = 1_000_000
            self.fmap_size = 32
            self.output_neurons = 256
            self.pde = False
            self.freeze_grad = False
            self.pde_regulizer = 0.
            self.apply_bounder = True
            self.discrete = False
            self.learning_steps=10_000
            
    #config = Mock()
    #main()
    
    sweep_id = wandb.sweep(
        sweep=sweep_configuration,
        project='mario_icm-ViZDoom_Fixated_buffer_experiment'
        )

    wandb.agent(sweep_id, function=main, count=3)
