import torch.optim
import os

from doom_samples.custom_dataset.stable_buffer import PairedImageDataset, MultiAgentDataset
from torch.utils.data import DataLoader
from icm_mine.icm import ICM
import wandb


def classical_train(train_dataloader, ICM, optim, test_dataloader, epochs=1):
    for i in range(epochs):
        for start_frames, end_frames, actions in train_dataloader:
            start_frames, end_frames, actions = start_frames.to("cuda:0"), end_frames.to("cuda:0"), actions.to("cuda:0")
            forward_loss, inverse_loss = ICM.get_losses(start_frames, actions, end_frames)
            icm_loss = forward_loss + inverse_loss
            optim.zero_grad()
            icm_loss.backward()
            optim.step()
            wandb.log({"Forward loss": forward_loss.cpu().detach(),
                       "Inverse loss": inverse_loss.cpu().detach(),
                       "ICM loss": icm_loss.cpu().detach()})

        correct_predictions = 0
        for start_frames, end_frames, actions in test_dataloader:
            start_frames, actions, end_frames = start_frames.to("cuda:0"), actions.to("cuda:0"), end_frames.to("cuda:0")
            with torch.no_grad():
                action_logits, _, _ = ICM.forward(start_frames, actions, end_frames)
                predicted_labels = action_logits.argmax(dim=1)
                correct_predictions += (predicted_labels == actions).sum()
        print(correct_predictions / (len(train_dataloader) * batch_size))
        wandb.log({"Accuracy": correct_predictions / (len(test_dataloader) * batch_size)},
                  step=len(train_dataloader) * epochs)


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


def slice_train(train_dataloader, ICM, optim):
    inverse_loss = torch.tensor(1.)
    counter = 0
    while inverse_loss > 0.25 and counter < 7_500:
        for start_frames, end_frames, actions in train_dataloader:
            start_frames, end_frames, actions = start_frames.to("cuda:0"), end_frames.to("cuda:0"), actions.to("cuda:0")
            forward_loss, inverse_loss = ICM.get_losses(start_frames, actions, end_frames)
            icm_loss = forward_loss + inverse_loss
            optim.zero_grad()
            icm_loss.backward()
            optim.step()
            if counter%1000 == 0:
                wandb.log({"Forward loss": forward_loss.cpu().detach(),
                           "Inverse loss": inverse_loss.cpu().detach(),
                           "ICM loss": icm_loss.cpu().detach()})
            counter += 1
    wandb.log({"Forward loss": forward_loss.cpu().detach(),
               "Inverse loss": inverse_loss.cpu().detach(),
               "ICM loss": icm_loss.cpu().detach()})



sweep_configuration = {
    "method": "grid",
    "parameters": {
        "batch_size": {"values": [1000, 2000]},
        "lr": {"values": [1e-02, 1e-03, 1e-04]},
        "dataset_size": {"values": [10_000, 1_000_000]}
    }
}

def main():
    import numpy as np
    import random
    torch.manual_seed(1)
    random.seed(1)
    np.random.seed(1)

    wandb.init()
    train, test = MultiAgentDataset([str(i) for i in range(20)], False, length=wandb.config.dataset_size), PairedImageDataset(True)

    train_dataloader = DataLoader(train, batch_size=wandb.config.batch_size, shuffle=False)

    icm = ICM(3, 4, 0.8, 0.2, False, 256, 0.2, 32)
    icm = icm.to("cuda:0")

    optim = torch.optim.Adam(icm.parameters(), lr=wandb.config.lr)
    slice_train(train_dataloader, icm, optim)


sweep_id = wandb.sweep(sweep=sweep_configuration, project='Fixated doom sweeps')
wandb.agent(sweep_id, function=main, count=32)
