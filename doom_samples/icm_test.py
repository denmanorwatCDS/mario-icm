import torch.optim
import os

from doom_samples.custom_dataset.stable_buffer import PairedImageDataset, MultiAgentDataset
from torch.utils.data import DataLoader
from icm_mine.icm import ICM
import wandb
train, test = MultiAgentDataset([str(i) for i in range(20)], False), PairedImageDataset(True)

batch_size = 100
train_dataloader = DataLoader(train, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test, batch_size=batch_size, shuffle=False)
dataset_size = len(train_dataloader)
dataset_iter = int(dataset_size/batch_size)
epochs = 10
ICM = ICM(3, 4, 0.8, 0.2, False, 256, 0.2, 32)
ICM = ICM.to("cuda:0")

optim = torch.optim.Adam(ICM.parameters(), lr=1e-03)
wandb.init()
print(len(train_dataloader))

for i in range(1):
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
            correct_predictions += (predicted_labels==actions).sum()
    print(correct_predictions/(len(train_dataloader)*batch_size))
    wandb.log({"Accuracy": correct_predictions/(len(test_dataloader)*batch_size)}, step=len(train_dataloader)*epochs)
