import pickle
from ICM.ICM_buffer import ICMBuffer
from Config import ICM_CFG, ENV_CFG
from Config.all_hyperparams_dict import HYPERPARAMS
from ICM.ICM import ICM
from torch import optim, nn
from torch.functional import F
from PIL import Image
import torch
import wandb

model = ICM(ENV_CFG.ACTION_SPACE_SIZE, temporal_channels = ENV_CFG.TEMPORAL_CHANNELS, 
    hidden_layer_neurons=ICM_CFG.HIDDEN_LAYERS, eta = ICM_CFG.ETA, feature_map_qty=ICM_CFG.FMAP_QTY
    ).to(ENV_CFG.DEVICE).train()
icm_optimizer = optim.Adam(model.parameters(), lr=ICM_CFG.LR)
icm_buffer = ICMBuffer(ICM_CFG.BATCH_SIZE, ICM_CFG.BUFFER_SIZE)

run = wandb.init(project = "ICM_debug", config = HYPERPARAMS)

path_to_file = "/home/dvasilev/mario_icm/debug/train_set/pickled_train_list.pkl"
with open(path_to_file, "rb") as pickled_buffer:
    icm_buffer.buffer = pickle.load(pickled_buffer)

def train(model, optimizer, buffer):
    observations, actions, next_observations = buffer.get_triplets()
    print("Observation shape: {}".format(observations.shape))
    predicted_actions, predicted_states, next_states =\
        model(observations, actions, next_observations)
    CE_loss = nn.CrossEntropyLoss()
    action_one_hot = F.one_hot(actions.flatten(), num_classes = ENV_CFG.ACTION_SPACE_SIZE)
    state_prediction_loss =\
        (1/2*(next_states-predicted_states)**2).sum(dim = 1).mean()
    action_prediction_loss =\
        CE_loss(predicted_actions, action_one_hot.argmax(dim = 1)).mean()
    icm_loss = (ICM_CFG.BETA*state_prediction_loss + 
                (1-ICM_CFG.BETA)*action_prediction_loss)
    log_losses(state_prediction_loss, action_prediction_loss, icm_loss)
    wandb.log({"Action prediction loss": action_prediction_loss.item()})

    optimizer.zero_grad()
    icm_loss.backward()
    optimizer.step()

def log_losses(state_prediction_loss, action_prediction_loss, icm_loss):
    state_value, action_value, icm_value =\
            state_prediction_loss.item(), action_prediction_loss.item(), icm_loss.item()
    wandb.log({"State prediction loss": state_value,
                "Action prediction loss": action_value,
                "Total icm loss": icm_value})

while True:
    train(model, icm_optimizer, icm_buffer)