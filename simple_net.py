import torch
from torch import nn
from Config import ENV_CFG, ICM_CFG

class SimplePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        state_dim = 288
        hidden_layer_neurons = ICM_CFG.HIDDEN_LAYERS
        action_classes = ENV_CFG.ACTION_SPACE_SIZE
        self.feature_extractor = nn.Sequential(nn.Conv2d(in_channels = ENV_CFG.TEMPORAL_CHANNELS, 
                                out_channels = ICM_CFG.FMAP_QTY, kernel_size = 3, stride = 2, 
                                padding = 1),
                      nn.ELU(),
                      nn.Conv2d(in_channels = ICM_CFG.FMAP_QTY, out_channels = ICM_CFG.FMAP_QTY, 
                                kernel_size = 3, stride = 2, padding = 1),
                      nn.ELU(),
                      nn.Conv2d(in_channels = ICM_CFG.FMAP_QTY, out_channels = ICM_CFG.FMAP_QTY, 
                                kernel_size = 3, stride = 2, padding = 1),
                      nn.ELU(),
                      nn.Conv2d(in_channels = ICM_CFG.FMAP_QTY, out_channels = ICM_CFG.FMAP_QTY, 
                                kernel_size = 3, stride = 2, padding = 1),
                      nn.ELU(),
                      nn.Flatten())

        self.predictor = torch.nn.Sequential(nn.Linear(2*state_dim, hidden_layer_neurons),
                                                     nn.ReLU(),
                                                     nn.Linear(hidden_layer_neurons, action_classes))

    def forward(self, previous_state, next_state):
        latent_prev, latent_next = self.feature_extractor(previous_state), self.feature_extractor(next_state)
        print(latent_prev)
        print(latent_next)
        cat_state = torch.cat([latent_prev, latent_next], 1)
        return self.predictor(cat_state)
