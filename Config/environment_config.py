import torch

#Config cell
SEED = 1

ALL_ACTION_SPACE = [['NOOP'],
    ['right'], ['right', 'A'], ['right', 'B'], ['right', 'A', 'B'],
    ['A'], 
    ['left'], ['left', 'A'], ['left', 'B'], ['left', 'A', 'B'],
    ['down'],
    ['up']]

ACTION_NAMES = ["+".join(action_sequence) for action_sequence in ALL_ACTION_SPACE]

# How many actions there are available
ACTION_SPACE_SIZE = len(ALL_ACTION_SPACE)

#Batch size
BATCH_SIZE = 32

#Resized image size
RESIZED_SIZE = (42, 42)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Temporal channels quantity
TEMPORAL_CHANNELS = 3 #4