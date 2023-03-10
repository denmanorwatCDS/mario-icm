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

#Resized image size
RESIZED_SIZE = (42, 42)

# Temporal channels quantity
TEMPORAL_CHANNELS = 4 #4

# Action skip
ACTION_SKIP = 6

# FPS of submitted video
FPS = 30

config_dict = {
    "SEED": SEED,
    "ACTION_SPACE_SIZE": ACTION_SPACE_SIZE,
    "RESIZED SIZE - resized image size": RESIZED_SIZE,
    "TEMPORAL CHANNELS - quantity of sequential frames, placed in channel places": TEMPORAL_CHANNELS,
    "ACTION_SKIP - how much action repeat in environment": ACTION_SKIP,
    "FPS - fps of submitted video": FPS
}