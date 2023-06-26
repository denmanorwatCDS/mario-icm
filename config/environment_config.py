#Config cell
SEED = 100

#Resized image size
RESIZED_SIZE = (42, 42)

# Temporal channels quantity
TEMPORAL_CHANNELS = 1 #4

# Action skip
ACTION_SKIP = 4

# FPS of submitted video
FPS = 30

# Device
MODEL_DEVICE = "cuda:0"
MOTIVATION_DEVICE = "cuda:0"


config_dict = {
    "SEED": SEED,
    "RESIZED SIZE - resized image size": RESIZED_SIZE,
    "TEMPORAL CHANNELS - quantity of sequential frames, placed in channel places": TEMPORAL_CHANNELS,
    "ACTION_SKIP - how much action repeat in environment": ACTION_SKIP,
    "FPS - fps of submitted video": FPS
}