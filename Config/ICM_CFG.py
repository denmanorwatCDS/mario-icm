# Eta coefficient for ICM - multiplier of reward
ETA = 0.2

# Beta coefficient for ICM - weighted sum of forward and backward model's losses
BETA = 0.2

# Hidden layers of ICM linear layer
HIDDEN_LAYERS = 256

# Feature maps quantity inside feature net
FMAP_QTY = 32

# Learn only on latest observations. If true, BUFFER_SIZE and BATCH_SIZE irrelevant
SAVE_LATEST = False

# Buffer size of ICM buffer
BUFFER_SIZE = 16_000#4

# Batch size of ICM buffer
BATCH_SIZE = 128

# ICM learning rate
LR = 1e-03