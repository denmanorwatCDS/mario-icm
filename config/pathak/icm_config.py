# Eta coefficient for ICM - multiplier of reward
# Pathak 
ETA = 0.2

# Beta coefficient for ICM - weighted sum of forward and backward model's losses
# Pathak
BETA = 0.2

# Hidden layers of ICM linear layer
# Pathak
HIDDEN_LAYERS = 256

# Feature maps quantity inside feature net
# Pathak
FMAP_QTY = 32

# ICM learning rate
# Pathak
LR = 1e-03

# ICM inverse loss scale
# Pathak: "batch" factorization (Pathak had 20)
INVERSE_SCALE=1

# ICM forward loss scale
# Pathak: "batch" and "features" factorization (Pathak had 288*20)
FORWARD_SCALE=1

# ICM grad norm
GRAD_NORM = 40.0

# Warmup steps:
WARMUP = 0

# Intrinsic reward coef:
INTRINSIC_REWARD_COEF = 1.

config_dict = {
    "ETA - multiplier of intrinsic reward": ETA,
    "BETA - weighted sum of forward and inverse losses": BETA,
    "HIDDEN_LAYERS - quantity of neurons in hidden layer of ICM": HIDDEN_LAYERS,
    "FMAP_QTY - quantity of feature maps in each convolutional layer": FMAP_QTY,
    "LR - learning rate of ICM model": LR,
    "INVERSE_SCALE - factor before inverse loss - as Pathak stated, for \"batch\' factorization": INVERSE_SCALE,
    "FORWARD_SCALE - similiar to INVERSE_SCALE, but not only batch, but also features factorization": FORWARD_SCALE,
    "GRAD_NORM - gradient norm clipping for ICM": GRAD_NORM
}
