from architecture.neural_network_ext import ActorCritic

# Learning rate of A2C optimizer
# SB3+Pathak (One changed parameter)
LR=0.0001 # was 0.0007

# Number of steps, at which buffer collection is happening
# SB3
NUM_STEPS=2000

# Number of training epochs on buffer of PPO
EPOCHS=5

# Batch size whilist training on buffer of PPO
BATCH_SIZE=200

# Number of steps, at which buffer for icm is collected. After this steps, icm will be updated
ICM_NUM_STEPS=20

# Number of architecture in A2C
# SB3
NUM_AGENTS=20

# Discount factor
# SB3
GAMMA=0.99

# Lambda parameter for GAE
# SB3
GAE_LAMBDA=0.95

# Entropy coef
# SB3; Was 0.0005
ENTROPY_COEF=0.

# Value loss coefficient
# SB3
VALUE_LOSS_COEF=0.25 # was 0.25

# Max grad norm
# SB3
MAX_GRAD_NORM=0.5

# Use RMS prop
# SB3 True
RMS_PROP=False

# Policy_kwargs
POLICY_KWARGS=dict(features_extractor_class=ActorCritic)

config_dict = {
    "LR": LR,
    "NUM_STEPS - number of steps, at which buffer collection is happening": NUM_STEPS,
    "EPOCHS - Number of training epochs on buffer of PPO": EPOCHS,
    "BATCH_SIZE - Batch size whilist training on buffer of PPO": BATCH_SIZE,
    "ICM_NUM_STEPS - Number of steps, at which buffer for icm is collected. After this steps, icm will be updated": ICM_NUM_STEPS,
    "NUM_AGENTS - number of architecture, that work in parallel": NUM_AGENTS,
    "GAMMA - discount factor of reward": GAMMA,
    "GAE_LAMBDA - lambda parameter for GAE": GAE_LAMBDA,
    "ENTROPY_COEF - entropy coefficient of entropy loss": ENTROPY_COEF,
    "VALUE_LOSS_COEF - value loss coefficient of A2C loss": VALUE_LOSS_COEF,
    "MAX_GRAD_NORM - maximal gradient norm for A2C gradients": MAX_GRAD_NORM,
    "RMS_PROP - do we use RMS_PROP or Adam as optimizer (False -> Adam, True -> RMS_PROP)": RMS_PROP,
    "Policy kwargs argument": str(POLICY_KWARGS)
}
