# Learning rate of A2C optimizer
LR = 1e-04

# Number of forward steps in A3C (default: 20):
NUM_STEPS = 20 # 20

# Number of agents in A2C
NUM_AGENTS = 20

# Discount factor
GAMMA = 0.99

# Lambda parameter for GAE
GAE_LAMBDA = 1.

# Entropy coef
ENTROPY_COEF = 0.0005

# Value loss coefficient
VALUE_LOSS_COEF = 0.5

# Max grad norm
MAX_GRAD_NORM = 40.

# Use RMS prop
RMS_PROP = False

# Policy hidden layer
POLICY_NEURONS = 256

# Value hidden layer
VALUE_NEURONS = 256