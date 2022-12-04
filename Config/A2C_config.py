# Use an optimizer without shared momentum
NO_SHARED = False

LR = 0.0001
MAX_EPISODE_LENGTH = 10000000 # Subject to change. It maybe to high or to low

# Number of forward steps in A3C (default: 20):
NUM_STEPS = 2

# Discount factor
GAMMA = 0.99

# Lambda parameter for GAE
GAE_LAMBDA = 1.

# Entropy coef
ENTROPY_COEF = 0.01

# Value loss coefficient
VALUE_LOSS_COEF = 0.5

# Max grad norm
MAX_GRAD_NORM = 50

# How many training processes to use
NUM_PROCESSES = 20