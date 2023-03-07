from agents.neural_network_ext import ActorCritic
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike

# Learning rate of A2C optimizer
# Pathak
LR = 1e-04

#LR multiplier
# Pathak: batch factorization (Pathak had 20)
LR_FACTOR = 1

# Number of forward steps in A3C (default: 20):
# Pathak
NUM_STEPS = 5 # 20

# Number of agents in A2C
# Pathak
NUM_AGENTS = 20

# Discount factor
# Pathak
GAMMA = 0.99

# Lambda parameter for GAE
# Pathak
GAE_LAMBDA = 1.

# Entropy coef
# Pathak
ENTROPY_COEF = 0.0005

# Value loss coefficient
# Pathak
VALUE_LOSS_COEF = 0.5

# Max grad norm
# Pathak
MAX_GRAD_NORM = 40.

# Use RMS prop
# Pathak
RMS_PROP = False

# Policy_kwargs
POLICY_KWARGS = dict(features_extractor_class=ActorCritic)

config_dict = {
    "LR": LR,
    "LR_FACTOR - batch normalization from Pathak repo": LR_FACTOR,
    "NUM_STEPS - quantity of rollout steps": NUM_STEPS,
    "NUM_AGENTS - number of agents, that work in parallel": NUM_AGENTS,
    "GAMMA - discount factor of reward": GAMMA,
    "GAE_LAMBDA - lambda parameter for GAE": GAE_LAMBDA,
    "ENTROPY_COEF - entropy coefficient of entropy loss": ENTROPY_COEF,
    "VALUE_LOSS_COEF - value loss coefficient of A2C loss": VALUE_LOSS_COEF,
    "MAX_GRAD_NORM - maximal gradient norm for A2C gradients": MAX_GRAD_NORM,
    "RMS_PROP - do we use RMS_PROP or Adam as optimizer (False -> Adam, True -> RMS_PROP)": RMS_PROP,
    "Policy kwargs argument": str(POLICY_KWARGS)
}