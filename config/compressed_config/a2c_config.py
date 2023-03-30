from agents.neural_network_ext import ActorCritic
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike

# Learning rate of A2C optimizer
# SB3+Pathak (One changed parameter)
LR = 0.0007 # was 0.0007

# Number of forward steps in A3C (default: 20):
# SB3
NUM_STEPS = 5 # 20

# Number of agents in A2C
# SB3
NUM_AGENTS = 6

# Discount factor
# SB3
GAMMA = 0.99

# Lambda parameter for GAE
# SB3
GAE_LAMBDA = 1.

# Entropy coef
# SB3
ENTROPY_COEF = 0.00005

# Value loss coefficient
# SB3
VALUE_LOSS_COEF = 0.25 # was 0.25

# Max grad norm
# SB3
MAX_GRAD_NORM = 0.5

# Use RMS prop
# SB3 True
RMS_PROP = False

# Policy_kwargs
POLICY_KWARGS = dict(features_extractor_class=ActorCritic)

config_dict = {
    "LR": LR,
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