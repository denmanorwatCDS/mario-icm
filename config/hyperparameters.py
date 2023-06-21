from config import agent_config, environment_config, icm_config

HYPERPARAMS = {}
for name, config in {"A2C": agent_config.config_dict, "ICM": icm_config.config_dict, "Environment": environment_config.config_dict}.items():
    for param, value in config.items():
        HYPERPARAMS[name+" " +param] = value
