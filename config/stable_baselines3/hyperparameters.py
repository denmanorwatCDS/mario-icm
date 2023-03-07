from config.stable_baselines3 import a2c_config, environment_config, icm_config

HYPERPARAMS = {}
for name, config in {"A2C": a2c_config.config_dict, "ICM": icm_config.config_dict, "Environment": environment_config.config_dict}.items():
    for param, value in config.items():
        HYPERPARAMS[name+" " +param] = value
