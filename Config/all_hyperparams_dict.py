from Config import A2C_config, environment_config, ICM_config

HYPERPARAMS = {
"Seed": environment_config.SEED,
"No_shared": A2C_config.NO_SHARED,
"LR": A2C_config.LR,
"Max_episode_length": A2C_config.MAX_EPISODE_LENGTH,
"Number_of_steps": A2C_config.NUM_STEPS,
"Gamma": A2C_config.GAMMA,
"GAE_lambda": A2C_config.GAE_LAMBDA,
"Entropy_coefficient": A2C_config.ENTROPY_COEF,
"Value_loss_coeficient": A2C_config.VALUE_LOSS_COEF,
"Maximum_grad_norm": A2C_config.MAX_GRAD_NORM,
"Num_processes": A2C_config.NUM_PROCESSES,
"Action_space_size": environment_config.ACTION_SPACE_SIZE, 
"Device": environment_config.DEVICE,
"ETA_ICM": ICM_config.ETA,
"STATE_SPACE_DIM_ICM": ICM_config.STATE_SPACE_DIM,
"BETA_ICM": ICM_config.BETA}