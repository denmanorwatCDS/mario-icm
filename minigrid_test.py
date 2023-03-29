from stable_baselines3.common.env_checker import check_env
from environments.fast_grid import GridWorld, FOUR_ROOMS_OBSTACLES

grid_size = FOUR_ROOMS_OBSTACLES.shape
env = GridWorld(grid_size=(16, 16),
                obstacle_mask=FOUR_ROOMS_OBSTACLES,
                agent_pos=(1, 1),
                goal_pos=(grid_size[0]-2, grid_size[1]-2), pixel_size=8, time_limit=50,
                color_map=dict(floor=.0, obstacle=.43, agent=.98, target=.8),
                const_punish=0.02*0.9, terminal_decay=1.,
                warp_size=(42, 42), beautiful=True)
check_env(env, warn=True)