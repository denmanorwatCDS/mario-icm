import pickle
import numpy as np
import gym
from environments.fast_grid import DIRECTIONS_ORDER
from environments.fast_grid import FOUR_ROOMS_OBSTACLES
import copy


def prepare_maps(_init):
    env = _init()
    grid_size = env.size
    prev_image_grid = [[0 for i in range(grid_size[1])] for j in range(grid_size[0])]
    new_image_grid = [[0 for i in range(grid_size[1])] for j in range(grid_size[0])]
    mask = FOUR_ROOMS_OBSTACLES.copy()
    mask[grid_size[0]-2, grid_size[1]-2] = 1
    for action in [0, 1, 2, 3]:
        for i in range(grid_size[1]):
            for j in range(grid_size[0]):
                if i != grid_size[1]-2 and j != grid_size[0]-2:
                    env = _init(agent_y_pos=i, agent_x_pos=j)
                    prev_image_grid[i][j] = env.reset()
                    new_image_grid[i][j] = env.step(action, log=False)[0]
                else:
                    prev_image_grid[i][j] = np.zeros((1, 42, 42), dtype=np.float64)
                    new_image_grid[i][j] = np.zeros((1, 42, 42), dtype=np.float64)
        prev_image_grid = np.array(prev_image_grid)
        new_image_grid = np.array(new_image_grid)
        image_dictionary = {"prev_image": prev_image_grid,
                            "new_image": new_image_grid,
                            "mask": mask}
        with open("tmp/image_array_{}.pkl".format(DIRECTIONS_ORDER[action]), "wb") as file:
            pickle.dump(image_dictionary, file)
