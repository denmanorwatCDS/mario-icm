import pickle
import numpy as np
import gym
from environments.fast_grid import DIRECTIONS_ORDER
from environments.fast_grid import FOUR_ROOMS_OBSTACLES
import copy
import torch
import os

"""
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
"""

def prepare_maps(_init):
    if os.path.exists("tmp/image_array.pkl"):
        return
    ACTIONS = [0, 1, 2, 3]
    env = _init()
    grid_size_0, grid_size_1 = env.size
    images = {"prev_obs": [], "action": [], "next_obs": [], "position": [],
              "obs_matrix": [[np.zeros((env.size)) for i in range(grid_size_0)] for j in range(grid_size_1)]}
    obs_shape = env.reset().shape
    for action in ACTIONS:
        for i in range(grid_size_0):
            for j in range(grid_size_1):
                if FOUR_ROOMS_OBSTACLES[i][j] == 0 and (i, j) != (grid_size_0-2, grid_size_1-2):
                    env = _init(i, j)
                    start_obs = env.reset()
                    next_obs, _, _, info = env.step(action)
                    pos = info["position"]
                    images["prev_obs"].append(start_obs)
                    images["action"].append(action)
                    images["next_obs"].append(next_obs)
                    images["position"].append(pos)
                    images["obs_matrix"][i][j] = start_obs
                elif (i, j) == (grid_size_0-2, grid_size_1-2):
                    env = _init(i, j)
                    start_obs = env.reset()
                    images["obs_matrix"][i][j] = start_obs
                else:
                    images["obs_matrix"][i][j] = np.zeros(obs_shape)
    with open("tmp/image_array.pkl", "wb") as file:
        images["prev_obs"] = torch.tensor(images["prev_obs"])
        images["action"] = torch.tensor(images["action"])
        images["next_obs"] = torch.tensor(images["next_obs"])
        images["position"] = torch.tensor(images["position"])
        images["grid_size"] = (grid_size_0, grid_size_1)
        images["obs_matrix"] = torch.tensor(images["obs_matrix"])
        pickle.dump(images, file)
