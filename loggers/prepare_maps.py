import pickle
import numpy as np
import gym

def prepare_maps(_init):
    env = _init()
    map = env.unwrapped.map
    mask = (map == 2)
    grid_side = mask.shape[0]
    image_grid = [[0 for i in range(grid_side)] for j in range(grid_side)]
    for i in range(grid_side):
        for j in range(grid_side):
            image_grid[i][j] = _init(agent_y_pos=i, agent_x_pos=j).reset()
    print(image_grid[i][j].shape)
    image_grid = np.array(image_grid)
    position_array = {"mask": mask,
                      "image_grid": image_grid}
    with open("tmp/image_array.pkl", "wb") as file:
        pickle.dump(position_array, file)