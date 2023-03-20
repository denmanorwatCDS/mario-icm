import numpy as np
import gym
from gym import spaces
from gym_minigrid.wrappers import FullyObsWrapper, RGBImgObsWrapper
from gym_minigrid.wrappers import ViewSizeWrapper
from torchvision.transforms import Grayscale

class RGBImgObsDirectionWrapper(RGBImgObsWrapper):
    """
    Wrapper to use fully observable RGB image as the only observation output,
    no language/mission. This can be used to have the agent to solve the
    gridworld in pixel space.
    """

    def __init__(self, env, grid_size, tile_size=8, restriction=None):
        self.restriction = restriction
        self.grid_size = grid_size
        self.gray = Grayscale(num_output_channels=1)

        if restriction is not None:
            super(RGBImgObsWrapper, self).__init__(env)
            self.tile_size = tile_size
            self.observation_space.spaces['image'] = spaces.Box(
                low=0,
                high=255,
                shape=((2*restriction+1) * tile_size, (2*restriction+1) * tile_size, 3),
                dtype='uint8'
            )
        else:
            super().__init__(env, tile_size)

    def step(self, action):
        new_obs, reward, done, info = self.env.step(action)
        info['position'] = self.unwrapped.agent_pos
        info["full_img"] = self.full_image
        obs = self.observation(new_obs)
        return obs, reward, done, info
    
    def observation(self, obs):
        current_position = self.unwrapped.agent_pos
        rgb_img = super().observation(obs)['image']
        self.full_image = rgb_img[..., 0:1]*0.2989 + rgb_img[..., 1:2]*0.5870 + 0.1140*rgb_img[..., 2:]
        if self.restriction is not None:
            current_x, current_y = current_position[0], current_position[1]
            min_x = max(0, (current_x-self.restriction)*self.tile_size)
            max_x = min((current_x+self.restriction+1)*self.tile_size, self.grid_size*self.tile_size)
            min_y = max(0, (current_y-self.restriction)*self.tile_size)
            max_y = min((current_y+self.restriction+1)*self.tile_size, self.grid_size*self.tile_size)
            rgb_img = rgb_img[min_y: max_y, min_x: max_x]
            rgb_img = self.center_image_at_agent(rgb_img, current_x, current_y)

        return {
            'mission': obs['mission'],
            'direction': obs['direction'],
            'image': rgb_img,
            'position': current_position
        }
    
    def center_image_at_agent(self, rgb_img, current_x, current_y):
        if self.restriction is not None:
            # Calculation in grid cells
            window_size = 2*self.restriction+1
            delta_x_right = max(0, (current_x+self.restriction)-(self.grid_size-1))
            delta_x_left = max(0, self.restriction - current_x)
            delta_y_high = max(0, (current_y+self.restriction)-(self.grid_size-1))
            delta_y_low = max(0, self.restriction - current_y)
            # Calculation in pixels
            centered_rgb_image = np.zeros(((self.restriction*2+1)*self.tile_size, (self.restriction*2+1)*self.tile_size, 3),
                                          dtype=np.uint8)
            centered_rgb_image[delta_y_low*self.tile_size: (window_size-delta_y_high)*self.tile_size,
                               delta_x_left*self.tile_size: (window_size-delta_x_right)*self.tile_size] = rgb_img
            centered_rgb_image[self.restriction*self.tile_size+1: (self.restriction+1)*self.tile_size, 
                               self.restriction*self.tile_size+1: (self.restriction+1)*self.tile_size] = 240
        return centered_rgb_image
