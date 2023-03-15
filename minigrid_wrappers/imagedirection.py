import gym
from gym import spaces
from gym_minigrid.wrappers import FullyObsWrapper, RGBImgObsWrapper

class FullyObsDirectionWrapper(FullyObsWrapper):
    """
    Fully observable gridworld using a compact grid encoding
    """

    def __init__(self, env):
        super().__init__(env)

    def observation(self, obs):
        full_grid = super().observation(obs)

        return {
            'mission': obs['mission'],
            'direction': obs['direction'],
            'image': full_grid
        }

class RGBImgObsDirectionWrapper(RGBImgObsWrapper):
    """
    Wrapper to use fully observable RGB image as the only observation output,
    no language/mission. This can be used to have the agent to solve the
    gridworld in pixel space.
    """

    def __init__(self, env, tile_size=8):
        super().__init__(env, tile_size=tile_size)

    def observation(self, obs):
        rgb_img = super().observation(obs)['image']

        return {
            'mission': obs['mission'],
            'direction': obs['direction'],
            'image': rgb_img
        }