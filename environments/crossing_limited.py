from gym_minigrid.envs.fourrooms import FourRoomsEnv
from gym_minigrid.register import register
from gym_minigrid.minigrid import MiniGridEnv
from gym_minigrid.minigrid import Lava

class FourRoomsEnvLimited(FourRoomsEnv):
    """
    Classic 4 rooms gridworld environment.
    Can specify agent and goal position, if not it set at random.
    """

    def __init__(self, grid_size, agent_pos=None, goal_pos=None):
        self._agent_default_pos = agent_pos
        self._goal_default_pos = goal_pos
        super(FourRoomsEnv, self).__init__(grid_size=grid_size, max_steps=5_000)
    

register(
    id='MiniGrid-FourRoomsEnvLimited-v0',
    entry_point='environments.crossing_limited:FourRoomsEnvLimited'
)