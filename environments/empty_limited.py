from gym_minigrid.envs.empty import EmptyEnv
from gym_minigrid.register import register
from gym_minigrid.minigrid import MiniGridEnv

class EmptyEnvLimited(EmptyEnv):
    """
    Classic 4 rooms gridworld environment.
    Can specify agent and goal position, if not it set at random.
    """
    def __init__(
        self,
        size=9,
        agent_start_pos=(1,1),
        agent_start_dir=0,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        super(EmptyEnv, self).__init__(
            grid_size=size,
            max_steps=500,
        )

    

register(
    id='MiniGrid-EmptyEnvLimited-v0',
    entry_point='environments.empty_limited:EmptyEnvLimited'
)