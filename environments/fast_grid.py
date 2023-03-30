from __future__ import annotations

import gym
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Tuple
from gym import spaces
from torchvision.transforms import Resize
from skimage.transform import resize
from PIL import Image



MOVE_DIRECTIONS = {
    # (i, j)-based coordinates [or (y, x) for the viewer]
    'right': (0, 1), 'down': (1, 0), 'left': (0, -1), 'up': (-1, 0)
}

DIRECTIONS_ORDER = ['right', 'down', 'left', 'up']
IDX_TO_OBJECT = ["FLOOR", "WALL"]
NAME_TO_COLOR = {"AGENT": np.array([240, 240, 240]).astype(np.uint8), "GOAL": np.array([0, 255, 0]).astype(np.uint8),
                 "WALL": np.array([100, 100, 100]).astype(np.uint8),  "FLOOR": np.array([0, 0, 255]).astype(np.uint8)}


FOUR_ROOMS_OBSTACLES = """
xxxxxxxxxxxxxxxx
x-------x------x
x-------x------x
x--------------x
x-------x------x
x-------x------x
x-------x------x
x-------x------x
xxxxx-xxxxxxx-xx
x-------x------x
x-------x------x
x--------------x
x-------x------x
x-------x------x
x-------x------x
xxxxxxxxxxxxxxxx
"""
FOUR_ROOMS_OBSTACLES = np.array([
    [int(c == 'x') for c in row]
    for row in FOUR_ROOMS_OBSTACLES.strip().split('\n')
])

MOVE_DIRECTIONS = {
    # (i, j)-based coordinates [or (y, x) for the viewer]
    'right': (0, 1), 'down': (1, 0), 'left': (0, -1), 'up': (-1, 0)
}
DIRECTIONS_ORDER = ['right', 'down', 'left', 'up']
TURN_DIRECTIONS = {'right': 1, 'left': -1}

def rgb2gray(img):
    gray = 0.2989*img[:, :, 0] + 0.5870*img[:, :, 1] + 0.1140*img[:, :, 2]
    return gray.astype(np.uint8)

class MoveDynamics:
    @staticmethod
    def try_move(
            position: tuple[int, int], move_direction: tuple[int, int],
            shape: tuple[int, int], obstacle_mask: np.ndarray
    ):
        """
        Performs the move if it's allowed.
        Returns new position and flag whether or not the move was successful.
        """
        new_position = MoveDynamics.move(position, move_direction)
        new_position = MoveDynamics.clip2d(new_position, shape)

        success = MoveDynamics.is_move_successful(position, new_position, obstacle_mask)
        if not success:
            new_position = position
        return new_position, success

    @staticmethod
    def is_move_successful(
            position: tuple[int, int], new_position: tuple[int, int],
            obstacle_mask: np.ndarray
    ):
        """Checks whether move is happened and is allowed."""
        # changed position but not stepped into the wall
        return new_position != position and not obstacle_mask[new_position]

    @staticmethod
    def move(position: tuple[int, int], direction: tuple[int, int]):
        """
        Calculates new position for the move to the specified direction
        without checking whether this move is allowed or not.
        """
        i, j = position
        i += direction[0]
        j += direction[1]
        return i, j

    @staticmethod
    def clip2d(position: tuple[int, int], shape: tuple[int, int]):
        """Clip position to be inside specified rectangle."""
        def clip(x, high):
            if x >= high:
                return high - 1
            if x < 0:
                return 0
            return x
        i, j = position
        i = clip(i, shape[0])
        j = clip(j, shape[1])
        return i, j



TPos = Tuple[int, int]
TSize2d = Tuple[int, int]

class GridWorld(gym.Env):
    size: TSize2d
    init_agent_pos: TPos
    obstacles: np.ndarray
    
    agent_pos: TPos
    
    terminal: TPos
    reward: float
    const_punish: float
    terminal_decay: float
    time_limit: int | None

    pixel_size: int
    color_map: dict
    
    _pos: TPos
    _obs: np.ndarray
    _step: int
    _terminated: bool
    _next_pos_cache: dict[tuple[TPos, int], TPos]

    def __init__(
        self,
        *,
        grid_size: TSize2d, 
        agent_pos: TPos = None,
        obstacle_mask: np.ndarray = None,
        goal_pos: TPos = None,
        reward: float = None,
        const_punish: float = 0.,
        terminal_decay: float = 1.0,
        time_limit: int = None,
        pixel_size: int = 1,
        color_map: dict = None,
        warp_size = None,
        beautiful = False
    ):

        self.size = grid_size
        self.init_agent_pos = agent_pos
        self.beautiful = beautiful
        
        self.obstacles = np.zeros(self.size, dtype=int)
        if obstacle_mask is not None:
            self.obstacles[:] = obstacle_mask

        self.terminal = goal_pos
        self.reward = reward
        assert const_punish >= 0., 'I expect negative punish'
        self.const_punish = const_punish
        assert terminal_decay <= 1., 'I expect terminal reward decaying, i.e. terminal_decay <= 1.0'
        self.terminal_decay = terminal_decay
        
        self.time_limit = time_limit
        
        self.pixel_size = pixel_size
        self.color_map = color_map

        self._pos = self.init_agent_pos
        self._obs = self._get_initial_observation()
        self._next_pos_cache = {}
        self.warp_size = warp_size

        shape = (1, *self.warp_size)
        if shape is None:
            shape = (1, self.size[0]*self.pixel_size, self.size[1]*self.pixel_size)

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=shape,
            dtype="uint8",
        )
        self.action_space = spaces.Discrete(4)

        self.reset()
    
    def reset(self):
        # do not forget to update observation
        #   1) regularly: old pos -> floor; init pos -> agent
        self._update_observation(self._pos, self.init_agent_pos)
        #   2) restore target if it was overwritten: terminal -> target
        self._fill(self.terminal, "GOAL")

        self._pos = self.init_agent_pos
        self._step = 0
        self._terminated = self._is_current_pos_terminal()

        obs = resize(self._obs, (42, 42))*255
        obs = np.expand_dims(obs, 0).astype(np.uint8)
        return obs
    
    def step(self, action, log=True):
        assert not self._terminated

        self._step += 1
        self._pos = self._move(action)

        # move before getting the reward
        reward = self._get_reward()
        
        # update terminated state after getting the reward
        terminated = self._is_current_pos_terminal()
        truncated = self.time_limit is not None and self._step >= self.time_limit
        self._terminated = terminated or truncated
        done = terminated or truncated

        obs = resize(self._obs, (42, 42))*255
        obs = np.expand_dims(obs, 0).astype(np.uint8)
        return obs, reward, done, dict(position=self._pos)
    
    def render_rgb(self):
        plt.imshow(self._obs)
        
    def _is_current_pos_terminal(self) -> bool:
        return self._pos == self.terminal
    
    def _get_reward(self) -> float:
        reward = 0
        if self._pos == self.terminal and not self._terminated:
            reward += self.terminal_decay ** self._step
        reward += self.const_punish
        return reward
        
    
    def _move(self, action) -> TPos:
        new_pos = self._next_pos_cache.get((self._pos, action), None)
        if new_pos is None:
            # call true dynamics to get new position
            new_pos, _ = MoveDynamics.try_move(
                position=self._pos,
                move_direction=MOVE_DIRECTIONS[DIRECTIONS_ORDER[action]],
                shape=self.size, obstacle_mask=self.obstacles
            )
            self._next_pos_cache[(self._pos, action)] = new_pos
        
        # update observation after move
        self._update_observation(self._pos, new_pos)
        return new_pos
    
    def _get_initial_observation(self) -> np.ndarray:
        # make observation map with pixel positions
        if not self.beautiful:
            obs = np.zeros(self.size)
            obs[self.obstacles == 0] = self.color_map['floor']
            obs[self.obstacles == 1] = self.color_map['obstacle']
            obs[self.terminal] = self.color_map['target']
            obs[self._pos] = self.color_map['agent']
        
        # then resize observation transforming each pixel to the square
            px = self.pixel_size

            obs = np.repeat(np.repeat(obs, px, axis=0), px, axis=1)

        # obs = rgb2gray(obs)
            return obs
        else:
            obs = np.zeros(np.array(self.size)*self.pixel_size).astype(np.uint8)
            for i in range(self.obstacles.shape[0]):
                for j in range(self.obstacles.shape[1]):
                    entity = self.obstacles[i][j]
                    i_px, j_px = i*self.pixel_size, j*self.pixel_size
                    obs[i_px: i_px+self.pixel_size, j_px: j_px+self.pixel_size] =\
                        self.render_tile(IDX_TO_OBJECT[entity])

            i, j = self._pos
            i, j = i*self.pixel_size, j*self.pixel_size
            obs[i: i + self.pixel_size, j: j + self.pixel_size] = self.render_tile("AGENT")

            i, j = self.terminal
            i, j = i * self.pixel_size, j * self.pixel_size
            obs[i: i + self.pixel_size, j: j + self.pixel_size] = self.render_tile("GOAL")
            return obs
    
    def _update_observation(self, pos, new_pos):
        # fill old pos with floor color, fill new pos with agent color
        floor, agent = "FLOOR", "AGENT"
        
        self._fill(pos, floor)
        self._fill(new_pos, agent)

    def _fill(self, pos, entity):
        # fill position with the color
        (i, j), px = pos, self.pixel_size
        i, j = i*px, j*px
        self._obs[i: i + px, j: j + px] = self.render_tile(entity)
        #print(self.render_tile(entity).dtype)
        #print(self._obs.dtype)


    def seed(self, seed: int) -> int:
        return seed

    def render_tile(self, entity):
        if entity in ["AGENT", "GOAL", "WALL"]:
            img = np.expand_dims(NAME_TO_COLOR[entity], (0, 1))
            img = np.tile(img, (self.pixel_size, self.pixel_size, 1))

            img = rgb2gray(img)
            return img

        elif entity == "FLOOR":
            img = np.zeros((96, 96, 3), dtype=np.uint8)
            img[:] = NAME_TO_COLOR["FLOOR"]//2
            img[0:2, :] = NAME_TO_COLOR["WALL"]
            img[:, 0:2] = NAME_TO_COLOR["WALL"]

            def downsample(img, pixel_size):
                """
                Downsample an image along both dimensions to pixel size
                """

                assert img.shape[0] % pixel_size == 0
                assert img.shape[1] % pixel_size == 0

                img = img.reshape(
                    [pixel_size, img.shape[0] // pixel_size, pixel_size, img.shape[1] // pixel_size, 3]
                )
                img = img.mean(axis=3)
                img = img.mean(axis=1)

                return img.astype(np.uint8)

            img = downsample(img, self.pixel_size)

            img = rgb2gray(img)
            return img
