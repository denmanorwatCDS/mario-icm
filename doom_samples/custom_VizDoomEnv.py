from vizdoom.gym_wrapper.base_gym_env import VizdoomEnv
import vizdoom.vizdoom as vzd
import numpy as np
import warnings
import gym

LABEL_COLORS = (
    np.random.default_rng(42).uniform(25, 256, size=(256, 3)).astype(np.uint8)
)

class CustomVizDoomEnv(VizdoomEnv):
    def __init__(self, level, frame_skip=1,
        max_buttons_pressed=1, render_mode = None):
        """
        Base class for Gym interface for ViZDoom. Thanks to https://github.com/shakenes/vizdoomgym
        Child classes are defined in vizdoom_env_definitions.py,

        Arguments:
            level (str): path to the config file to load. Most settings should be set by this config file.
            frame_skip (int): how many frames should be advanced per action. 1 = take action on every frame. Default: 1.

        This environment forces window to be hidden. Use `render()` function to see the game.

        Observations are dictionaries with different amount of entries, depending on if depth/label buffers were
        enabled in the config file:
          "rgb"           = the RGB image (always available) in shape (HEIGHT, WIDTH, CHANNELS)
          "depth"         = the depth image in shape (HEIGHT, WIDTH), if enabled by the config file,
          "labels"        = the label image buffer in shape (HEIGHT, WIDTH), if enabled by the config file. For info on labels, access `env.state.labels` variable.
          "automap"       = the automap image buffer in shape (HEIGHT, WIDTH, CHANNELS), if enabled by the config file
          "gamevariables" = all game variables, in the order specified by the config file

        Action space is always a Discrete one, one choice for each button (only one button can be pressed down at a time).
        """
        self.frame_skip = frame_skip

        # init game
        self.game = vzd.DoomGame()
        self.game.load_config("/home/dvasilev/doom_icm/mario_icm/custom_my_way_home.cfg")
        self.game.set_window_visible(False)

        screen_format = self.game.get_screen_format()
        if screen_format != vzd.ScreenFormat.RGB24:
            warnings.warn(f"Detected screen format {screen_format.name}. Only RGB24 is supported in the Gym wrapper. Forcing RGB24.")
            self.game.set_screen_format(vzd.ScreenFormat.RGB24)

        self.game.init()
        self.state = None
        self.window_surface = None
        self.isopen = True

        self.depth = self.game.is_depth_buffer_enabled()
        self.labels = self.game.is_labels_buffer_enabled()
        self.automap = self.game.is_automap_buffer_enabled()

        allowed_buttons = []
        for button in self.game.get_available_buttons():
            if "DELTA" in button.name:
                warnings.warn(f"Removing button {button.name}. DELTA buttons are currently not supported in Gym wrapper. Use binary buttons instead.")
            else:
                allowed_buttons.append(button)
        self.game.set_available_buttons(allowed_buttons)
        self.action_space = gym.spaces.Discrete(len(allowed_buttons))

        # specify observation space(s)
        spaces = {
            "rgb": gym.spaces.Box(
                0,
                255,
                (
                    self.game.get_screen_height(),
                    self.game.get_screen_width(),
                    3,
                ),
                dtype=np.uint8,
            )
        }

        if self.depth:
            spaces["depth"] = gym.spaces.Box(
                0,
                255,
                (
                    self.game.get_screen_height(),
                    self.game.get_screen_width(),
                ),
                dtype=np.uint8,
            )

        if self.labels:
            spaces["labels"] = gym.spaces.Box(
                0,
                255,
                (
                    self.game.get_screen_height(),
                    self.game.get_screen_width(),
                ),
                dtype=np.uint8,
            )

        if self.automap:
            spaces["automap"] = gym.spaces.Box(
                0,
                255,
                (
                    self.game.get_screen_height(),
                    self.game.get_screen_width(),
                    # "automap" buffer uses same number of channels
                    # as the main screen buffer
                    3,
                ),
                dtype=np.uint8,
            )

        self.num_game_variables = self.game.get_available_game_variables_size()
        if self.num_game_variables > 0:
            spaces["gamevariables"] = gym.spaces.Box(
                np.finfo(np.float32).min,
                np.finfo(np.float32).max,
                (self.num_game_variables,),
                dtype=np.float32
            )

        self.observation_space = gym.spaces.Dict(spaces)