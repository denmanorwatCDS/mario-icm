from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from Environment_wrappers.wrappers import ResizeAndGrayscale, IntrinsicWrapper
from torch import optim
from ICM.ICM import ICM
from ICM.ICM_buffer import ICMBuffer
from Agents.neural_network import ActorCritic
from stable_baselines3.a2c.a2c import A2C
from Logger.WandB_logger import WandBLogger

from Config.environment_config import ALL_ACTION_SPACE, RESIZED_SIZE,\
    TEMPORAL_CHANNELS, DEVICE, ACTION_NAMES
from Config.ICM_config import STATE_SPACE_DIM, ETA
from Config.all_hyperparams_dict import HYPERPARAMS

wandb_logger = WandBLogger(HYPERPARAMS, "A2C_free", ALL_ACTION_SPACE, ACTION_NAMES)

env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, ALL_ACTION_SPACE)
action_dim = env.action_space.n
icm = ICM(action_dim, state_dim = STATE_SPACE_DIM,
          temporal_channels = TEMPORAL_CHANNELS, eta = ETA).to(DEVICE).train()
icm_optimizer = optim.Adam(icm.parameters())
icm_buffer = ICMBuffer(32, 512)

env = MaxAndSkipEnv(env, skip=6)
env = ResizeAndGrayscale(env, RESIZED_SIZE, TEMPORAL_CHANNELS)
env = IntrinsicWrapper(env, icm, icm_optimizer, icm_buffer, True)

policy_kwargs = dict(
    features_extractor_class=ActorCritic,
    net_arch=[dict(pi=[256], vf=[256])]
)

model = A2C("CnnPolicy", env, verbose=1, 
            policy_kwargs=policy_kwargs,)
model.learn(total_timesteps=10_000_000)

obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()

env.close()
