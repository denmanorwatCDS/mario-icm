from PIL import Image
from math import pi
import gymnasium as gym
from MuJoCo_custom.EgocentricCylinderMaze import EgocentricCylinderMazeEnv
from MuJoCo_custom.EgocentricAntMaze import EgocentricAntMazeEnv
env = EgocentricCylinderMazeEnv(render_mode="rgb_array")
#env = gym.make('PointMaze_UMazeDense-v3', render_mode="rgb_array")
env.reset(seed=1)
A = env.render()
im = Image.fromarray(A)
im.save("your_file_0.jpeg")
env.step([1, pi/9])
for i in range(3):
    env.step([1, 0])
    env.step([1, 0])
A = env.render()
im = Image.fromarray(A)
im.save("your_file_1.jpeg")
env.step([1, -2*pi/9])
for i in range(3):
    env.step([1, 0])
    env.step([1, 0])
env.step([0, pi/9])
A = env.render()
im = Image.fromarray(A)
im.save("your_file_2.jpeg")
env.step([1, pi/9])
for i in range(5):
    env.step([1, 0])
    env.step([1, 0])
A = env.render()
im = Image.fromarray(A)
im.save("your_file_3.jpeg")