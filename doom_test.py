import gym
from vizdoom import gym_wrapper
from PIL import Image

if __name__ == "__main__":
    env = gym.make("VizdoomMyWayHome-v0", render_mode="human")
    obs, info = env.reset()
    im = Image.fromarray(obs["screen"])
    im.save("your_file.jpeg")
