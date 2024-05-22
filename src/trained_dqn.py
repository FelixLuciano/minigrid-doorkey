import gymnasium as gym
from minigrid.wrappers import ImgObsWrapper, PositionBonus
from stable_baselines3 import DQN

from CustomRewardWrapper import CustomRewardWrapper


model = DQN.load("./results/dqn/dqn_minigrid")

env = gym.make("MiniGrid-DoorKey-6x6-v0", render_mode="human")
env = ImgObsWrapper(env)
env = CustomRewardWrapper(env)
env = PositionBonus(env)

(obs,_) = env.reset()

for i in range(1000):
    action, _state = model.predict(obs)
    obs, reward, done, truncated, info = env.step(action)

    env.render()

    if done:
        obs = env.reset()