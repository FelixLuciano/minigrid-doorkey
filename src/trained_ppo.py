import gymnasium as gym
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3 import PPO

model = PPO.load("./src/results/ppo_minigrid")

env = gym.make("MiniGrid-DoorKey-8x8-v0", render_mode="human")
env = ImgObsWrapper(env)


(obs,_) = env.reset()
for i in range(1000):
    action, _state = model.predict(obs)
    obs, reward, done, truncated, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()