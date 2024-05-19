from minigrid.wrappers import ImgObsWrapper, ActionBonus
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from MinigridFeaturesExtractor import MinigridFeaturesExtractor
import gymnasium as gym

policy_kwargs = dict(
    features_extractor_class=MinigridFeaturesExtractor,
    features_extractor_kwargs=dict(features_dim=128),
)

env = gym.make("MiniGrid-DoorKey-8x8-v0", render_mode="rgb_array")
env = ImgObsWrapper(env)
env = ActionBonus(env)

model = DQN("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
model.learn(50_000)
model.save("./src/results/dqn_minigrid")


mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
print(f'Mean reward: {mean_reward} +/- {std_reward:.2f}')
