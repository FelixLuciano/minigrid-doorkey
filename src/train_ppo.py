from minigrid.wrappers import ImgObsWrapper, ActionBonus
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure
from minigrid.wrappers import PositionBonus
from MinigridFeaturesExtractor import MinigridFeaturesExtractor
import gymnasium as gym

results_path = "./results/ppo"

new_logger = configure(results_path, ["stdout", "csv", "tensorboard"])

policy_kwargs = dict(
    features_extractor_class=MinigridFeaturesExtractor,
    features_extractor_kwargs=dict(features_dim=128),
)

env = gym.make("MiniGrid-DoorKey-8x8-v0", render_mode="rgb_array")
_ = env.reset(seed=42)
env = ImgObsWrapper(env)
env = ActionBonus(env)
env = PositionBonus(env)

model = PPO("MlpPolicy", 
            env=env, 
            policy_kwargs=policy_kwargs, 
            verbose=0,
            learning_rate=1e-3,
            n_steps=128,
            batch_size=64,
            gae_lambda=0.99,
            gamma=0.99,
            n_epochs=10,
            ent_coef=0.0,
            clip_range=0.2,
            )

model.set_logger(new_logger)
model.learn(1_000_000, progress_bar=True)
model.save(results_path + "/ppo_minigrid")


mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
print(f'Mean reward: {mean_reward} +/- {std_reward:.2f}')
