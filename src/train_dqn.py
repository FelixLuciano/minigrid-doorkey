from minigrid.wrappers import ImgObsWrapper, PositionBonus
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure
from MinigridFeaturesExtractor import MinigridFeaturesExtractor
import gymnasium as gym

from CustomRewardWrapper import CustomRewardWrapper

for i in range(5):
    results_path = f"./results/dqn/{i}"

    new_logger = configure(results_path, ["stdout", "csv", "tensorboard"])

    policy_kwargs = dict(
        features_extractor_class=MinigridFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=128),
    )

    env = gym.make("MiniGrid-DoorKey-6x6-v0", render_mode="rgb_array")
    env = ImgObsWrapper(env)
    env = CustomRewardWrapper(env)
    env = PositionBonus(env)

    model = DQN("CnnPolicy", 
                env, 
                policy_kwargs=policy_kwargs, 
                verbose=0,
                exploration_fraction=0.001,
                exploration_final_eps=0.5)

    model.set_logger(new_logger)
    model.learn(500_000, progress_bar=True)
    model.save(results_path + "/dqn_minigrid")


    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
    print(f'Mean reward: {mean_reward} +/- {std_reward:.2f}')
