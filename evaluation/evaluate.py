from ray.rllib.env import ParallelPettingZooEnv
import os
from ray.tune.registry import register_env
from agents.ppo_agent import ActionMaskedModel
from ray.rllib.models import ModelCatalog
from ray.rllib.algorithms.ppo import PPOConfig
import time


def evaluate(env_fn, render_mode: str | None = None, model_path='models/test', **env_kwargs):
    # Load the environment
    def env_creator(_):
        env = env_fn(render_mode=render_mode, **env_kwargs)
        return ParallelPettingZooEnv(env)

    ModelCatalog.register_custom_model("action_masked_model", ActionMaskedModel)
    register_env("reverse_auction", env_creator)

    # Set up the model path
    model_path = os.path.abspath(model_path)

    # Define a config for evaluation, enabling `create_env_on_driver`
    algo = (
        PPOConfig()
        .environment("reverse_auction")
        .framework("torch")  # Or "tf" depending on your setup
        .rollouts(num_env_runners=0)
        .training(model={"custom_model": "action_masked_model"}, entropy_coeff=0.01)
        .evaluation(
            evaluation_interval=1,
            evaluation_duration=1,
            evaluation_duration_unit="episodes",
            evaluation_config={"env_config": {"render_mode": render_mode}, "explore": True},
        )
        .resources(num_gpus=0)
        .build()
    )
    algo.restore(model_path)

    start_time = time.perf_counter()

    # Run evaluation directly
    results = algo.evaluate()

    # Extract mean reward
    mean_rewards = results["env_runners"]["policy_reward_mean"]

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.6f} seconds")

    # Print and return results
    print("Evaluation Results:")
    print(f"Mean rewards over {1} games:", mean_rewards)
    algo.stop()
