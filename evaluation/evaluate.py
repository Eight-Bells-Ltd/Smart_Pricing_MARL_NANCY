import ray
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.env import ParallelPettingZooEnv
import os
from ray.tune.registry import register_env
from agents.ppo_agent import ActionMaskedModel
from ray.rllib.models import ModelCatalog


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
    config = {
        "env": "reverse_auction",
        "model": {
            "custom_model": "action_masked_model",
        },
        "num_workers": 0,
        "evaluation_num_workers": 0,
        # "num_envs_per_worker":1/4,
        "num_gpus": 1,
        "create_env_on_driver": False,
        "explore": True,  # Use non-deterministic policy for evaluation
        "evaluation_interval": 1,
        "evaluation_duration": 1,
        "evaluation_duration_unit": "episodes",
        "evaluation_config": {
            "env_config": {"render_mode": render_mode},
            "explore": True,
            "entropy_coeff": 0.01
        }
    }

    # Load PPO from checkpoint with the above config
    algo = PPO(config=config, env="reverse_auction")
    algo.restore(model_path)

    import time
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
