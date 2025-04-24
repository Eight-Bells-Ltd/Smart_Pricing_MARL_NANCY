import os
import time
from ray.rllib.env import ParallelPettingZooEnv
from ray.tune.registry import register_env
from agents.ppo_agent import ActionMaskedModel
from ray.rllib.models import ModelCatalog
from ray.rllib.algorithms.ppo import PPOConfig

previous_auction_winner = {}
LOADED_ALGO = None  # Global variable to hold the loaded algorithm


def evaluate(env_fn, render_mode: str | None = None, model_path='models/test', **env_kwargs):
    """
    Evaluates the model with the given environment initial conditions.

    On first call, the environment and model are created and loaded.
    On subsequent calls, the evaluation environments are reset with the new options.
    """
    global LOADED_ALGO

    if LOADED_ALGO is None:
        start_time = time.perf_counter()

        # Define an environment creator that merges RLlib's config with env_kwargs.
        def env_creator(config):
            # Merge the evaluation config with the initial conditions provided in env_kwargs.
            merged_config = {**env_kwargs, **config}
            # Remove "render_mode" if present to avoid duplicates.
            merged_config.pop("render_mode", None)
            env = env_fn(render_mode=render_mode, **merged_config)
            return ParallelPettingZooEnv(env)

        # Register the custom model and environment.
        ModelCatalog.register_custom_model("action_masked_model", ActionMaskedModel)
        register_env("reverse_auction", env_creator)

        # Convert the model_path to an absolute path.
        model_path = os.path.abspath(model_path)

        # Build the PPO algorithm configuration.
        algo = (
            PPOConfig()
            .environment("reverse_auction")
            .framework("torch")  # Or "tf" if you're using TensorFlow
            .rollouts(num_env_runners=0)
            .training(model={"custom_model": "action_masked_model"}, entropy_coeff=0.01)
            .evaluation(
                evaluation_interval=1,
                evaluation_duration=1,
                evaluation_duration_unit="episodes",
                evaluation_config={
                    "env_config": {"render_mode": render_mode, **env_kwargs},
                    "explore": True,
                },
            )
            .resources(num_gpus=0)
            .build()
        )

        # Restore the model from the specified path.
        algo.restore(model_path)
        LOADED_ALGO = algo

        end_time = time.perf_counter()
        print(f"env & loading : {end_time - start_time:.6f} seconds")
    else:
        # When the algorithm is already loaded, reset the evaluation environments
        # with the latest initial conditions via the options parameter.
        if LOADED_ALGO.eval_env_runner is not None:
            LOADED_ALGO.eval_env_runner.env.reset(options=env_kwargs)
            LOADED_ALGO.env_runner.env.reset(options=env_kwargs)
            LOADED_ALGO.env_runner_group.local_env_runner.env.reset(options=env_kwargs)
            LOADED_ALGO.eval_env_runner_group.local_env_runner.env.reset(options=env_kwargs)

    start_time = time.perf_counter()

    # Run evaluation (which now uses the reset environments with the updated conditions)
    results = LOADED_ALGO.evaluate()

    end_time = time.perf_counter()
    print(f"running: {end_time - start_time:.6f} seconds")

    # Return previous_auction_winner (ensure that your evaluation logic updates this global as expected)
    return previous_auction_winner
