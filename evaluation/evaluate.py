import os
import time
import threading
from ray.rllib.env import ParallelPettingZooEnv
from ray.tune.registry import register_env
from agents.ppo_agent import ActionMaskedModel
from ray.rllib.models import ModelCatalog
from ray.rllib.algorithms.ppo import PPOConfig

previous_auction_winner = {}
LOADED_ALGO = None
_LAST_FP = None
_LOCK = threading.Lock()

def _fingerprint(model_path, render_mode, env_kwargs):
    # Anything that affects the agent set or env config
    return (
        os.path.abspath(model_path),
        render_mode,
        tuple(sorted(env_kwargs.get("possible_agents", []))),
        env_kwargs.get("num_bidders"),
        env_kwargs.get("max_rounds"),
        env_kwargs.get("initial_prices"),
        env_kwargs.get("min_limit_bid"),
        env_kwargs.get("max_limit_bid"),
    )

def reset_algo_if_loaded():
    global LOADED_ALGO, _LAST_FP
    with _LOCK:
        if LOADED_ALGO is not None:
            try:
                LOADED_ALGO.stop()
            except Exception:
                pass
        LOADED_ALGO = None
        _LAST_FP = None

def evaluate(env_fn, render_mode: str | None = None, model_path='models/test', **env_kwargs):
    global LOADED_ALGO, _LAST_FP
    fp = _fingerprint(model_path, render_mode, env_kwargs)

    with _LOCK:
        if LOADED_ALGO is None or _LAST_FP != fp:
            start_time = time.perf_counter()

            def env_creator(config):
                merged = {**config, **env_kwargs}
                merged.pop("render_mode", None)
                return ParallelPettingZooEnv(env_fn(render_mode=render_mode, **merged))

            ModelCatalog.register_custom_model("action_masked_model", ActionMaskedModel)
            register_env("reverse_auction", env_creator)

            algo = (
                PPOConfig()
                .environment("reverse_auction")
                .framework("torch")
                .rollouts(num_env_runners=0)
                .training(model={"custom_model": "action_masked_model"}, entropy_coeff=0.01)
                .evaluation(
                    evaluation_interval=1,
                    evaluation_duration=1,
                    evaluation_duration_unit="episodes",
                    evaluation_config={"explore": True},
                )
                .resources(num_gpus=0)
                .build()
            )

            algo.restore(os.path.abspath(model_path))
            LOADED_ALGO = algo
            _LAST_FP = fp

            end_time = time.perf_counter()
            print(f"env & loading : {end_time - start_time:.6f} seconds")

    start_time = time.perf_counter()
    results = LOADED_ALGO.evaluate()
    end_time = time.perf_counter()
    print(f"running: {end_time - start_time:.6f} seconds")

    return previous_auction_winner
