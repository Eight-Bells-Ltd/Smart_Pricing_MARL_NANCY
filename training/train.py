from tqdm import tqdm
import matplotlib.pyplot as plt

from ray.rllib.env import ParallelPettingZooEnv
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.policy.policy import PolicySpec
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog


from ray.rllib.utils.framework import try_import_torch
torch, nn = try_import_torch()
torch.backends.cudnn.benchmark = True
use_gpu = False#torch.cuda.is_available()
print("Torch is using CUDA: ", torch.cuda.is_available())
print("Torch device count: ", torch.cuda.device_count())

from agents.ppo_agent import ActionMaskedModel

def train(env_fn, steps: int = 10_000, learning_rate=1e-3, batch_size=256, model_path='models/test', **env_kwargs):

    def env_creator(_):
        env = env_fn(**env_kwargs)
        env.reset(seed=None)
        return ParallelPettingZooEnv(env)

    register_env("reverse_auction", env_creator)
    ModelCatalog.register_custom_model("action_masked_model", ActionMaskedModel)

    def policy_mapping_fn(agent_id, episode, **kwargs):
        return "shared_policy"

    # Configure the PPO algorithm
    config = (
        PPOConfig()
        .environment("reverse_auction")
        .framework("torch")
        .rollouts(num_env_runners=6, num_envs_per_worker=1, rollout_fragment_length='auto')
        # .evaluation(
        #         evaluation_interval= 50,
        #         evaluation_duration= 1,#irrelevant
        #         evaluation_duration_unit= "episodes",#irrelevant
        #         evaluation_config= {
        #             "env_config": {"render_mode": 'evaluate'},
        #             "explore": True,
        #             "entropy_coeff": 0.01
        #         }
        # )
        .training(
            train_batch_size=batch_size,
            minibatch_size=batch_size,
            lr=learning_rate,
            vf_clip_param=1000,
            model={
                "custom_model": "action_masked_model",
                "fcnet_hiddens": [32, 32],
                "fcnet_activation": "relu",
                "vf_share_layers": True
            },
            entropy_coeff_schedule=[
                [0, 0.9],  # Start with high exploration
                [steps//2 * batch_size, 0.1],
                [steps * batch_size, 0.001]
            ]
        )
        .multi_agent(
            policies={
                "shared_policy": PolicySpec(policy_class=None)
            },
            policy_mapping_fn=policy_mapping_fn,
        )
        .resources(num_gpus=0, num_gpus_per_worker=0)

    )

    trainer = PPO(config=config)

    # Initialize an empty list to store reward means
    reward_means = []
    reward_max = []
    reward_min = []
    with tqdm(total=steps) as pbar:
        for i in range(steps):
            # Perform a training iteration and capture the result
            result = trainer.train()

            # Extract useful statistics
            eval_metrics = result.get("env_runners", {})
            policy_reward_mean = eval_metrics.get("policy_reward_mean", float('nan'))['shared_policy']

            # Append to the list
            reward_means.append(policy_reward_mean)
            reward_max.append(eval_metrics.get("policy_reward_max", float('nan'))['shared_policy'])
            reward_min.append(eval_metrics.get("policy_reward_min", float('nan'))['shared_policy'])

            # Update tqdm description with the latest reward mean
            pbar.set_description(f"policy_reward_mean: {policy_reward_mean:.2f}")
            pbar.update(1)  # Increment progress bar

            # Save the plot every 50 steps
            if (i + 1) % 50 == 0:
                # eval_results = trainer.evaluate()
                # print(f"Evaluation results at step {i + 1}:", eval_results)

                plt.figure(figsize=(10, 5))
                plt.plot(reward_means, label="policy_reward_mean", color="blue")
                plt.fill_between(
                    range(len(reward_means)),
                    reward_min,
                    reward_max,
                    color="blue",
                    alpha=0.1,
                    label="Reward Range (Min-Max)"
                )

                # Labels, title, and legend
                plt.xlabel("Training Steps")
                plt.ylabel("Episode Reward")
                plt.title("Training Rewards")
                plt.legend(loc="best")
                plt.grid(True)

                # Save the plot with a unique filename for each interval
                plt.savefig(f"./outputs/training_reward.png")
                plt.close()  # Close the plot to free up memory

                trainer.save(model_path+f'{i+1}')
                print(f"Policy saved at: {model_path}")

    # model_filename = f"{time.strftime('%Y%m%d-%H%M%S')}"
    # model_path = os.path.join('models', model_filename)
    info = trainer.save(model_path)
    print(f"Policy saved at: {model_path}")
    print(info)

    print(f"Finished training...")
