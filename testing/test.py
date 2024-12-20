import ray
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.env import ParallelPettingZooEnv
import os
from ray.tune.registry import register_env
from agents.ppo_agent import ActionMaskedModel
from ray.rllib.models import ModelCatalog
import pandas as pd
import matplotlib.pyplot as plt
import glob

def analyze_auction_data(input_folder, output_folder, max_bidders=9, min_bidders=2):
    """
    Analyze all auction CSVs in a directory and gather statistics:
        1. Distribution of auction winners.
        2. Percentage of lowest starting price agent winning.
        3. Percentage of lowest min_limit_price agent winning.
        4. Percentage of agents bidding the lowest possible bid.
        5. Percentage of agents bidding the highest possible bid.
        6. Distribution of winning prices.
        7. Average percentage price difference per round.
        8. Average difference between winning bid and lowest allowed bid.
        9. Plot showing average difference between each agent's final bid and their allowed minimum.
    """
    os.makedirs(output_folder, exist_ok=True)
    all_files = [f for f in os.listdir(input_folder) if f.startswith("auction_") and f.endswith(".csv")]

    winner_counts = {}
    lowest_start_winner_count = 0
    lowest_min_limit_winner_count = 0
    lowest_bid_counts = 0
    highest_bid_counts = 0
    total_auctions = len(all_files)
    winning_prices = []
    round_percentage_diffs = []
    winning_bid_diffs = []
    participant_final_diffs = {f"provider_{i}": [] for i in range(9)}  # Track differences per participant
    rank_differences = []  # Track differences between rank 1 and rank 2

    for file in all_files:
        data = pd.read_csv(os.path.join(input_folder, file))

        # Find the final bids for each agent in the auction
        final_bids = data.groupby("Agent")["My_Bid"].last()
        sorted_bids = final_bids.sort_values()
        winner = sorted_bids.index[0]  # Agent with the lowest final bid
        winning_price = sorted_bids.iloc[0]
        winning_prices.append(winning_price)

        # Record difference between rank 1 and rank 2
        if len(sorted_bids) > 1:
            rank_differences.append(sorted_bids.iloc[1] - sorted_bids.iloc[0])

        # Find the lowest allowed bid among all participants
        lowest_allowed_bid = data.groupby("Agent")["My_Min"].first().min()
        winning_bid_diffs.append(winning_price - lowest_allowed_bid)

        # Update participant differences relative to their min allowed bids
        for agent in final_bids.index:
            min_allowed_bid = data[data["Agent"] == agent]["My_Min"].iloc[0]
            participant_final_diffs[agent].append(final_bids[agent] - min_allowed_bid)

        # Update winner counts
        winner_counts[winner] = winner_counts.get(winner, 0) + 1

        # Check if the agent with the lowest starting price was the winner
        start_bids = data.groupby("Agent")["My_Bid"].first()
        if start_bids.idxmin() == winner:
            lowest_start_winner_count += 1

        # Check if the agent with the lowest min_limit_bid was the winner
        min_limits = data.groupby("Agent")["My_Min"].first()
        if min_limits.idxmin() == winner:
            lowest_min_limit_winner_count += 1

        # Check if any agent bid the lowest or highest possible bids
        lowest_possible_bids = data.groupby("Agent")["My_Min"].first()
        highest_possible_bids = data.groupby("Agent")["My_Max"].first()

        for agent in final_bids.index:
            if data[data["Agent"] == agent]["My_Bid"].eq(lowest_possible_bids[agent]).any():
                lowest_bid_counts += 1
            if data[data["Agent"] == agent]["My_Bid"].eq(highest_possible_bids[agent]).any():
                highest_bid_counts += 1

        # Calculate average percentage difference per round
        round_diffs = []
        for round_num in range(1, int(data["Round"].max())):
            prev_round = data[data["Round"] == round_num]["My_Bid"].mean()
            curr_round = data[data["Round"] == round_num + 1]["My_Bid"].mean()
            if prev_round and curr_round:
                diff = ((curr_round - prev_round) / prev_round) * 100
                round_diffs.append(diff)
        round_percentage_diffs.append(round_diffs)

    # Aggregate average percentage differences across auctions
    avg_round_percentage_diffs = pd.DataFrame(round_percentage_diffs).mean(axis=0)

    # Calculate average winning bid difference
    avg_winning_bid_diff = sum(winning_bid_diffs) / len(winning_bid_diffs)

    # Calculate average rank difference (rank 1 vs rank 2)
    avg_rank_difference = sum(rank_differences) / len(rank_differences) if rank_differences else 0

    # Calculate average final bid differences per agent
    avg_final_bid_diffs = {agent: (sum(diffs) / len(diffs)) if diffs else 0 for agent, diffs in
                           participant_final_diffs.items()}

    # Save distribution of winners plot
    plt.figure()
    sorted_winner_counts = dict(sorted(winner_counts.items()))  # Sort by agent name
    plt.bar(sorted_winner_counts.keys(), sorted_winner_counts.values(), label="Actual Distribution")

    # Add theoretical distribution as a line plot
    average_num_bidders = (min_bidders + max_bidders) / 2
    providers = sorted(sorted_winner_counts.keys())
    participation_prob = [(max_bidders - int(agent.split("_")[1])) / (max_bidders - min_bidders + 1) for agent in
                          providers]
    expected_wins = [total_auctions * prob * (1 / average_num_bidders) for prob in participation_prob]
    plt.plot(providers, expected_wins, marker='o', linestyle='-', color='red', label="Theoretical Distribution")

    plt.xlabel("Agents")
    plt.ylabel("Number of Wins")
    plt.title("Distribution of Auction Winners")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "winner_distribution.png"))

    # Save winning price distribution plot
    plt.figure()
    plt.hist(winning_prices, bins=20, edgecolor='black')
    plt.xlabel("Winning Price")
    plt.ylabel("Frequency")
    plt.title("Distribution of Winning Prices")
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "winning_price_distribution.png"))

    # Save average percentage difference per round plot
    plt.figure()
    plt.bar(range(1, len(avg_round_percentage_diffs) + 1), avg_round_percentage_diffs)
    plt.xlabel("Round")
    plt.ylabel("Average Percentage Difference")
    plt.title("Average Percentage Price Difference Per Round")
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "avg_percentage_difference_per_round.png"))

    # Save average final bid difference plot per participant
    plt.figure()
    sorted_avg_diffs = dict(sorted(avg_final_bid_diffs.items()))
    plt.bar(sorted_avg_diffs.keys(), sorted_avg_diffs.values())
    plt.xlabel("Agents")
    plt.ylabel("Average Final Bid - Allowed Min Difference")
    plt.title("Average Final Bid Difference from Allowed Min")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "avg_final_bid_difference.png"))

    # Save statistics to a file
    stats = {
        "Total Auctions": total_auctions,
        "Average Winning Bid Difference": avg_winning_bid_diff,
        "Average Rank 1 vs Rank 2 Difference": avg_rank_difference,
        "Percentage of Lowest Starting Price Wins": (lowest_start_winner_count / total_auctions) * 100,
        "Percentage of Lowest Min Limit Price Wins": (lowest_min_limit_winner_count / total_auctions) * 100,
        "Percentage of Agents Bidding Lowest Possible Bid": (lowest_bid_counts / (
                    len(all_files) * len(winner_counts))) * 100, #FIXME mallon ine la8os
        "Percentage of Agents Bidding Highest Possible Bid": (highest_bid_counts / (
                    len(all_files) * len(winner_counts))) * 100 #FIXME mallon ine la8os
    }

    stats_file = os.path.join(output_folder, "auction_statistics.txt")
    with open(stats_file, "w") as f:
        for key, value in stats.items():
            f.write(f"{key}: {value:.2f}\n")

    print(f"Auction analysis completed. Results saved in {output_folder}.")
def test(env_fn, num_games: int = 100, render_mode: str | None = None, model_path='models/test', **env_kwargs):
    files = glob.glob('outputs/csvs/*csv')
    for f in files:
        os.remove(f)

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
        "evaluation_duration": num_games,
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
    print(f"Mean rewards over {num_games} games:", mean_rewards)
    algo.stop()

    analyze_auction_data('outputs/csvs', 'outputs')
