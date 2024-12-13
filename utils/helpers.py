import numpy as np
import json
import math

# multipliers = [1.1, 1.2, 1.3, 1.0, 0.9, 0.8, 0.7]
multipliers = np.arange(1.99, 0.01, -0.01).round(2)

def get_results(bids, agent_list):
    # Determine the winner and winning price
    winner_data = {"winner": agent_list[np.argmin(bids)], "price": int(np.min(bids))}
    
    # Write results to a JSON file
    with open('auction_results.json', 'w') as json_file:
        json.dump(winner_data, json_file, indent=4)

    print("Data successfully written to auction_results.json")

def update_bid(action, bid):

    return bid * multipliers[action]

def calculate_reward(current_rank, num_agents, avg_min_bid, bid, initial_bid, round, max_rounds, action):
    reward = 0
    normalized_rank = (num_agents - current_rank) / (num_agents - 1)
    reward = 40 * normalized_rank # 100 when at rank 1, 0 at the worst rank

    # End-of-auction rewards/penalties
    if round == max_rounds and current_rank == 1:
        if current_rank == 1:
            reward += 60

    proximity_to_initial_bid = (bid - avg_min_bid) / (initial_bid - avg_min_bid) # Scales bid between 0 and 1
    reward *= proximity_to_initial_bid # bid penalty for bids closer to avg_min_bid

    bid_update_penalty = 1 - abs(1 - multipliers[action])  # Calculate the deviation from 1
    reward *= bid_update_penalty ** 2

    return reward

# def calculate_reward(current_rank, previous_rank, avg_min, bid, initial_bid, round, max_rounds):
#
#     reward = 20 if avg_min < bid < initial_bid else -80
#
#     if current_rank < previous_rank: reward += 10
#
#     # reward += 10 if current_rank < previous_rank else -10 if current_rank > previous_rank else 0
#
#     if round == max_rounds: reward += 5 * max_rounds if current_rank == 1 else -10 * max_rounds
#
#     return reward


def generate_action_mask(curr_bid, max_bid, min_bid):
    if curr_bid > max_bid or curr_bid < min_bid:
        print(f"!! ERROR !! min={min_bid} curr_bid={round(curr_bid,2)} max={max_bid}")
        # pass

    # Calculate all potential new bids
    new_bids = curr_bid * multipliers
    #create a mask
    mask = (new_bids <= max_bid) & (new_bids >= min_bid)
    mask = mask.astype(np.int8)

    return mask