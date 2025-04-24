import numpy as np
import json

# multipliers = np.array([1.1, 1.2, 1.3, 1.0, 0.9, 0.8, 0.7])
multipliers = np.arange(1.99, 0.01, -0.01).round(2)

def get_results(bids, agent_list):
    # Determine the winner and winning price
    winner_data = {"winner": agent_list[np.argmin(bids)], "price": int(np.min(bids))}

    # Write results to a JSON file
    with open('./outputs/auction_results.json', 'w') as json_file:
        json.dump(winner_data, json_file, indent=4)

    print("Data successfully written to auction_results.json")

def get_results2(bids, agent_list):
    # Determine the winner and winning price
    winner_data = {"winner": agent_list[np.argmin(bids)], "price": int(np.min(bids))}

    return winner_data

def update_bid(action, bid):
    return bid * multipliers[action]

def calculate_reward(current_rank, num_agents, min_limit_bid, curr_bid, max_limit_bid, round, max_rounds, action):
    reward = 0

    proximity_to_max_bid = (curr_bid - min_limit_bid) / (max_limit_bid - min_limit_bid) # Scales bid between 0 and 1
    # End-of-auction rewards/penalties
    # if round+1 == max_rounds and current_rank == 1:
    #     reward += 70 * proximity_to_max_bid
    #     reward += 30 * (1 - abs(multipliers[action] - 1))
        # reward += 100
    #     reward += 40 * proximity_to_max_bid
    #     reward += 20 * (1 - abs(multipliers[action] - 1))

    reward += 100 * proximity_to_max_bid # bid penalty for bids closer to min_limit_bid

    # reward += 20 *  (1 - abs(multipliers[action] - 1))

    normalized_rank = (num_agents - current_rank) / (num_agents - 1)
    reward *= normalized_rank
    reward *= round / max_rounds

    return reward


def generate_action_mask(curr_bid, max_bid, min_bid):
    if curr_bid > max_bid or curr_bid < min_bid:
        print(f"\n!! ERROR !! min={min_bid} curr_bid={round(curr_bid,2)} max={max_bid}\n")
        # pass

    # Calculate all potential new bids
    new_bids = curr_bid * multipliers
    #create a mask
    mask = (new_bids <= max_bid) & (new_bids >= min_bid)
    mask = mask.astype(np.int8)

    return mask


def load_balancing(min_bids, max_bids, availability):
    availability = [100-x for x in availability]
    # Get overall min and max of availability for normalization
    avail_min = min(availability)
    avail_max = max(availability)

    new_min_bids = []
    new_max_bids = []
    initial_bids = []

    for old_min, old_max, avail in zip(min_bids, max_bids, availability):
        # Normalize the availability
        if avail_max != avail_min:
            norm = (avail - avail_min) / (avail_max - avail_min)
        else:
            norm = 0  # If all values are the same, no adjustment

        # Adjust min bid based on normalized availability.
        # The factor here scales the maximum adjustment
        addition = norm * (old_max - old_min) * 0.2
        new_min = old_min + addition

        # If new min is higher than current max, adjust the max bid accordingly
        if new_min >= old_max:
            new_max = new_min + (old_max - old_min)
        else:
            new_max = old_max

        # The initial bid is the midpoint between new min and new max
        mid = (new_min + new_max) / 2

        new_min_bids.append(new_min)
        new_max_bids.append(new_max)
        initial_bids.append(mid)

    print(min_bids, max_bids, availability)
    print(new_min_bids, new_max_bids, initial_bids)

    return new_min_bids, new_max_bids, initial_bids

print(load_balancing([20,30,40],[90,100,80],[40,70,80]))