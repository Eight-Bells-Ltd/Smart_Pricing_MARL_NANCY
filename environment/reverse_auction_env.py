import functools
import numpy as np
import matplotlib.pyplot as plt
from gymnasium.spaces import Discrete, Dict, Box
from pettingzoo import ParallelEnv
import os

import json


class ReverseAuctionEnv(ParallelEnv):
  
    metadata = {"render_modes": ["human"], "name": "reverse_auction_v1"}

    def __init__(self, render_mode="human", num_bidders=5, possible_agents={}, initial_bids={}, avg_min=30, max_rounds=20):
        self.possible_agents = possible_agents if initial_bids != {} else ["provider_" + str(r) for r in range(num_bidders)]
        self.agent_name_mapping = dict(zip(self.possible_agents, list(range(len(self.possible_agents)))))
        self.render_mode = render_mode
        self.max_rounds = max_rounds
        self.round = 1
        self.avg_min = avg_min
        self.done = False
        self.initial_bids = initial_bids if initial_bids != {} else np.random.randint(85, 100, size=num_bidders).astype(np.float64)       
        self.bids = self.initial_bids.copy()
        self.prev_ranks = np.ones(num_bidders, dtype=int)

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return Dict({
            'current_rank': Discrete(len(self.possible_agents) + 1),
            'previous_rank': Discrete(len(self.possible_agents) + 1),
            'current_round': Discrete(self.max_rounds + 1)
        })

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(9)  # 0->same bid, 1-4>higher bid, 5-8>lower bid

    
    def render(self):
        if self.render_mode == "human":
            num_agents = len(self.possible_agents)
            round_number = self.round
            plt.figure(figsize=(10, 6))
    
            for i in range(num_agents):
                agent_bid = self.bids[i]
                plt.plot([i + 1], [agent_bid], marker='o', label=f"{self.possible_agents[i]} Bid")
            
            plt.xlabel("Agents")
            plt.ylabel("Bid Value")
            plt.ylim(0, 100)  # Set y-axis boundaries
            plt.title(f"Bids of Each Agent in Round {round_number}")
            plt.legend()
            plt.grid(True)
            
            output_folder = "outputs/pngs"
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
    
            output_path = os.path.join(output_folder, f"round_{round_number}_bids.png")
            plt.savefig(output_path)
            plt.close()

            
        
    def close(self):

        min_index = np.argmin(self.bids)
        winner = self.possible_agents[min_index]
   

        data = {
            "winner": winner,
            "price": np.min(self.bids)
        }

        with open('auction_results.json', 'w') as json_file:
            json.dump(data, json_file, indent=4)

        print("Data successfully written to output.json") 

        pass

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.round = 1
        self.done = False
        self.initial_bids = np.random.randint(55, 60, size=len(self.possible_agents)).astype(np.float64)  
        self.bids = self.initial_bids.copy()
        self.prev_ranks = np.ones(len(self.agents), dtype=int)

        rank = 1

        observations = {agent: {'current_rank': rank, 'previous_rank': rank, 'current_round': self.round} for agent in self.agents}
        self.state = observations
        infos = {agent: {} for agent in self.agents}
        return observations, infos

    def step(self, actions):
        if not actions:
            #self.agents = []
            return {}, {}, {}, {}, {}

        if self.round == self.max_rounds:
            self.done = True

        rewards = {}
        observations = {}

        
        for agent in self.agents:
            action = actions[agent]
            agent_id = self.agent_name_mapping[agent]

            if action == 0:  # Same bid
                pass
            elif 1 <= action <= 4:  # Higher bid
                self.bids[agent_id] *= 1.1 + (action - 1) * 0.1
            elif 5 <= action <= 8:  # Lower bid
                self.bids[agent_id] *= 0.9 - (action - 5) * 0.1


        sorted_indices = np.argsort(self.bids)
        ranks = np.empty_like(sorted_indices)
        ranks[sorted_indices] = np.arange(len(sorted_indices)) + 1

        for agent in self.agents:
            agent_id = self.agent_name_mapping[agent]
            previous_rank = self.prev_ranks[agent_id]
            current_rank = ranks[agent_id]

            rewards[agent] = 0
            
            # Reward for improving rank
            if current_rank < previous_rank:
                rewards[agent] += 1  
            # Negative reward for worsening rank
            elif current_rank > previous_rank:
                rewards[agent] -= 1  

            
            # Negative reward for bid outside range
            if not (self.avg_min < self.bids[agent_id] < self.initial_bids[agent_id]):
                rewards[agent] -= 10  
            else:
                rewards[agent] += 2
                if self.round == self.max_rounds:
                    if current_rank == 1: rewards[agent] += 10
                    else: rewards[agent] -= 10
            
            observations[agent] = {
                'current_rank': current_rank,
                'previous_rank': previous_rank,
                'current_round': self.round
            }
            
            self.prev_ranks[agent_id] = current_rank


        self.state = observations

        infos = {agent: {} for agent in self.agents}
        terminations = {agent: self.done for agent in self.agents}
        truncations = {agent: self.done for agent in self.agents}

        if(self.done!=True): self.round += 1

        if self.done:
            self.agents = [] 


        return observations, rewards, terminations, truncations, infos
    



    # # Move the reward calculation to a separate method:
    # def calculate_reward(self, agent, current_rank, previous_rank):
    #     # Implement the reward calculation logic here
    #     pass