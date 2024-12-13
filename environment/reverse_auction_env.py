import functools
import numpy as np
from gymnasium.spaces import Discrete, Dict, MultiBinary, Box
from pettingzoo import ParallelEnv

from utils.helpers import calculate_reward, update_bid, get_results, generate_action_mask
from render.render import render_env, render_final_plot


class ReverseAuctionEnv(ParallelEnv):
    metadata = {
        "name": "smart_pricing"
    }

    def __init__(self, render_mode="no", num_bidders=5, possible_agents=None, initial_prices=None, avg_min=30,
                 max_rounds=20):

        # self.num_bidders = num_bidders
        possible_agents=None
        self.num_bidders = np.random.randint(2, 10)

        self.possible_agents = possible_agents if possible_agents is not None else ["provider_" + str(r) for r in
                                                                                    range(self.num_bidders)]
        self.agent_name_mapping = dict(zip(self.possible_agents, list(range(len(self.possible_agents)))))
        self.render_mode = render_mode
        self.max_rounds = max_rounds
        self.round = 1
        self.avg_min = np.random.randint(20, 40, size=self.num_bidders)
        self.done = False
        self.initial_bids = np.random.randint(80, 100, size=self.num_bidders)
        self.bids = self.initial_bids.copy()
        # self.bids = 50 * np.ones_like(self.initial_bids)
        self.prev_ranks = np.ones(self.num_bidders, dtype=int)
        self.metadata["num_bidders"] = self.num_bidders

        # Initialize histories
        self.my_bid_history = {agent: np.zeros(self.max_rounds) for agent in self.possible_agents}
        self.lowest_bid_history = np.zeros(self.max_rounds)

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return Dict({
            'observations': Dict({
                'current_rank': Discrete(100 + 1),
                'previous_rank': Discrete(100 + 1),
                'remaining_rounds': Discrete(self.max_rounds + 1),
                'my_bid_history': Box(0, np.inf, shape=(self.max_rounds,)),
                'my_max': Box(0, np.inf),
                'my_min': Box(0, np.inf)
            }),
            'action_mask': MultiBinary(len(np.arange(1.99, 0.01, -0.01).round(2).tolist()))
        })

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(len(np.arange(1.99, 0.01, -0.01).round(2).tolist()))

    def render(self):
        if self.render_mode == "human":
            output_folder = "outputs/pngs"
            render_env(
                num_agents=len(self.possible_agents),
                round_number=self.round,
                bids=self.bids,
                agent_list=self.possible_agents,
                output_folder=output_folder
            )
            render_final_plot(self.possible_agents, "outputs/pngs/out.csv", output_folder)
            print(self.avg_min)

    def close(self):
        get_results(self.bids, self.possible_agents)

    def reset(self, seed=None, options=None):
        self.num_bidders = np.random.randint(2, 10)
        self.possible_agents = ["provider_" + str(r) for r in range(self.num_bidders)]
        self.agents = self.possible_agents[:]
        self.agent_name_mapping = dict(zip(self.possible_agents, list(range(len(self.possible_agents)))))
        self.round = 1
        self.avg_min = np.random.randint(20, 40, size=self.num_bidders)
        self.done = False
        self.initial_bids = np.random.randint(80, 100, size=self.num_bidders)
        self.bids = self.initial_bids.copy()
        # self.bids = 50* np.ones_like(self.initial_bids)
        self.prev_ranks = np.ones(len(self.agents), dtype=int)

        # Reset histories
        self.my_bid_history = {agent: np.zeros(self.max_rounds) for agent in self.possible_agents}
        self.lowest_bid_history = np.ones(self.max_rounds)* np.max(self.bids)

        # Calculate initial ranks
        sorted_indices = np.argsort(self.bids)
        ranks = np.empty_like(sorted_indices)
        ranks[sorted_indices] = np.arange(len(sorted_indices)) + 1

        # Initialize observations with histories
        observations = {
            agent: {
                'observations': {
                    'current_rank': ranks[self.agent_name_mapping[agent]],
                    'previous_rank': 1,
                    'remaining_rounds': self.max_rounds,
                    'my_bid_history': self.my_bid_history[agent],
                    'my_max': np.array([self.initial_bids[self.agent_name_mapping[agent]]], dtype=np.float32),
                    'my_min': np.array([self.avg_min[self.agent_name_mapping[agent]]], dtype=np.float32)
                },
                'action_mask': generate_action_mask(
                    self.bids[self.agent_name_mapping[agent]],
                    self.initial_bids[self.agent_name_mapping[agent]],
                    self.avg_min[self.agent_name_mapping[agent]]
                )
            } for agent in self.agents
        }

        infos = {agent: {} for agent in self.agents}
        return observations, infos

    def step(self, actions):
        if not actions:
            return {}, {}, {}, {}, {}

        if self.render_mode == "human":
            self.render()

        self.done = self.round == self.max_rounds

        # Update bids based on actions and record action history
        for agent, action in actions.items():
            agent_id = self.agent_name_mapping[agent]
            self.bids[agent_id] = update_bid(action, self.bids[agent_id])
            self.my_bid_history[agent][self.round - 1] = self.bids[agent_id]  # Update history

        # Update lowest bid history
        self.lowest_bid_history[self.round - 1] = np.min(self.bids)

        # Calculate new ranks
        sorted_indices = np.argsort(self.bids)
        ranks = np.empty_like(sorted_indices)
        ranks[sorted_indices] = np.arange(len(sorted_indices)) + 1

        # Compute rewards and observations
        rewards = {}
        observations = {}

        for agent, action in actions.items():
            agent_id = self.agent_name_mapping[agent]
            previous_rank = self.prev_ranks[agent_id]
            current_rank = ranks[agent_id]

            rewards[agent] = calculate_reward(
                current_rank, len(self.possible_agents), self.avg_min[agent_id], self.bids[agent_id],
                self.initial_bids[agent_id], self.round, self.max_rounds, action)
            # rewards[agent] = calculate_reward(
            #         current_rank, previous_rank, self.avg_min[agent_id], self.bids[agent_id],
            #         self.initial_bids[agent_id], self.round, self.max_rounds)

            observations[agent] = {
                'observations': {
                    'current_rank': current_rank,
                    'previous_rank': previous_rank,
                    'remaining_rounds': self.max_rounds - self.round,
                    'my_bid_history': self.my_bid_history[agent],
                    'my_max': np.array([self.initial_bids[agent_id]], dtype=np.float32),
                    'my_min': np.array([self.avg_min[agent_id]], dtype=np.float32)
                },
                'action_mask': generate_action_mask(
                    self.bids[agent_id],
                    self.initial_bids[agent_id],
                    self.avg_min[agent_id]
                )
            }

            self.prev_ranks[agent_id] = current_rank

        infos = {agent: {} for agent in self.agents}
        terminations = {agent: self.done for agent in self.agents}
        truncations = {agent: self.done for agent in self.agents}

        if not self.done:
            self.round += 1
        else:
            self.agents = []

        return observations, rewards, terminations, truncations, infos
