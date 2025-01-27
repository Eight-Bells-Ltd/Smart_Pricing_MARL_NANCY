import functools
import numpy as np
from gymnasium.spaces import Discrete, Dict, MultiBinary, Box
from pettingzoo import ParallelEnv

from utils.helpers import calculate_reward, update_bid, get_results, generate_action_mask
from render.render import render_env, render_final_plot, plot_all_rounds, save_data


class ReverseAuctionEnv(ParallelEnv):
    metadata = {
        "name": "smart_pricing"
    }

    def __init__(self, render_mode="no", num_bidders=5, possible_agents=None, initial_prices=None, min_limit_bid=30,
                 max_rounds=20):#TODO take initial_prices into account

        # self.num_bidders = num_bidders
        #TODO take num_bidders into account for evaluation
        self.max_rounds = max_rounds
        self.render_mode = render_mode
        self.reset()

    def reset(self, seed=None, options=None):
        self.num_bidders = np.random.randint(2, 10)
        self.possible_agents = ["provider_" + str(r) for r in range(self.num_bidders)] #FIXME how to fix possible agents without reitialising env?
        self.agents = self.possible_agents[:]
        self.agent_name_mapping = dict(zip(self.possible_agents, list(range(len(self.possible_agents)))))
        self.curr_round = 1
        self.min_limit_bid = np.random.randint(20, 40, size=self.num_bidders)#TODO take min_limit_bid into account
        # self.min_limit_bid[1] = 50
        self.done = False
        self.max_limit_bid = np.random.randint(80, 100, size=self.num_bidders)#TODO add max_limit_bid and tak it into account
        self.curr_bids = np.random.uniform(self.min_limit_bid, self.max_limit_bid)
        # self.curr_bids[1]=50
        self.prev_ranks = np.ones(len(self.agents), dtype=int)
        self.metadata["num_bidders"] = self.num_bidders

        # Reset histories
        self.my_bid_history = self.my_bid_history = {agent: np.array([self.curr_bids[self.agent_name_mapping[agent]], *np.zeros(self.max_rounds - 1)])
                                                     for agent in self.possible_agents}
        self.lowest_bid_history = np.ones(self.max_rounds) * np.min(self.curr_bids)

        # Calculate initial ranks
        sorted_indices = np.argsort(self.curr_bids)
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
                    # 'my_max': np.array([self.max_limit_bid[self.agent_name_mapping[agent]]], dtype=np.float32),
                    # 'my_min': np.array([self.min_limit_bid[self.agent_name_mapping[agent]]], dtype=np.float32)
                },
                'action_mask': generate_action_mask(
                    self.curr_bids[self.agent_name_mapping[agent]],
                    self.max_limit_bid[self.agent_name_mapping[agent]],
                    self.min_limit_bid[self.agent_name_mapping[agent]]
                )
            } for agent in self.agents
        }

        infos = {agent: {} for agent in self.agents}
        return observations, infos

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return Dict({
            'observations': Dict({
                'current_rank': Discrete(100 + 1),
                'previous_rank': Discrete(100 + 1),
                'remaining_rounds': Discrete(self.max_rounds + 1),
                'my_bid_history': Box(0, np.inf, shape=(self.max_rounds,)),
                # 'my_max': Box(0, np.inf),
                # 'my_min': Box(0, np.inf)
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
                round_number=self.curr_round,
                bids=self.curr_bids,
                agent_list=self.possible_agents,
                output_folder=output_folder
            )
            render_final_plot(self.possible_agents, "outputs/pngs/out.csv", output_folder)
            if self.done:
                plot_all_rounds( self.my_bid_history,"outputs",self.agent_name_mapping,self.min_limit_bid,self.max_limit_bid)
        elif self.render_mode == "evaluate" and self.done:
            print(self.curr_bids)
            print(self.possible_agents)
            get_results(self.curr_bids, self.possible_agents)
        elif self.render_mode == "test" and self.done:
            save_data(self.my_bid_history, "outputs/csvs", self.agent_name_mapping, self.min_limit_bid, self.max_limit_bid, auction_id=None)

    def close(self):
        #for some reason close is called twice. and always on a !self.done env state
        pass
        # if self.done:
        #     import os
        #     process_id = os.getpid()
        #     print(f"Current Process ID: {process_id}")
        #
        #     print(self.curr_bids)
        #     print(self.possible_agents)
        #     get_results(self.curr_bids, self.possible_agents)

    def step(self, actions):
        if not actions:
            return {}, {}, {}, {}, {}

        self.done = self.curr_round+1 == self.max_rounds

        # Update bids based on actions and record action history
        for agent, action in actions.items():
            agent_id = self.agent_name_mapping[agent]
            self.curr_bids[agent_id] = update_bid(action, self.curr_bids[agent_id])
            self.my_bid_history[agent][self.curr_round] = self.curr_bids[agent_id]  # Update history

        # Update lowest bid history
        self.lowest_bid_history[self.curr_round] = np.min(self.curr_bids)

        # Calculate new ranks
        sorted_indices = np.argsort(self.curr_bids)
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
                current_rank, len(self.possible_agents), self.min_limit_bid[agent_id], self.curr_bids[agent_id],
                self.max_limit_bid[agent_id], self.curr_round, self.max_rounds, action)
            # rewards[agent] = calculate_reward(
            #         current_rank, previous_rank, self.avg_min[agent_id], self.bids[agent_id],
            #         self.initial_bids[agent_id], self.round, self.max_rounds)

            observations[agent] = {
                'observations': {
                    'current_rank': current_rank,
                    'previous_rank': previous_rank,
                    'remaining_rounds': self.max_rounds - self.curr_round,
                    'my_bid_history': self.my_bid_history[agent],
                    # 'my_max': np.array([self.max_limit_bid[agent_id]], dtype=np.float32),
                    # 'my_min': np.array([self.min_limit_bid[agent_id]], dtype=np.float32)
                },
                'action_mask': generate_action_mask(
                    self.curr_bids[agent_id],
                    self.max_limit_bid[agent_id],
                    self.min_limit_bid[agent_id]
                )
            }

            self.prev_ranks[agent_id] = current_rank

        self.render()

        infos = {agent: {} for agent in self.agents}
        terminations = {agent: self.done for agent in self.agents}
        truncations = {agent: self.done for agent in self.agents}

        if not self.done:
            self.curr_round += 1
        else:
            self.agents = []

        return observations, rewards, terminations, truncations, infos
