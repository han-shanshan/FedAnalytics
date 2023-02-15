import logging
import math

from federated_analytics.FA_components.local_analyzer.client_analyzer_creator import create_local_analyzer
from federated_analytics.constants import FA_TASK_HEAVY_HITTER_TRIEHH
from .client import Client
from collections import defaultdict
import numpy as np
from ...utils import client_sampling

"""
Federated Heavy Hitters Discovery with Differential Privacy: https://arxiv.org/pdf/1902.08534.pdf
reference: https://github.com/google-research/federated/tree/master/triehh
"""
class TrieHHSimulator(object):
    def __init__(self, args=None, dataset=None):
        if hasattr(args, "max_word_len"):
            self.MAX_L = args.max_word_len
        else:
            self.MAX_L = 10
        if hasattr(args, "epsilon"):
            self.epsilon = args.epsilon
        else:
            self.epsilon = 1.0
        if hasattr(args, "delta"):
            self.delta = args.delta
        else:
            self.delta = 2.3e-12
        self.num_runs = args.comm_round

        self.round_counter = 1
        self.args = args
        [
            train_data_num,
            local_datasize_dict,
            train_data_local_dict,
        ] = dataset

        self.train_data_num_in_total = train_data_num
        self.client_list = []
        self.local_datasize_dict = local_datasize_dict
        self.train_data_local_dict = train_data_local_dict
        self.local_analyzer = create_local_analyzer(FA_TASK_HEAVY_HITTER_TRIEHH, args)
        self._setup_clients(
            local_datasize_dict, train_data_local_dict, self.local_analyzer,
        )
        self.w_global = {}  # self.trie = {}
        self.total_sample_num = 0

        self._set_theta()

        # batch size: the number of words in total that are sent to the server;
        # check Corollary 1 in the paper.
        # Done in _set_theta: We need to make sure theta >= np.e ** (self.epsilon/self.MAX_L) - 1
        self.batch_size = int(train_data_num * (np.e ** (self.epsilon / self.MAX_L) - 1) / (
                self.theta * np.e ** (self.epsilon / self.MAX_L)))
        print(f'Batch size used by TrieHH: {self.batch_size}')

    def _setup_clients(
            self, local_datasize_dict, train_data_local_dict, local_analyzer,
    ):
        logging.info("############setup_clients (START)#############")
        for client_idx in range(self.args.client_num_per_round):
            c = Client(
                client_idx,
                train_data_local_dict[client_idx],
                local_datasize_dict[client_idx],
                self.args,
                local_analyzer,
            )
            self.client_list.append(c)
        logging.info("############setup_clients (END)#############")

    def _set_theta(self):
        theta = 5  # initial guess
        delta_inverse = 1 / self.delta
        while ((theta - 3) / (theta - 2)) * math.factorial(theta) < delta_inverse:
            theta += 1
        while theta < np.e ** (self.epsilon / self.MAX_L) - 1:
            theta += 1
        self.theta = theta
        print(f'Theta used by TrieHH: {self.theta}')

    def train(self):
        logging.info("self.local_analyzer = {}".format(self.local_analyzer))
        local_sample_num = dict()
        for round_idx in range(self.args.comm_round):
            logging.info("################Communication round : {}".format(round_idx))
            w_locals = []
            client_indexes = client_sampling(
                round_idx, self.args.client_num_in_total, self.args.client_num_per_round
            )
            print(f"self.local_datasize_dict={self.local_datasize_dict}, local_sample_num={local_sample_num}")
            for i in client_indexes:
                local_sample_num[i] = math.ceil(self.batch_size / self.args.client_num_per_round)
            for idx, client in enumerate(self.client_list):
                client_idx = client_indexes[idx]
                client.update_local_dataset(
                    client_idx,
                    self.train_data_local_dict[client_idx],
                    local_sample_num[client_idx]
                )
                w = client.local_analyze(w_global=None)
                w_locals.extend(w)
            # update global weights
            self.w_global = self._aggregate(w_locals)
            self.print_heavy_hitters()
            print(f"round_idx={round_idx}, aggregation result = {self.w_global}")

    def _aggregate(self, w_locals):
        print(f"previous w_local_len={len(w_locals)}")
        if (len(w_locals) > self.batch_size):
            idxs = np.random.choice(range(len(w_locals)), self.batch_size, replace=False)
            w_locals = [w_locals[i] for i in idxs]
        print(f"len = {len(w_locals)}, w_locals={w_locals}")
        while True:
            votes = defaultdict(int)
            """notes from the author of the paper: 
            # I encourage you to think about how we could rewrite this function to do
            # one client update (i.e. return 1 vote from 1 chosen client).
            # Then you can have an outer for loop that iterates over chosen clients
            # and calls self.client_update() for each chosen and accumulates the votes."""

            for word in w_locals:
                vote_result = self.client_vote(word, self.round_counter)
                if vote_result > 0:
                    votes[word[0:self.round_counter]] += vote_result
            self.server_update(votes)
            self.round_counter += 1
            if self.quit_sign or self.round_counter > self.MAX_L:
                print("end of discovery") 
                break

    def client_vote(self, word, r):
        if len(word) < r:
            return 0
        pre = word[0:r - 1]
        # print(f"self.w_global={self.w_global}")
        # print(f"pre = {pre}, type={type(self.w_global)}")
        if self.w_global is None:
            return 1
        if pre and (pre not in self.w_global):
            return 0
        return 1

    def server_update(self, votes):
        # It might make more sense to define a small class called server_state
        # server_state can track 2 things: 1) updated trie, and 2) quit_sign
        # server_state can be initialized in the constructor of SimulateTrieHH
        # and server_update would just update server_state
        # (i.e, it would update self.server_state.trie & self.server_state.quit_sign)
        self.quit_sign = True
        for prefix in votes:
            if votes[prefix] >= self.theta:
                self.w_global[prefix] = None
                self.quit_sign = False

    def print_heavy_hitters(self):
        heavy_hitters = []
        print(f"self.w_global = {self.w_global}")
        # raw_result = self.w_global.keys()
        # results = []
        # for word in raw_result:
        #     if word[-1:] == '$':
        #         results.append(word.rstrip('$'))
        # print(f'Discovered {len(results)} heavy hitters in run #{self.round_counter + 1}')
        # print(results)
        # heavy_hitters.append(results)
