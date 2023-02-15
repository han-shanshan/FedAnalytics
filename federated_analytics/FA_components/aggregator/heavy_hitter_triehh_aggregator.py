import math
import numpy as np
from collections import defaultdict
from federated_analytics.core import ServerAggregator
from typing import List, Tuple, Any

"""
Federated Heavy Hitters Discovery with Differential Privacy: https://arxiv.org/pdf/1902.08534.pdf
reference: https://github.com/google-research/federated/tree/master/triehh
"""


class HeavyHitterTriehhAggregator(ServerAggregator):
    def __init__(self, args, train_data_num):
        super().__init__(args)
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
        self.w_global = {}  # self.trie = {}
        self.total_sample_num = 0
        self.quit_sign = False
        self.theta = self._set_theta()

        # batch size: the number of words in total that are sent to the server;
        # check Corollary 1 in the paper.
        # Done in _set_theta: We need to make sure theta >= np.e ** (self.epsilon/self.MAX_L) - 1
        self.batch_size = int(train_data_num * (np.e ** (self.epsilon / self.MAX_L) - 1) / (
                self.theta * np.e ** (self.epsilon / self.MAX_L)))
        print(f'Batch size used by TrieHH: {self.batch_size}')

    def aggregate(self, local_submission_list: List[Tuple[float, Any]]):
        # w_locals = [local_submission for (num, local_submission) in local_submission_list]
        w_locals = []
        # print(f"local_submission_list[0]={local_submission_list[0]}")

        for (num, local_submission) in local_submission_list:
            w_locals.extend(local_submission)
        if (len(w_locals) > self.batch_size):
            idxs = np.random.choice(range(len(w_locals)), self.batch_size, replace=False)
            w_locals = [w_locals[i] for i in idxs]

        while True:
            votes = defaultdict(int)
            """notes from the author of the paper: 
            # I encourage you to think about how we could rewrite this function to do
            # one client update (i.e. return 1 vote from 1 chosen client).
            # Then you can have an outer for loop that iterates over chosen clients
            # and calls self.client_update() for each chosen and accumulates the votes."""

            for word in w_locals:
                vote_result = self._client_vote(word, self.round_counter)
                if vote_result > 0:
                    votes[word[0:self.round_counter]] += vote_result
            self.server_update(votes)
            self.round_counter += 1
            if self.quit_sign or self.round_counter > self.MAX_L:
                print("end of discovery")
                break
        self.print_heavy_hitters()

    def _set_theta(self):
        theta = 5  # initial guess
        delta_inverse = 1 / self.delta
        while ((theta - 3) / (theta - 2)) * math.factorial(theta) < delta_inverse:
            theta += 1
        while theta < np.e ** (self.epsilon / self.MAX_L) - 1:
            theta += 1
        print(f'Theta used by TrieHH: {theta}')
        return theta

    def _client_vote(self, word, r):
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
