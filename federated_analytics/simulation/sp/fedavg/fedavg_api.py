import logging
import random

import numpy as np

from federated_analytics.constants import FA_TASK_AVG
from federated_analytics.ml.trainer.trainer_creator import create_model_trainer
from .client import Client


""" todo: 
Mode 1: (online mode) each client stores its AVG result and the total number of data being sampled so far; 
later computation will use this result.
Mode 2: (offline mode, no need to use iterations) clients do not store previous results; 
server collects results from clients and does a weighted avg each round.
Finally, server does a weighted avg for all rounds.
Mode 3: (online mode, server does not need to store avg results for each rounds, the clients do not store their answers) 
similar to fl, the server sends the AVG result & total sample num so far to each client; 
(or, AVG result + cdp && a fake total sample num, the server can do further computation to get the real answer)
Mode 4: (online mode) server sets 2 local var: avg and total sample num. The server collects answers from clients each round and compute AVG
using avg, total sample num, and training num of the current round
"""
class FedAvgAPI(object):
    def __init__(self, args, dataset):
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
        self.model_trainer = create_model_trainer(FA_TASK_AVG, args)
        self._setup_clients(
            local_datasize_dict, train_data_local_dict, self.model_trainer,
        )
        self.w_global = 0.0
        self.total_sample_num = 0

    def _setup_clients(
        self, local_datasize_dict, train_data_local_dict, model_trainer,
    ):
        logging.info("############setup_clients (START)#############")
        for client_idx in range(self.args.client_num_per_round):
            c = Client(
                client_idx,
                train_data_local_dict[client_idx],
                local_datasize_dict[client_idx],
                self.args,
                model_trainer,
            )
            self.client_list.append(c)
        logging.info("############setup_clients (END)#############")

    def train(self):
        logging.info("self.model_trainer = {}".format(self.model_trainer))
        # w_global = self.model_trainer.get_model_params()
        local_sample_num = dict()
        for round_idx in range(self.args.comm_round):
            logging.info("################Communication round : {}".format(round_idx))
            w_locals = []

            """
            for scalability: following the original FedAvg algorithm, we uniformly sample a fraction of clients in each round.
            Instead of changing the 'Client' instances, our implementation keeps the 'Client' instances and then updates their local dataset 
            """
            client_indexes = self._client_sampling(
                round_idx, self.args.client_num_in_total, self.args.client_num_per_round
            )
            print(f"self.local_datasize_dict={self.local_datasize_dict}, local_sample_num={local_sample_num}")
            for i in client_indexes:
                local_sample_num[i] = random.randint(1, self.local_datasize_dict[i])

            # logging.info("client_indexes = " + str(client_indexes))
            for idx, client in enumerate(self.client_list):
                # update dataset
                client_idx = client_indexes[idx]
                client.update_local_dataset(
                    client_idx,
                    self.train_data_local_dict[client_idx],
                    local_sample_num[client_idx]
                )
                # train on new dataset
                w = client.train(w_global=None)
                w_locals.append((client.get_sample_number(), w))
            # update global weights
            self.w_global = self._aggregate(w_locals)
            print(f"round_idx={round_idx}, aggregation result = {self.w_global}")

    def _client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        if client_num_in_total == client_num_per_round:
            client_indexes = [client_index for client_index in range(client_num_in_total)]
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            np.random.seed(round_idx)  # make sure for each comparison, we are selecting the same clients each round
            client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
        logging.info("client_indexes = %s" % str(client_indexes))
        return client_indexes

    def _aggregate(self, w_locals):
        training_num = 0
        for idx in range(len(w_locals)):
            (sample_num, averaged_params) = w_locals[idx]
            training_num += sample_num

        (sample_num, averaged_params) = w_locals[0]
        for i in range(0, len(w_locals)):
            local_sample_number, local_model_params = w_locals[i]
            w = local_sample_number / training_num
            if i == 0:
                averaged_params = local_model_params * w
            else:
                averaged_params += local_model_params * w
        self.total_sample_num += training_num
        averaged_params = averaged_params * (training_num / self.total_sample_num) + self.w_global * ((self.total_sample_num - training_num)/self.total_sample_num)
        return averaged_params

    # def _aggregate_noniid_avg(self, w_locals):
    #     """
    #     The old aggregate method will impact the model performance when it comes to Non-IID setting
    #     Args:
    #         w_locals:
    #     Returns:
    #     """
    #     (_, averaged_params) = w_locals[0]
    #     for k in averaged_params.keys():
    #         temp_w = []
    #         for (_, local_w) in w_locals:
    #             temp_w.append(local_w[k])
    #         averaged_params[k] = sum(temp_w) / len(temp_w)
    #     return averaged_params
