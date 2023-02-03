import logging
import random
from federated_analytics.constants import FA_TASK_UNION
from federated_analytics.ml.trainer.trainer_creator import create_model_trainer
from .client import Client
from ...utils import client_sampling


class IntersectionSimulator(object):
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
        self.model_trainer = create_model_trainer(FA_TASK_UNION, args)
        self._setup_clients(
            local_datasize_dict, train_data_local_dict, self.model_trainer,
        )
        self.w_global = []
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
        local_sample_num = dict()
        for round_idx in range(self.args.comm_round):
            logging.info("################Communication round : {}".format(round_idx))
            w_locals = []

            """
            for scalability: following the original FedAvg algorithm, we uniformly sample a fraction of clients in each round.
            Instead of changing the 'Client' instances, our implementation keeps the 'Client' instances and then updates their local dataset 
            """
            client_indexes = client_sampling(
                round_idx, self.args.client_num_in_total, self.args.client_num_per_round
            )
            print(f"self.local_datasize_dict={self.local_datasize_dict}, local_sample_num={local_sample_num}")
            for i in client_indexes:
                local_sample_num[i] = random.randint(1, self.local_datasize_dict[i])

            for idx, client in enumerate(self.client_list):
                # update dataset
                client_idx = client_indexes[idx]
                client.update_local_dataset(
                    client_idx,
                    self.train_data_local_dict[client_idx],
                    local_sample_num[client_idx]
                )
                w = client.train(w_global=None)
                w_locals.append((client.get_sample_number(), w))
            self.w_global = self._aggregate(w_locals)
            print(f"round_idx={round_idx}, aggregation result = {self.w_global}, cardinality = {self.get_cardinality()}")

    def _aggregate(self, w_locals):
        (sample_num, param) = w_locals[0]
        for i in range(0, len(w_locals)):
            _, local_model_params = w_locals[i]
            param = list(set(param) & set(local_model_params))
        return param

    def get_cardinality(self):
        return len(self.w_global)
