import logging
import random
from federated_analytics.constants import FA_TASK_UNION
from federated_analytics.ml.trainer.trainer_creator import create_model_trainer
from .client import Client
from ...utils import client_sampling


def get_union_of_two_lists_keep_duplicates(list1, list2):
    """
    Keep duplicates in the union, e.g., list1=[1,2,3,2,3], list2=[2,3,2,3]. intersect(list1, list2) = [1,2,3,2,3]
    :param list1: first list
    :param list2: second list
    :return: intersection of the 2 lists
    """
    union = []
    for item in list1:
        union.append(item)
        if item in list2:
            list2.remove(list2.index(item))
    union.extend(list2)
    return union


def get_union_of_two_lists_remove_duplicates(list1, list2):
    """
    Remove duplicates in the union, e.g., list1=[1,2,3,2,3], list2=[2,3,2,3]. intersect(list1, list2) = [1,2,3]
    :param list1: first list
    :param list2: second list
    :return: intersection of the 2 lists
    """
    return list(set(list1 + list2))


class UnionSimulator(object):
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

            # logging.info("client_indexes = " + str(client_indexes))
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
            print(f"round_idx={round_idx}, aggregation result = {self.w_global}")

    def _aggregate(self, w_locals):
        (sample_num, param) = w_locals[0]
        for i in range(0, len(w_locals)):
            _, local_model_params = w_locals[i]
            param = get_union_of_two_lists_remove_duplicates(param, local_model_params)
        return param
