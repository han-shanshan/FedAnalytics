import logging
import random

from federated_analytics.constants import FA_TASK_K_PERCENTILE_ELEMENT
from federated_analytics.ml.trainer.trainer_creator import create_model_trainer
from .client import Client
from ...utils import client_sampling


class KPercentileElementSimulator(object):
    def __init__(self, args=None, dataset=None):
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
        self.model_trainer = create_model_trainer(FA_TASK_K_PERCENTILE_ELEMENT, args)
        self._setup_clients(
            local_datasize_dict, train_data_local_dict, self.model_trainer,
        )
        self.w_global = 100
        self.quit = False
        self.total_sample_num = 0
        self.k_percentage_numbers = int(self.train_data_num_in_total * args.k / 100)
        # self.flag = 100
        self.previous_w_global = 100
        if hasattr(args, "use_all_data") and args.use_all_data in [False]:
            self.use_all_data = False  # in each iteration, each client randomly sample some data to compute
        else:
            self.use_all_data = True  # in each iteration, each client uses its all local data to compute


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
            client_indexes = client_sampling(
                round_idx, self.args.client_num_in_total, self.args.client_num_per_round
            )

            if self.use_all_data:
                local_sample_num = self.local_datasize_dict
            else:
                for i in client_indexes:
                    local_sample_num[i] = random.randint(1, self.local_datasize_dict[i])

            for idx, client in enumerate(self.client_list):
                client_idx = client_indexes[idx]
                client.update_local_dataset(
                    client_idx,
                    self.train_data_local_dict[client_idx],
                    self.local_datasize_dict[client_idx]
                )
                w = client.train(w_global=self.w_global)
                w_locals.append((local_sample_num[client_idx], w))
            self.w_global = self._aggregate(w_locals)
            print(f"w_locals={w_locals}")
            print(f"round_idx={round_idx}, k_percentage_element = {self.w_global}")

    def _aggregate(self, w_locals):
        if self.quit:
            return self.w_global
        total_sample_num_this_round = 0
        local_satisfied_data_num_current_round = 0
        print(f"flag={self.w_global}, w_locals={w_locals}")
        for (sample_num, w_local) in w_locals:
            total_sample_num_this_round += sample_num
            local_satisfied_data_num_current_round += w_local
        if total_sample_num_this_round == int(self.train_data_num_in_total * local_satisfied_data_num_current_round / self.k_percentage_numbers):
            self.quit = True
            self.previous_w_global = self.w_global
        elif total_sample_num_this_round > int(self.train_data_num_in_total * local_satisfied_data_num_current_round / self.k_percentage_numbers):
            # reduce w_global
            if self.previous_w_global >= self.w_global:
                self.previous_w_global = self.w_global
                self.w_global = int(self.w_global/2)
            else:
                new_w_global = int((self.previous_w_global + self.w_global)/2)
                self.previous_w_global = self.w_global
                self.w_global = new_w_global
        else:  # increase w_global
            if self.previous_w_global <= self.w_global:
                self.previous_w_global = self.w_global
                self.w_global = int(2 * self.w_global)
            else:
                new_w_global = int((self.previous_w_global + self.w_global) / 2)
                self.previous_w_global = self.w_global
                self.w_global = new_w_global
        return self.w_global


