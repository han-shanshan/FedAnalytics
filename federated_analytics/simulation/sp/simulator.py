import logging
import random

from federated_analytics.simulation.sp.client import Client
from federated_analytics.FA_components.aggregator.global_analyzer_creator import create_global_analyzer
from federated_analytics.FA_components.local_analyzer.client_analyzer_creator import create_local_analyzer
from federated_analytics.simulation.utils import client_sampling


class SimulatorSingleProcess:
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
        self.local_analyzer = create_local_analyzer(args)
        self.aggregator = create_global_analyzer(args)
        self._setup_clients(
            local_datasize_dict, train_data_local_dict, self.local_analyzer,
        )

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

    def analyze(self):
        logging.info("self.local_analyzer = {}".format(self.local_analyzer))
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
                w = client.local_analyze(w_global=None)
                w_locals.append((client.get_sample_number(), w))
            result = self.aggregator.aggregate(w_locals)
            print(f"round_idx={round_idx}, aggregation result = {result}")

    def run(self):
        self.analyze()

        # def __init__(self, args, dataset):
        #     from .sp.fedavg import FedAvgSimulator
        #     if args.task == FA_TASK_AVG:
        #         self.fl_trainer = FedAvgSimulator(args, dataset)
        #     elif args.task == FA_TASK_HEAVY_HITTER_TRIEHH:
        #         self.fl_trainer = TrieHHSimulator(args, dataset)
        #     elif args.task == FA_TASK_UNION:
        #         self.fl_trainer = UnionSimulator(args, dataset)
        #     elif args.task == FA_TASK_K_PERCENTILE_ELEMENT:
        #         self.fl_trainer = KPercentileElementSimulator(args, dataset)
        #     elif args.task == FA_TASK_INTERSECTION or args.task == FA_TASK_CARDINALITY:
        #         self.fl_trainer = IntersectionSimulator(args, dataset)
        #     elif args.task == FA_TASK_FREQ or args.task == FA_TASK_HISTOGRAM:
        #         self.fl_trainer = FrequencyEstimationSimulator(args, dataset)
        #     else:
        #         raise Exception("Exception")
