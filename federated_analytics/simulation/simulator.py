from ..constants import FA_TASK_AVG
from ..core import ClientTrainer, ServerAggregator


class SimulatorSingleProcess:
    def __init__(self, args, dataset, client_trainer: ClientTrainer = None, server_aggregator: ServerAggregator = None,):
        from .sp.fedavg import FedAvgAPI
        if args.task == FA_TASK_AVG:
            self.fl_trainer = FedAvgAPI(args, dataset)
        else:
            raise Exception("Exception")

    def run(self):
        self.fl_trainer.train()

