from ..constants import (
    FedML_FEDERATED_OPTIMIZER_BASE_FRAMEWORK,
    FedML_FEDERATED_OPTIMIZER_FEDAVG,
)
from ..core import ClientTrainer, ServerAggregator


class SimulatorSingleProcess:
    def __init__(self, args, device, dataset, task, client_trainer=None, server_aggregator=None):
        from .sp.fedavg import FedAvgAPI
        if args.federated_optimizer == FedML_FEDERATED_OPTIMIZER_FEDAVG:
            self.fl_trainer = FedAvgAPI(args, device, dataset, task)
        else:
            raise Exception("Exception")

    def run(self):
        self.fl_trainer.train()


class SimulatorMPI:
    def __init__(
        self,
        args,
        device,
        dataset,
        model,
        client_trainer: ClientTrainer = None,
        server_aggregator: ServerAggregator = None,
    ):
        from .mpi.base_framework.algorithm_api import FedML_Base_distributed
        from .mpi.fedavg.FedAvgAPI import FedML_FedAvg_distributed

        if args.federated_optimizer == FedML_FEDERATED_OPTIMIZER_FEDAVG:
            FedML_FedAvg_distributed(
                args,
                args.process_id,
                args.worker_num,
                args.comm,
                device,
                dataset,
                model,
                client_trainer=client_trainer,
                server_aggregator=server_aggregator,
            )
        elif args.federated_optimizer == FedML_FEDERATED_OPTIMIZER_BASE_FRAMEWORK:
            FedML_Base_distributed(args, args.process_id, args.worker_num, args.comm)
        else:
            raise Exception("Exception")

    def run(self):
        pass
