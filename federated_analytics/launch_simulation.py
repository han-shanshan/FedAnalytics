from . import init, data
from .runner import FedMLRunner
from .constants import (
    FEDML_SIMULATION_TYPE_SP,
)


def run_simulation(backend=FEDML_SIMULATION_TYPE_SP):
    # federated_analytics._global_training_type = FEDML_TRAINING_PLATFORM_SIMULATION
    # federated_analytics._global_comm_backend = backend

    # init FedML framework
    # args = federated_analytics.init()
    args = init()

    # load data
    dataset, output_dim = data.load(args)  # todo

    # start training
    fedml_runner = FedMLRunner(args, dataset)
    fedml_runner.run()

