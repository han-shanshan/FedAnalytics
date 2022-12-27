import fedml
from .runner import FedMLRunner
from .constants import (
    FEDML_TRAINING_PLATFORM_SIMULATION,
    FEDML_SIMULATION_TYPE_SP,
)


def run_simulation(backend=FEDML_SIMULATION_TYPE_SP):
    fedml._global_training_type = FEDML_TRAINING_PLATFORM_SIMULATION
    fedml._global_comm_backend = backend

    # init FedML framework
    args = fedml.init()

    # init device
    device = fedml.device.get_device(args)

    # load data
    dataset, output_dim = fedml.data.load(args)

    # start training
    fedml_runner = FedMLRunner(args, device, dataset)
    fedml_runner.run()

