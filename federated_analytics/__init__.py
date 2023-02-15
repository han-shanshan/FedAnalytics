# import os
# import sys
# sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))
import federated_analytics
from .arguments import load_arguments
from .constants import (
    FEDML_TRAINING_PLATFORM_SIMULATION,
    FEDML_SIMULATION_TYPE_SP,
)

_global_training_type = None
_global_comm_backend = None


def init(args=None):
    if args is None:
        args = load_arguments(federated_analytics._global_training_type, federated_analytics._global_comm_backend)
    federated_analytics._global_training_type = args.training_type
    federated_analytics._global_comm_backend = args.backend
    if args.training_type == FEDML_TRAINING_PLATFORM_SIMULATION and hasattr(args, "backend") and args.backend == "sp":
        args = init_simulation_sp(args)
    else:
        raise Exception("no such setting: training_type = {}, backend = {}".format(args.training_type, args.backend))
    # os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

    return args


def init_simulation_sp(args):
    return args


from .launch_simulation import run_simulation
from .runner import FARunner

__all__ = [
    "data",
    "FARunner",
    "run_simulation",
    "init"
]
