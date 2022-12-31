import logging
import random
import numpy as np
import torch
import fedml
from .constants import (
    FEDML_TRAINING_PLATFORM_SIMULATION,
    FEDML_SIMULATION_TYPE_SP,
)

_global_training_type = None
_global_comm_backend = None


def init(args=None):
    fedml._global_training_type = args.training_type
    fedml._global_comm_backend = args.backend
    seed = args.random_seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if args.training_type == FEDML_TRAINING_PLATFORM_SIMULATION and hasattr(args, "backend") and args.backend == "sp":
        args = init_simulation_sp(args)
    else:
        raise Exception("no such setting: training_type = {}, backend = {}".format(args.training_type, args.backend))

    print_args(args)

    return args


def print_args(args):
    mqtt_config_path = None
    s3_config_path = None
    args_copy = args
    if hasattr(args_copy, "mqtt_config_path"):
        mqtt_config_path = args_copy.mqtt_config_path
        args_copy.mqtt_config_path = ""
    if hasattr(args_copy, "s3_config_path"):
        s3_config_path = args_copy.s3_config_path
        args_copy.s3_config_path = ""
    logging.info("==== args = {}".format(vars(args_copy)))
    if hasattr(args_copy, "mqtt_config_path"):
        args_copy.mqtt_config_path = mqtt_config_path
    if hasattr(args_copy, "s3_config_path"):
        args_copy.s3_config_path = s3_config_path


def init_simulation_sp(args):
    return args


from fedml import device
from fedml import data
from fedml import model
from .launch_simulation import run_simulation
from .runner import FedMLRunner

__all__ = [
    "device",
    "data",
    "model",
    "FedMLRunner",
    "run_simulation",
]
