import subprocess
from fedml.arguments import load_arguments
from fedml.constants import (
    FEDML_TRAINING_PLATFORM_CROSS_SILO,
    FEDML_CROSS_SILO_SCENARIO_HORIZONTAL,
)


class CrossSiloLauncher:
    @staticmethod
    def launch_dist_trainers(torch_client_filename, inputs):
        # this is only used by the client (DDP or single process), so there is no need to specify the backend.
        args = load_arguments(FEDML_TRAINING_PLATFORM_CROSS_SILO)
        if args.scenario == FEDML_CROSS_SILO_SCENARIO_HORIZONTAL:
            CrossSiloLauncher._run_cross_silo_horizontal(args, torch_client_filename, inputs)
        else:
            raise Exception("we do not support {}, check whether this is typo in args.scenario".format(args.scenario))

    @staticmethod
    def _run_cross_silo_horizontal(args, torch_client_filename, inputs):
        python_path = subprocess.run(["which", "python"], capture_output=True, text=True).stdout.strip()
        process_arguments = [python_path, torch_client_filename] + inputs
        subprocess.run(process_arguments)