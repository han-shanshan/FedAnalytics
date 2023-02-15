from .constants import (
    FEDML_TRAINING_PLATFORM_SIMULATION,
    FEDML_SIMULATION_TYPE_SP,
)


class FARunner:
    def __init__(
        self,
        args,
        dataset,
        client_trainer=None,
        server_aggregator=None,
    ):

        if args.training_type == FEDML_TRAINING_PLATFORM_SIMULATION:
            init_runner_func = self._init_simulation_runner
        else:
            raise Exception("no such setting")

        self.runner = init_runner_func(
            args, dataset, client_trainer, server_aggregator
        )

    def _init_simulation_runner(
        self, args, dataset, client_trainer=None, server_aggregator=None
    ):
        if hasattr(args, "backend") and args.backend == FEDML_SIMULATION_TYPE_SP:
            from federated_analytics.simulation.sp.simulator import SimulatorSingleProcess

            runner = SimulatorSingleProcess(args, dataset)
        else:
            raise Exception("not such backend {}".format(args.backend))

        return runner

    def run(self):
        self.runner.run()
