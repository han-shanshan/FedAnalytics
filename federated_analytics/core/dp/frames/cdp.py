from federated_analytics.core.dp.frames.base_dp_solution import BaseDPFrame
from federated_analytics.core.dp.mechanisms.dp_mechanism import DPMechanism


class GlobalDP(BaseDPFrame):
    def __init__(self, args):
        super().__init__(args)

        self.set_cdp(
            DPMechanism(args.mechanism_type, args.epsilon, args.delta, args.sensitivity)
        )

    def add_global_noise(self, global_model: float):
        return super().add_global_noise(agg_result=global_model)