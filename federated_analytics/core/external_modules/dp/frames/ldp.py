from federated_analytics.core.external_modules.dp.frames.base_dp_solution import BaseDPFrame
from federated_analytics.core.external_modules.dp.mechanisms.dp_mechanism import DPMechanism


class LocalDP(BaseDPFrame):
    def __init__(self, args):
        super().__init__(args)
        self.set_ldp(DPMechanism(args.mechanism_type, args.epsilon, args.delta, args.sensitivity))

    def add_local_noise(self, local_val: float):
        return super().add_local_noise(local_val=local_val)
