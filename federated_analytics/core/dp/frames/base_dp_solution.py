from abc import ABC
from typing import List, Tuple
from federated_analytics.core.dp.mechanisms.dp_mechanism import DPMechanism


class BaseDPFrame(ABC):
    def __init__(self, args=None):
        self.cdp = None
        self.ldp = None
        self.args = args

    def set_cdp(self, dp_mechanism: DPMechanism):
        self.cdp = dp_mechanism

    def set_ldp(self, dp_mechanism: DPMechanism):
        self.ldp = dp_mechanism

    def add_local_noise(self, local_val):
        return self.ldp.add_noise(real_val=local_val)

    def add_global_noise(self, agg_result):
        return self.cdp.add_noise(real_val=agg_result)

    def set_params_for_dp(self, raw_client_value_list: List[Tuple[float, float]]):
        pass


