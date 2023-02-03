import logging
from typing import List, Tuple

from federated_analytics.core.external_modules.dp.common.constants import DP_LDP, DP_CDP
from federated_analytics.core.external_modules.dp.frames.cdp import GlobalDP
from federated_analytics.core.external_modules.dp.frames.ldp import LocalDP


class FedMLDifferentialPrivacy:
    _dp_instance = None

    @staticmethod
    def get_instance():
        if FedMLDifferentialPrivacy._dp_instance is None:
            FedMLDifferentialPrivacy._dp_instance = FedMLDifferentialPrivacy()
        return FedMLDifferentialPrivacy._dp_instance

    def __init__(self):
        self.dp_solution_type = None
        self.dp_solution = None
        self.dp_accountant = None
        self.is_enabled = False
        self.privacy_engine = None
        self.current_round = 0
        self.accountant = None

    def init(self, args):
        if hasattr(args, "enable_dp") and args.enable_dp:
            logging.info(".......init dp......." + args.dp_solution_type + "-" + args.dp_solution_type)
            self.is_enabled = True
            self.dp_solution_type = args.dp_solution_type.strip()
            logging.info("self.dp_solution = {}".format(self.dp_solution_type))
            print(f"dp_solution_type={self.dp_solution_type}")

            if self.dp_solution_type == DP_LDP:
                self.dp_solution = LocalDP(args)
            elif self.dp_solution_type == DP_CDP:
                self.dp_solution = GlobalDP(args)
            else:
                raise Exception("dp solution is not defined")

    def is_dp_enabled(self):
        return self.is_enabled

    def is_local_dp_enabled(self):
        return self.is_enabled and self.dp_solution_type in [DP_LDP]

    def is_global_dp_enabled(self):
        return self.is_enabled and self.dp_solution_type in [DP_CDP]

    def add_local_noise(self, local_val: float):
        if self.dp_solution is None:
            raise Exception("dp solution is not initialized!")
        return self.dp_solution.add_local_noise(local_val)

    def add_global_noise(self, aggr_val: float):
        if self.dp_solution is None:
            raise Exception("dp solution is not initialized!")
        return self.dp_solution.add_global_noise(aggr_val)

    def set_params_for_dp(self, raw_client_value_list: List[Tuple[float, float]]):
        if self.dp_solution is None:
            raise Exception("dp solution is not initialized!")
        self.dp_solution.set_params_for_dp(raw_client_value_list)

