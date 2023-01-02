from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import List, Tuple
from ...ml.aggregator.agg_operator import FedMLAggOperator


class ServerAggregator(ABC):
    """Abstract base class for federated learning trainer."""

    def __init__(self, model, args):
        self.model = model
        self.id = 0
        self.args = args
        self.eval_data = None

    def set_id(self, aggregator_id):
        self.id = aggregator_id

    @abstractmethod
    def get_model_params(self):
        pass

    @abstractmethod
    def set_model_params(self, model_parameters):
        pass

    def on_before_aggregation(
            self, raw_client_model_or_grad_list: List[Tuple[float, OrderedDict]]
    ):
        return raw_client_model_or_grad_list

    def aggregate(self, raw_client_model_or_grad_list: List[Tuple[float, OrderedDict]]):
        return FedMLAggOperator.agg(self.args, raw_client_model_or_grad_list)

    def on_after_aggregation(self, aggregated_model_or_grad: OrderedDict) -> OrderedDict:
        return aggregated_model_or_grad
