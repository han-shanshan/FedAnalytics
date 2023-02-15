from abc import ABC
from typing import List, Tuple


class ServerAggregator(ABC):
    def __init__(self, args):
        self.id = 0
        self.args = args
        self.eval_data = None
        self.exchange_info = None

    def set_id(self, aggregator_id):
        self.id = aggregator_id

    def get_exchange_info(self):
        return self.exchange_info

    def set_exchange_info(self, exchange_info):
        self.exchange_info = exchange_info

    def aggregate(self, local_submissions: List[Tuple[float, float]]):
        # return FAAggOperator.agg(self.args, local_submissions)
        pass
