from .sp.heavy_hitter.triehh_api import TrieHHSimulator
from .sp.intersection.intersection_api import IntersectionSimulator
from .sp.k_percentile_element.k_percentile_element_api import KPercentileElementSimulator
from .sp.union.union_api import UnionSimulator
from ..constants import FA_TASK_AVG, FA_TASK_HEAVY_HITTER_TRIEHH, FA_TASK_UNION, FA_TASK_K_PERCENTILE_ELEMENT, \
    FA_TASK_INTERSECTION, FA_TASK_CARDINALITY
from ..core import ClientTrainer, ServerAggregator


class SimulatorSingleProcess:
    def __init__(self, args, dataset, client_trainer: ClientTrainer = None, server_aggregator: ServerAggregator = None,):
        from .sp.fedavg import FedAvgSimulator
        if args.task == FA_TASK_AVG:
            self.fl_trainer = FedAvgSimulator(args, dataset)
        elif args.task == FA_TASK_HEAVY_HITTER_TRIEHH:
            self.fl_trainer = TrieHHSimulator(args, dataset)
        elif args.task == FA_TASK_UNION:
            self.fl_trainer = UnionSimulator(args, dataset)
        elif args.task == FA_TASK_K_PERCENTILE_ELEMENT:
            self.fl_trainer = KPercentileElementSimulator(args, dataset)
        elif args.task == FA_TASK_INTERSECTION or args.task == FA_TASK_CARDINALITY:
            self.fl_trainer = IntersectionSimulator(args, dataset)
        else:
            raise Exception("Exception")

    def run(self):
        self.fl_trainer.train()

