from federated_analytics.FA_components.aggregator.FedAVGAggregator import FedAVGAggregator
from federated_analytics.constants import FA_TASK_AVG


def create_global_analyzer(args):
    task_type = args.task
    if task_type == FA_TASK_AVG:
        return FedAVGAggregator(args)
    # if task_type == FA_TASK_HEAVY_HITTER_TRIEHH:
    #     return TrieHHClientAnalyzer(args)
    # if task_type == FA_TASK_UNION:
    #     return UnionClientAnalyzer(args)
    # if task_type == FA_TASK_K_PERCENTILE_ELEMENT:
    #     return KPercentileElementClientAnalyzer(args)
    # if task_type == FA_TASK_INTERSECTION or task_type == FA_TASK_CARDINALITY:
    #     return IntersectionClientAnalyzer(args)
    # if task_type == FA_TASK_FREQ or task_type == FA_TASK_HISTOGRAM:
    #     return FrequencyEstimationClientAnalyzer(args)

