from federated_analytics.FA_components.aggregator.avg_aggregator import AVGAggregator
from federated_analytics.FA_components.aggregator.frequency_estimation_aggregator import FrequencyEstimationAggregator
from federated_analytics.FA_components.aggregator.heavy_hitter_triehh_aggregator import HeavyHitterTriehhAggregator
from federated_analytics.FA_components.aggregator.intersection_aggregator import IntersectionAggregator
from federated_analytics.FA_components.aggregator.k_percentile_element_aggregator import KPercentileElementAggregator
from federated_analytics.FA_components.aggregator.union_aggregator import UnionAggregator
from federated_analytics.constants import FA_TASK_AVG, FA_TASK_FREQ, FA_TASK_HISTOGRAM, FA_TASK_INTERSECTION, \
    FA_TASK_CARDINALITY, FA_TASK_HEAVY_HITTER_TRIEHH, FA_TASK_UNION, FA_TASK_K_PERCENTILE_ELEMENT


def create_global_analyzer(args, train_data_num):
    task_type = args.task
    if task_type == FA_TASK_AVG:
        return AVGAggregator(args)
    if task_type == FA_TASK_INTERSECTION or task_type == FA_TASK_CARDINALITY:
        return IntersectionAggregator(args)
    if task_type == FA_TASK_FREQ or task_type == FA_TASK_HISTOGRAM:
        return FrequencyEstimationAggregator(args)
    if task_type == FA_TASK_UNION:
        return UnionAggregator(args)
    if task_type == FA_TASK_K_PERCENTILE_ELEMENT:
        return KPercentileElementAggregator(args, train_data_num)
    if task_type == FA_TASK_HEAVY_HITTER_TRIEHH:
        return HeavyHitterTriehhAggregator(args, train_data_num)



