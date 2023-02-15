from federated_analytics.FA_components.local_analyzer.avg import AverageClientAnalyzer
from federated_analytics.constants import FA_TASK_AVG, FA_TASK_HEAVY_HITTER_TRIEHH, FA_TASK_UNION, \
    FA_TASK_K_PERCENTILE_ELEMENT, FA_TASK_INTERSECTION, FA_TASK_CARDINALITY, FA_TASK_FREQ, FA_TASK_HISTOGRAM
from federated_analytics.FA_components.local_analyzer.frequency_estimation import FrequencyEstimationClientAnalyzer
from federated_analytics.FA_components.local_analyzer.intersection import IntersectionClientAnalyzer
from federated_analytics.FA_components.local_analyzer.k_percentage_element import KPercentileElementClientAnalyzer
from federated_analytics.FA_components.local_analyzer.heavy_hitter_triehh import TrieHHClientAnalyzer
from federated_analytics.FA_components.local_analyzer.union import UnionClientAnalyzer


def create_local_analyzer(args):
    task_type = args.task
    if task_type == FA_TASK_AVG:
        return AverageClientAnalyzer(args)
    if task_type == FA_TASK_HEAVY_HITTER_TRIEHH:
        return TrieHHClientAnalyzer(args)
    if task_type == FA_TASK_UNION:
        return UnionClientAnalyzer(args)
    if task_type == FA_TASK_K_PERCENTILE_ELEMENT:
        return KPercentileElementClientAnalyzer(args)
    if task_type == FA_TASK_INTERSECTION or task_type == FA_TASK_CARDINALITY:
        return IntersectionClientAnalyzer(args)
    if task_type == FA_TASK_FREQ or task_type == FA_TASK_HISTOGRAM:
        return FrequencyEstimationClientAnalyzer(args)

