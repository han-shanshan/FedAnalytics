from federated_analytics.constants import FA_TASK_AVG, FA_TASK_HEAVY_HITTER_TRIEHH, FA_TASK_UNION, \
    FA_TASK_K_PERCENTILE_ELEMENT, FA_TASK_INTERSECTION, FA_TASK_CARDINALITY, FA_TASK_FREQ
from federated_analytics.ml.trainer.avg import Average
from federated_analytics.ml.trainer.frequency_estimation import FrequencyEstimation
from federated_analytics.ml.trainer.intersection import Intersection
from federated_analytics.ml.trainer.k_percentage_element import KPercentileElement
from federated_analytics.ml.trainer.triehh import TrieHH
from federated_analytics.ml.trainer.union import Union


def create_model_trainer(task_type, args):
    if task_type == FA_TASK_AVG:
        return Average(args)
    if task_type == FA_TASK_HEAVY_HITTER_TRIEHH:
        return TrieHH(args)
    if task_type == FA_TASK_UNION:
        return Union(args)
    if task_type == FA_TASK_K_PERCENTILE_ELEMENT:
        return KPercentileElement(args)
    if task_type == FA_TASK_INTERSECTION or task_type == FA_TASK_CARDINALITY:
        return Intersection(args)
    if task_type == FA_TASK_FREQ:
        return FrequencyEstimation(args)

