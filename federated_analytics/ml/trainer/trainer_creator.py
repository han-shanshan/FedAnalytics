from federated_analytics.constants import FA_TASK_AVG, FA_TASK_HEAVY_HITTER_TRIEHH
from federated_analytics.ml.trainer.avg import Average
from federated_analytics.ml.trainer.triehh import TrieHH


def create_model_trainer(task_type, args):
    if task_type == FA_TASK_AVG:
        return Average(args)
    if task_type == FA_TASK_HEAVY_HITTER_TRIEHH:
        return TrieHH(args)

