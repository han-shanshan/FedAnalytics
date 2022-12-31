from python.fedml.constants import FA_TASK_AVG
from python.fedml.ml.trainer.avg import Average


def create_model_trainer(task_type, args):
    if task_type == FA_TASK_AVG:
        return Average(args)
