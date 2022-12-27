from python.fedml.ml.trainer.avg import Average


def create_model_trainer(task_type):
    if task_type == "avg":
        return Average(args=None)