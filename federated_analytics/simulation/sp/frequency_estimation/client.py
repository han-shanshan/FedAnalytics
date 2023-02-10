import numpy as np


class Client:
    def __init__(
            self, client_idx, local_training_data, local_datasize, args, model_trainer,
    ):
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_datasize = local_datasize
        self.local_sample_number = 0
        self.args = args
        self.model_trainer = model_trainer

    def update_local_dataset(self, client_idx, local_training_data, local_sample_number):
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_sample_number = local_sample_number
        self.model_trainer.set_id(client_idx)

    def get_sample_number(self):
        return self.local_sample_number

    def train(self, w_global):
        self.model_trainer.set_model_params(w_global)
        idxs = np.random.choice(range(len(self.local_training_data)), self.local_sample_number, replace=False)
        train_data = [self.local_training_data[i] for i in idxs]
        # print(f"train data = {train_data}")
        self.model_trainer.train(train_data, self.args)
        return self.model_trainer.get_model_params()
