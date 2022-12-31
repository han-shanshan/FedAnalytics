from python.fedml.core import ClientTrainer


class Average(ClientTrainer):
    # def __init__(self, args):
    #     super().__init__(args)
        # self.total_sample_num_global = 0
        # self.total_sample_num_local = 0

    def get_model_params(self):
        return self.answer

    def set_model_params(self, model_parameters):
        self.answer = model_parameters

    def train(self, train_data, args):
        average = 0.0
        for value in train_data:
            average = average + value / self.local_sample_number
        self.set_model_params(average)
