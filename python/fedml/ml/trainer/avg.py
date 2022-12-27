from python.fedml.core import ClientTrainer


class Average(ClientTrainer):
    def get_model_params(self):
        return self.local_answer

    def set_model_params(self, model_parameters):
        self.local_answer = model_parameters

    def train(self, train_data, args):
        average = 0.0
        for value in train_data:
            average = average + value / self.local_sample_number
        self.set_model_params(average)