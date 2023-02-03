from federated_analytics.core import ClientTrainer


class Union(ClientTrainer):
    def get_model_params(self):
        return self.answer

    def set_model_params(self, model_parameters):
        self.answer = model_parameters

    def train(self, train_data, args):
        self.set_model_params(list(set(train_data)))
