from federated_analytics.core import ClientTrainer


class FrequencyEstimation(ClientTrainer):
    def get_model_params(self):
        return self.answer

    def set_model_params(self, model_parameters):
        self.answer = model_parameters

    def train(self, train_data, args):
        counter_dict = dict()

        for value in train_data:
            if counter_dict.get(value) is None:
                counter_dict[value] = 1
            else:
                counter_dict[value] = counter_dict[value] + 1
        self.set_model_params(counter_dict)
