from federated_analytics.core import ClientAnalyzer


class FrequencyEstimationClientAnalyzer(ClientAnalyzer):
    def get_exchange_info(self):
        return self.answer

    def set_exchange_info(self, model_parameters):
        self.answer = model_parameters

    def local_analyze(self, train_data, args):
        counter_dict = dict()

        for value in train_data:
            if counter_dict.get(value) is None:
                counter_dict[value] = 1
            else:
                counter_dict[value] = counter_dict[value] + 1
        self.set_exchange_info(counter_dict)
