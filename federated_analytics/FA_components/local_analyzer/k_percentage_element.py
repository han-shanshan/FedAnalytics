from federated_analytics.core import ClientAnalyzer


class KPercentileElementClientAnalyzer(ClientAnalyzer):
    def get_exchange_info(self):
        return self.answer

    def set_exchange_info(self, model_parameters):
        self.answer = model_parameters

    def local_analyze(self, train_data, args):
        counter = 0
        for data in train_data:
            if data >= self.answer:  # flag
                counter += 1
        self.set_exchange_info(counter)  # number of values that are larger than flag
