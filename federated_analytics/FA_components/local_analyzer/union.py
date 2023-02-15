from federated_analytics.core import ClientAnalyzer


class UnionClientAnalyzer(ClientAnalyzer):
    def get_exchange_info(self):
        return self.answer

    def set_exchange_info(self, model_parameters):
        self.answer = model_parameters

    def local_analyze(self, train_data, args):
        self.set_exchange_info(list(set(train_data)))
