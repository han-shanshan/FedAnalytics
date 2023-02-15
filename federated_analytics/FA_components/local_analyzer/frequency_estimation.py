from federated_analytics.core import ClientAnalyzer


class FrequencyEstimationClientAnalyzer(ClientAnalyzer):
    def local_analyze(self, train_data, args):
        counter_dict = dict()

        for value in train_data:
            if counter_dict.get(value) is None:
                counter_dict[value] = 1
            else:
                counter_dict[value] = counter_dict[value] + 1
        self.set_client_submission(counter_dict)