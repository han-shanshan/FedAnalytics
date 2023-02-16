from federated_analytics.core import FAClientAnalyzer


class IntersectionClientAnalyzer(FAClientAnalyzer):
    def local_analyze(self, train_data, args):
        self.set_client_submission(list(set(train_data)))