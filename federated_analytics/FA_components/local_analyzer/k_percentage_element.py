from federated_analytics.core import ClientAnalyzer


class KPercentileElementClientAnalyzer(ClientAnalyzer):
    def local_analyze(self, train_data, args):
        counter = 0
        for data in train_data:
            # print(f"train_data={train_data}")
            if data >= self.server_data:  # flag
                counter += 1
        self.set_client_submission(counter)  # number of values that are larger than flag
