from federated_analytics.core import ClientTrainer


class Average(ClientTrainer):
    def get_model_params(self):
        return self.answer

    def set_model_params(self, model_parameters):
        self.answer = model_parameters

    def train(self, train_data, args):
        # print(f"self.local_sample_number = {self.local_sample_number}")
        sample_num = len(train_data)
        average = 0.0
        # if len() <= self.local_sample_number:
        #     train_data = local_data
        # else:
        #     # np.random.seed(round_idx)  # make sure for each comparison, we are selecting the same clients each round
        #     train_data = [local_data[i] for i in np.random.choice(range(len(local_data)), self.local_sample_number, replace=False)]
        # print(f"{self.local_sample_number}, {train_data}")
        for value in train_data:
            average = average + value / sample_num
        self.set_model_params(average)
