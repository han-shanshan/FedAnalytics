from federated_analytics.core import ClientTrainer


class KPercentileElement(ClientTrainer):
    # def __init__(self):
    #     super().__init__()
    #     self.flag = None
    #
    # def get_flag(self):
    #     return self.flag
    #
    # def set_flag(self, flag):
    #     self.flag = flag

    def get_model_params(self):
        return self.answer

    def set_model_params(self, model_parameters):
        self.answer = model_parameters

    def train(self, train_data, args):
        counter = 0
        for data in train_data:
            if data >= self.answer:  # flag
                counter += 1
        self.set_model_params(counter) # number of values that are larger than flag
