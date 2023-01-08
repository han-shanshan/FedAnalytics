from federated_analytics.core import ClientTrainer

"""
Federated Heavy Hitters with Differential Privacy. NeurIPS2019 
http://proceedings.mlr.press/v108/zhu20a/zhu20a.pdf
opensource: 
  https://github.com/google-research/federated/tree/master/triehh
  https://github.com/triehh/triehh
"""


class TrieHH(ClientTrainer):
    def get_model_params(self):
        return self.answer

    def set_model_params(self, model_parameters):
        self.answer = model_parameters

    def train(self, train_data, args):
        self.set_model_params(train_data)


