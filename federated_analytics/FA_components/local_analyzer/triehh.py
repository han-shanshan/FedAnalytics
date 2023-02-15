from federated_analytics.core import ClientAnalyzer

"""
Federated Heavy Hitters with Differential Privacy. NeurIPS2019 
http://proceedings.mlr.press/v108/zhu20a/zhu20a.pdf
opensource: 
  https://github.com/google-research/federated/tree/master/triehh
  https://github.com/triehh/triehh
"""


class TrieHHClientAnalyzer(ClientAnalyzer):
    def get_exchange_info(self):
        return self.answer

    def set_exchange_info(self, model_parameters):
        self.answer = model_parameters

    def local_analyze(self, train_data, args):
        self.set_exchange_info(train_data)


