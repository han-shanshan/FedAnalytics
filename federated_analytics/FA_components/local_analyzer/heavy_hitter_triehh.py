from federated_analytics.core import FAClientAnalyzer

"""
Federated Heavy Hitters with Differential Privacy. NeurIPS2019 
http://proceedings.mlr.press/v108/zhu20a/zhu20a.pdf
opensource: 
  https://github.com/google-research/federated/tree/master/triehh
  https://github.com/triehh/triehh
"""


class TrieHHClientAnalyzer(FAClientAnalyzer):
    def local_analyze(self, train_data, args):
        self.set_client_submission(train_data)

