from abc import ABC, abstractmethod


class ClientAnalyzer(ABC):
    def __init__(self, args):
        self.answer = 0
        self.id = 0
        self.args = args
        self.local_train_dataset = None

    def set_id(self, trainer_id):
        self.id = trainer_id

    @abstractmethod
    def get_exchange_info(self):
        pass

    @abstractmethod
    def set_exchange_info(self, model_parameters):
        pass

    @abstractmethod
    def local_analyze(self, train_data, args):
        pass
