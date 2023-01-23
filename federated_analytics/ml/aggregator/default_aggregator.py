from ...core.frame.server_aggregator import ServerAggregator


class DefaultServerAggregator(ServerAggregator):
    def __init__(self, model, args):
        super().__init__(model, args)

    def get_model_params(self):
        return self.model.state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)
