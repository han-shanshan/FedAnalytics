from .default_aggregator import DefaultServerAggregator


def create_server_aggregator(model, args):
    return DefaultServerAggregator(model, args)
