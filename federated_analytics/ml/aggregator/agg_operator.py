from collections import OrderedDict
from typing import List, Tuple
from federated_analytics.constants import FA_TASK_AVG


class FedMLAggOperator:
    @staticmethod
    def agg(args, raw_grad_list: List[Tuple[float, OrderedDict]]) -> OrderedDict:
        training_num = 0
        if args.federated_optimizer == FA_TASK_AVG:
            for i in range(len(raw_grad_list)):
                local_sample_num, _ = raw_grad_list[i]
                training_num += local_sample_num
        else:
            for i in range(len(raw_grad_list)):
                local_sample_num, local_model_params = raw_grad_list[i]
                training_num += local_sample_num

        avg_params = aggregator(args, raw_grad_list, training_num)  # todo: avg: consider previous rounds?
        return avg_params


def aggregator(args, raw_grad_list, training_num):
    avg_params = OrderedDict()
    if args.federated_optimizer == FA_TASK_AVG:
        (num0, avg_params) = raw_grad_list[0]
        for k in avg_params.keys():
            for i in range(0, len(raw_grad_list)):
                local_sample_number, local_model_params = raw_grad_list[i]
                w = local_sample_number / training_num
                if i == 0:
                    avg_params[k] = local_model_params[k] * w
                else:
                    avg_params[k] += local_model_params[k] * w
    return avg_params