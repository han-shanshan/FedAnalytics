import sys
import os

module_path = os.path.abspath(os.getcwd() + '/../../..')
print(module_path)
if module_path not in sys.path:
    sys.path.append(module_path)
from federated_analytics import FedMLRunner, init
from federated_analytics.data import load

if __name__ == "__main__":
    # init FedML framework
    args = init()

    # load data
    dataset = load(args)

    # start training
    fa_runner = FedMLRunner(args, dataset)
    fa_runner.run()
