import argparse
import logging
import os
import random


def load(args):
    return load_synthetic_data(args)


def generate_fake_data(data_cache_dir):
    file_path = data_cache_dir + "/fake.txt"

    if not os.path.exists(file_path):
        f = open(file_path, "a")
        for i in range(100):
            f.write(f"{random.randint(1, 100)}\n")
        f.close()


def read_data(data_dir):
    train_files = os.listdir(data_dir)
    train_files = [f for f in train_files if f.endswith(".txt")]
    dataset = []
    for f in train_files:
        file_path = os.path.join(data_dir, f)
        f2 = open(file_path, "r")
        lines = [int(line.strip()) for line in f2]
        dataset.extend(lines)
    return dataset


def load_partition_data_fake(data_dir, client_num_in_total):
    dataset = read_data(data_dir=data_dir)
    client_num = client_num_in_total
    client_data_num = int(len(dataset) / client_num)
    local_data_dict = dict()
    train_data_local_num_dict = dict()
    start_counter = 0
    datasize = len(dataset)

    for i in range(client_num):
        local_data_dict[i] = dataset[start_counter:start_counter + client_data_num]
        start_counter += client_data_num
        train_data_local_num_dict[i] = client_data_num
    return (
        datasize,
        train_data_local_num_dict,
        local_data_dict,
    )


def load_synthetic_data(args):
    dataset_name = args.dataset
    if dataset_name == "fake":
        data_cache_dir = os.path.abspath(os.getcwd() + '/../../../federated_analytics' + args.data_cache_dir + "/fake_data")
        if not os.path.exists(data_cache_dir):
            os.makedirs(data_cache_dir)
        print(f"---data_cache_dir={data_cache_dir}")
        generate_fake_data(data_cache_dir)
        logging.info("load_data. dataset_name = %s" % dataset_name)
        (
            datasize,
            train_data_local_num_dict,
            local_data_dict,
        ) = load_partition_data_fake(data_dir=data_cache_dir, client_num_in_total=int(args.client_num_in_total))

        dataset = [
            datasize,
            train_data_local_num_dict,
            local_data_dict,
        ]
        print(f"datasize, train_data_local_num_dict, local_data_dict,{dataset}")
        return dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="FedML")
    parser.add_argument(
        "--yaml_config_file",
        "--cf",
        help="yaml configuration file",
        type=str,
        default="",
    )
    parser.add_argument("--dataset", type=str, default="fake")
    parser.add_argument("--data_cache_dir", type=str, default="fake_data")
    parser.add_argument("--client_num_in_total", type=int, default=10)

    args, unknown = parser.parse_known_args()

    load_synthetic_data(args=args)
    # read_data(train_data_dir="fake_data")
