import argparse
import logging
import os
import pickle
import random
import wget
import zipfile
from federated_analytics.constants import FA_DATA_TWITTER_Sentiment140_URL
from federated_analytics.data.preprocessing.preprocess_twitter_sentiment140 import preprocess_twitter_data


def load(args):
    return load_synthetic_data(args)


def generate_fake_data(data_cache_dir):
    file_path = data_cache_dir + "/fake.txt"

    if not os.path.exists(file_path):
        f = open(file_path, "a")
        for i in range(10000):
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


def load_partition_data_fake(data_dir, client_num):
    dataset = read_data(data_dir=data_dir)
    return equally_partition_a_dataset(client_num, dataset)


def load_partition_data_twitter_sentiment140(data_dir, client_num_in_total):
    clients_triehh_file = os.path.join(data_dir, 'clients_triehh.txt')
    with open(clients_triehh_file, 'rb') as fp:
        dataset = pickle.load(fp)

    return equally_partition_a_dataset(client_num_in_total, dataset)


def equally_partition_a_dataset(client_num_in_total, dataset):
    client_data_num = int(len(dataset) / client_num_in_total)
    local_data_dict = dict()
    train_data_local_num_dict = dict()
    start_counter = 0
    datasize = len(dataset)
    for i in range(client_num_in_total):
        local_data_dict[i] = dataset[start_counter:start_counter + client_data_num]
        start_counter += client_data_num
        train_data_local_num_dict[i] = client_data_num
    return (
        datasize,
        train_data_local_num_dict,
        local_data_dict,
    )


def download_twitter_Sentiment140(data_cache_dir):
    if not os.path.exists(data_cache_dir):
        os.makedirs(data_cache_dir)
    file_path = os.path.join(data_cache_dir, "trainingandtestdata.zip")

    if not os.path.exists(file_path): # Download the file (if we haven't already)
        wget.download(FA_DATA_TWITTER_Sentiment140_URL, out=file_path)

    with zipfile.ZipFile(file_path, "r") as zip_ref:
        zip_ref.extractall(data_cache_dir)


def load_synthetic_data(args):
    dataset_name = args.dataset
    if dataset_name == "fake":
        data_cache_dir = os.path.abspath(
            os.getcwd() + '/../../../federated_analytics' + args.data_cache_dir + "/fake_data")
        if not os.path.exists(data_cache_dir):
            os.makedirs(data_cache_dir)
        print(f"---data_cache_dir={data_cache_dir}")
        generate_fake_data(data_cache_dir)
        logging.info("load_data. dataset_name = %s" % dataset_name)
        (
            datasize,
            train_data_local_num_dict,
            local_data_dict,
        ) = load_partition_data_fake(data_dir=data_cache_dir, client_num=int(args.client_num_in_total))

        dataset = [
            datasize,
            train_data_local_num_dict,
            local_data_dict,
        ]
        # print(f"datasize, train_data_local_num_dict, local_data_dict,{dataset}")
    elif dataset_name == "twitter":
        path = os.path.abspath(os.getcwd() + '../../../federated_analytics/' + args.data_cache_dir + '/twitter_Sentiment140/')
        download_twitter_Sentiment140(data_cache_dir=path)
        preprocess_twitter_data(path=path)
        (
            datasize,
            train_data_local_num_dict,
            local_data_dict,
        ) = load_partition_data_twitter_sentiment140(data_dir=path, client_num_in_total=int(args.client_num_in_total))

        dataset = [
            datasize,
            train_data_local_num_dict,
            local_data_dict,
        ]
        # print(f"datasize, train_data_local_num_dict, local_data_dict,{dataset}")
    else:
        raise "Not Implemented Error"
    return dataset


# with open(clients_triehh_file, 'rb') as fp:
#         self.clients = pickle.load(fp)
#     self.client_num = len(self.clients)

# def read_data(data_dir):
#     train_files = os.listdir(data_dir)
#     train_files = [f for f in train_files if f.endswith(".txt")]
#     dataset = []
#     for f in train_files:
#         file_path = os.path.join(data_dir, f)
#         f2 = open(file_path, "r")
#         lines = [int(line.strip()) for line in f2]
#         dataset.extend(lines)
#     return dataset


def load_synthetic_data_test():
    parser = argparse.ArgumentParser(description="FedML")
    parser.add_argument(
        "--yaml_config_file",
        "--cf",
        help="yaml configuration file",
        type=str,
        default="",
    )
    parser.add_argument("--dataset", type=str, default="twitter")
    parser.add_argument("--data_cache_dir", type=str, default="data")
    parser.add_argument("--client_num_in_total", type=int, default=10)

    args, unknown = parser.parse_known_args()

    load_synthetic_data(args=args)


if __name__ == '__main__':
    # read_data(train_data_dir="fake_data")
    # download_twitter_Sentiment140("data")
    load_synthetic_data_test()
