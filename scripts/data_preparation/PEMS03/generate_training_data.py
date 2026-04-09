import os
import sys
import pickle
import argparse

import numpy as np
import pandas as pd

# TODO: remove it when basicts can be installed by pip
sys.path.append(os.path.abspath(__file__ + "/../../../.."))
from basicts.data.transform import standard_transform


def get_adjacency_matrix(distance_df, num_sensors=358, normalized_k=0.1):
    """Generate adjacency matrix from distance CSV file."""
    sensor_ids = [x for x in range(num_sensors)]

    dist_mx = np.zeros((num_sensors, num_sensors), dtype=np.float32)
    dist_mx[:] = np.inf

    sensor_id_to_ind = {}
    for i, sensor_id in enumerate(sensor_ids):
        sensor_id_to_ind[sensor_id] = i

    for row in distance_df.values:
        if row[0] not in sensor_id_to_ind or row[1] not in sensor_id_to_ind:
            continue
        dist_mx[sensor_id_to_ind[row[0]], sensor_id_to_ind[row[1]]] = row[2]

    distances = dist_mx[~np.isinf(dist_mx)].flatten()
    std = distances.std()
    adj_mx = np.exp(-np.square(dist_mx / std))
    adj_mx[adj_mx < normalized_k] = 0

    return sensor_ids, sensor_id_to_ind, adj_mx


def generate_data(args: argparse.Namespace):
    """Preprocess and generate train/valid/test datasets.

    Default settings of PEMS03 dataset:
        - Normalization method: standard norm.
        - Dataset division: 6:2:2.
        - Window size: history 12, future 12.
        - Channels (features): three channels [traffic flow, time of day, day of week]
        - Target: predict the traffic speed of the future 12 time steps.

    Args:
        args (argparse): configurations of preprocessing
    """
    target_channel = args.target_channel
    future_seq_len = args.future_seq_len
    history_seq_len = args.history_seq_len
    add_time_of_day = args.tod
    add_day_of_week = args.dow
    output_dir = args.output_dir
    data_file_path = args.data_file_path
    steps_per_day = args.steps_per_day

    # read data
    data = np.load(data_file_path)["data"]
    data = data[..., target_channel]
    print("raw time series shape: {0}".format(data.shape))

    l, n, f = data.shape
    num_samples = l - (history_seq_len + future_seq_len) + 1

    # Fixed dataset split for PEMS03
    test_num_short = 5237
    valid_num_short = 5237
    train_num_short = num_samples - valid_num_short - test_num_short

    print("number of training samples:{0}".format(train_num_short))
    print("number of validation samples:{0}".format(valid_num_short))
    print("number of test samples:{0}".format(test_num_short))

    index_list = []
    for t in range(history_seq_len, num_samples + history_seq_len):
        index = (t - history_seq_len, t, t + future_seq_len)
        index_list.append(index)

    train_index = index_list[:train_num_short]
    valid_index = index_list[train_num_short: train_num_short + valid_num_short]
    test_index = index_list[train_num_short + valid_num_short: train_num_short + valid_num_short + test_num_short]

    scaler = standard_transform
    data_norm = scaler(data, output_dir, train_index, history_seq_len, future_seq_len)

    # add external feature
    feature_list = [data_norm]

    if add_time_of_day:
        # numerical time_of_day
        tod = [i % steps_per_day / steps_per_day for i in range(data_norm.shape[0])]
        tod = np.array(tod)
        tod_tiled = np.tile(tod, [1, n, 1]).transpose((2, 1, 0))
        feature_list.append(tod_tiled)

    if add_day_of_week:
        # numerical day_of_week
        dow = [(i // steps_per_day) % 7 for i in range(data_norm.shape[0])]
        dow = np.array(dow)
        dow_tiled = np.tile(dow, [1, n, 1]).transpose((2, 1, 0))
        feature_list.append(dow_tiled)

    processed_data = np.concatenate(feature_list, axis=-1)
    print("Final processed data shape: {0}".format(processed_data.shape))

    # dump data
    index = {}
    index["train"] = train_index
    index["valid"] = valid_index
    index["test"] = test_index
    with open(output_dir + "/index_in{0}_out{1}.pkl".format(history_seq_len, future_seq_len), "wb") as f:
        pickle.dump(index, f)

    data = {}
    data["processed_data"] = processed_data
    with open(output_dir + "/data_in{0}_out{1}.pkl".format(history_seq_len, future_seq_len), "wb") as f:
        pickle.dump(data, f)

    # generate and save adj
    if os.path.exists(args.graph_file_path):
        print("Generating adjacency matrix from: {}".format(args.graph_file_path))
        distance_df = pd.read_csv(args.graph_file_path, dtype={'from': 'int', 'to': 'int'})
        sensor_ids, sensor_id_to_ind, adj_mx = get_adjacency_matrix(distance_df, num_sensors=n)
        with open(os.path.join(output_dir, 'adj_mx.pkl'), 'wb') as f:
            pickle.dump([sensor_ids, sensor_id_to_ind, adj_mx], f, protocol=2)
        print("Saved adjacency matrix to: {}".format(os.path.join(output_dir, 'adj_mx.pkl')))
    else:
        print("Warning: graph file not found at {}, skipping adj generation".format(args.graph_file_path))


if __name__ == "__main__":
    # sliding window size for generating history sequence and target sequence
    HISTORY_SEQ_LEN = 12
    FUTURE_SEQ_LEN = 12

    TRAIN_RATIO = 0.6
    VALID_RATIO = 0.2
    TARGET_CHANNEL = [0]
    STEPS_PER_DAY = 288

    DATASET_NAME = "PEMS03"
    TOD = True
    DOW = True

    # Get script directory for relative paths
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

    OUTPUT_DIR = os.path.join(SCRIPT_DIR, "../../../datasets", DATASET_NAME)
    DATA_FILE_PATH = os.path.join(SCRIPT_DIR, "datasets/raw_data/{0}/{0}.npz".format(DATASET_NAME))
    GRAPH_FILE_PATH = os.path.join(SCRIPT_DIR, "datasets/raw_data/{0}/PEMS03.csv".format(DATASET_NAME))

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str,
                        default=OUTPUT_DIR, help="Output directory.")
    parser.add_argument("--data_file_path", type=str,
                        default=DATA_FILE_PATH, help="Raw traffic readings.")
    parser.add_argument("--graph_file_path", type=str,
                        default=GRAPH_FILE_PATH, help="Raw traffic readings.")
    parser.add_argument("--history_seq_len", type=int,
                        default=HISTORY_SEQ_LEN, help="Sequence Length.")
    parser.add_argument("--future_seq_len", type=int,
                        default=FUTURE_SEQ_LEN, help="Sequence Length.")
    parser.add_argument("--steps_per_day", type=int,
                        default=STEPS_PER_DAY, help="Sequence Length.")
    parser.add_argument("--tod", type=bool, default=TOD,
                        help="Add feature time_of_day.")
    parser.add_argument("--dow", type=bool, default=DOW,
                        help="Add feature day_of_week.")
    parser.add_argument("--target_channel", type=list,
                        default=TARGET_CHANNEL, help="Selected channels.")
    parser.add_argument("--train_ratio", type=float,
                        default=TRAIN_RATIO, help="Train ratio")
    parser.add_argument("--valid_ratio", type=float,
                        default=VALID_RATIO, help="Validate ratio.")

    args_metr = parser.parse_args()

    # print args
    print("-" * (20 + 45 + 5))
    for key, value in sorted(vars(args_metr).items()):
        print("|{0:>20} = {1:<45}|".format(key, str(value)))
    print("-" * (20 + 45 + 5))

    if not os.path.exists(args_metr.output_dir):
        os.makedirs(args_metr.output_dir)

    # Generate training data
    generate_data(args_metr)
