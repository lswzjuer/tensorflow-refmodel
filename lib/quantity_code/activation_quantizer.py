import argparse
import numpy as np
import os
from collections import OrderedDict, defaultdict, Counter
import yaml
import time
import cv2
from random import shuffle
import tensorflow as tf

from quantity.tools.distribution_collector import DistributionCollector
from quantity.tools.quantizer import Quantizer
from quantity.tools import (
    load_graph, print_ops, get_params, build_net, get_activations, prune_net_info, get_merge_groups)


def prepare_data(cfg, file_type, file_dir=None):
    file_dir = file_dir if file_dir else cfg['PATH']['DATA_PATH']
    files_path = []

    for root, dir, files in os.walk(file_dir):
        for name in files:
            file_path = root + "/" + name
            if not file_path.endswith(file_type):
                continue
            files_path.append(file_path)

    return files_path


def activation_quantize(net, net_def, cfg, images_files, task_type):
    interval_num = cfg['SETTINGS']['INTERVAL_NUM']
    statistic = cfg['SETTINGS']['STATISTIC']
    worker_num = cfg['SETTINGS']['WORKER_NUM']
    table_file = cfg['OUTPUT']['FEAT_BIT_TABLE']

    named_feats = get_activations(net, images_files[0], task_type)
    net_info = build_net(net_def)
    net_info = prune_net_info(net_info, named_feats)

    table_file_content = []
    bottom_feat_names = get_merge_groups(net_info)
    top_feat_names = list(named_feats.keys())

    max_vals = {}
    distribution_intervals = {}
    for i, feat_name in enumerate(top_feat_names):
        max_vals[feat_name] = 0
        distribution_intervals[feat_name] = 0

    collector = DistributionCollector(top_feat_names, interval_num=interval_num,
                                      statistic=statistic, worker_num=worker_num,
                                      debug=False)
    quantizer = Quantizer(top_feat_names, worker_num=worker_num, debug=False)

    # run float32 inference on calibration dataset to find the activations range
    for i, image in enumerate(images_files):
        named_feats = get_activations(net, image, task_type)
        print("loop stage 1 : %d" % (i))
        # find max threshold
        tensors = {}
        for name, feat in named_feats.items():
            tensors[name] = feat.flatten()
        collector.refresh_max_val(tensors)

    print(collector.max_vals)
    distribution_intervals = collector.distribution_intervals
    for b_names in bottom_feat_names:
        assert len(b_names) > 1
        tmp_distribution_interval = 0
        # distributions
        for pre_feat_name in b_names:
            tmp_distribution_interval = max(tmp_distribution_interval,
                                            distribution_intervals[pre_feat_name])
        for pre_feat_name in b_names:
            distribution_intervals[pre_feat_name] = tmp_distribution_interval

    # for each layer, collect histograms of activations
    print("\nCollect histograms of activations:")
    for i, image in enumerate(images_files):
        named_feats = get_activations(net, image, task_type)
        print("loop stage 2 : %d" % (i))
        start = time.clock()
        tensors = {}
        for name, feat in named_feats.items():
            tensors[name] = feat.flatten()
        collector.add_to_distributions(tensors)
        end = time.clock()
        print("add cost %.3f s" % (end - start))

    distributions = collector.distributions

    # refresh the distribution of the bottom feat of layers like concat and eltwise.
    for b_names in bottom_feat_names:
        assert len(b_names) > 1
        tmp_distributions = np.zeros(interval_num)
        # distributions
        for pre_feat_name in b_names:
            tmp_distributions += distributions[pre_feat_name]
        for pre_feat_name in b_names:
            distributions[pre_feat_name] = tmp_distributions

    quantizer.quantize(distributions, distribution_intervals)
    bits = quantizer.bits
    for feat_name in top_feat_names:
        feat_str = feat_name + " " + str(bits[feat_name])
        if feat_name != 'image' and net_info[feat_name]['inputs'] != [None]:
            for inp_name in net_info[feat_name]['inputs']:
                feat_str = feat_str + ' ' + str(bits[inp_name])
        print(feat_str)
        table_file_content.append(feat_str)

    with open(table_file, 'w') as f:
        for tabel_line in table_file_content:
            f.write(tabel_line + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Quantize weight of model and generate .tabel, weight/ and bias/")
    parser.add_argument(
        "--task",
        default='seg',
        type=str,
        help="Frozen model file to import")
    parser.add_argument(
        '--yml',
        default="./quantity/configs.yml",
        dest='yml',
        help="path to the yml file"
    )
    args = parser.parse_args()

    params = {'obj': ["/home/liuzili/workspace/OneDrive_1_2019-5-10/03_benchmark/02_task_specific_benchmarks/02_objdet/jaad_pdet_ssd_tf_graph.pb",
                      '.npy', "/home/liuzili/workspace/OneDrive_1_2019-5-10/obj_data"],
              'seg': ["/home/liuzili/workspace/OneDrive_1_2019-5-10/03_benchmark/02_task_specific_benchmarks/01_semseg/mscoco_fcn_padded_lscape_prediction_tf_graph.pb",
                      '.jpg', "/home/liuzili/workspace/OneDrive_1_2019-5-10/seg_data"],
              'act': ["/home/liuzili/workspace/OneDrive_1_2019-5-10/03_benchmark/02_task_specific_benchmarks/03_actrec/conv_lstm_benchmark_tf_graph.pb",
                      '.npy', "/home/liuzili/workspace/OneDrive_1_2019-5-10/act_data"],
              'seg3': ["/home/liuzili/workspace/OneDrive_1_2019-5-10/seg_dot3.pb",
                       'jpg', "/home/liuzili/workspace/OneDrive_1_2019-5-10/seg_data"],
              'seg4': ["/home/liuzili/workspace/OneDrive_1_2019-5-10/seg_dot4.pb",
                       'jpg', "/home/liuzili/workspace/OneDrive_1_2019-5-10/seg_data"]
              }

    frozen_model_filename, file_type, file_dir = params[args.task]

    graph, graph_def = load_graph(frozen_model_filename)
    print_ops(graph)

    assert os.path.isfile(args.yml), args.yml
    with open(args.yml) as f:
        quantity_cfg = yaml.load(f)

    max_cali_num = quantity_cfg['SETTINGS']['MAX_CALI_IMG_NUM']

    file_names = prepare_data(quantity_cfg, file_type, file_dir)[:max_cali_num]
    activation_quantize(graph, graph_def, quantity_cfg, file_names, args.task)


if __name__ == "__main__":
    main()
