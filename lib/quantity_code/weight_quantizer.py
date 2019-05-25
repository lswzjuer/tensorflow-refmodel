import argparse
import numpy as np
import math
import os
import json
import yaml
from collections import defaultdict
from termcolor import colored

from quantity.tools.distribution_collector import DistributionCollector
from quantity.tools.quantizer import Quantizer
from quantity.tools import load_graph, print_ops, get_params


def init_dir(cfg):
    weight_dir = cfg['OUTPUT']['WEIGHT_DIR']
    bias_dir = cfg['OUTPUT']['BIAS_DIR']

    if not os.path.exists(weight_dir):
        os.makedirs(weight_dir)

    if not os.path.exists(bias_dir):
        os.makedirs(bias_dir)


def weight_quantize(model, cfg, task_type):
    interval_nun = cfg['SETTINGS']['INTERVAL_NUM']
    statistic = cfg['SETTINGS']['STATISTIC']
    worker_num = cfg['SETTINGS']['WORKER_NUM']
    support_dilation = cfg['SETTINGS']['SUPPORT_DILATION']
    weight_dir = cfg['OUTPUT']['WEIGHT_DIR']
    bias_dir = cfg['OUTPUT']['BIAS_DIR']
    table_file = cfg['OUTPUT']['WEIGHT_BIT_TABLE']

    params, kernel_shapes, layer_types = get_params(model, task_type)
    param_names = list(params.keys())
    param_shapes = {}
    for name, param in params.items():
        param_shapes[name] = param.shape

    table_file_content = []
    npz_file = defaultdict(dict)

    for name, param in params.items():
        params[name] = params[name].flatten()

    collector = DistributionCollector(param_names, interval_num=interval_nun,
                                      statistic=statistic, worker_num=worker_num)
    quantizer = Quantizer(param_names, worker_num=worker_num)

    collector.refresh_max_val(params)
    print(colored('max vals:', 'green'), collector.max_vals)

    collector.add_to_distributions(params)
    quantizer.quantize(collector.distributions, collector.distribution_intervals)

    for name, bit in quantizer.bits.items():
        param_quantity = np.around(params[name] * math.pow(2, bit))
        param_quantity = np.clip(param_quantity, -128, 127)
        param_file_name = name.replace('/', '_')

        table_line = param_file_name + " " + str(bit)
        table_file_content.append(table_line)

        content = param_quantity.reshape(param_shapes[name]).astype(np.int32)
        layer_name = '/'.join(name.split('/')[:-1])
        weight_type = name.split('/')[-1]
        if weight_type == 'kernel':
            npz_file[layer_name]['weights'] = content
        elif weight_type == 'bias':
            npz_file[layer_name]['biases'] = content
        else:
            raise NotImplementedError(name)

        content = param_quantity.reshape(param_shapes[name]).astype(np.int32).tolist()
        if name.endswith('kernel'):
            with open(os.path.join(weight_dir, param_file_name + '.json'), 'w') as file:
                json.dump(content, file, indent=4)
        elif name.endswith('bias'):
            with open(os.path.join(bias_dir, param_file_name + '.json'), 'w') as file:
                json.dump(content, file, indent=4)
        else:
            raise NotImplementedError(name)

    np.savez('workdir/params.npz', **npz_file)

    with open(table_file, 'w') as f:
        for tabel_line in table_file_content:
            f.write(tabel_line + "\n")


def dilation_to_zero_padding(tensor, dilation):
    out_channel, in_channel = tensor.shape[:2]
    assert tensor.shape[2] == tensor.shape[3] and dilation == (2, 2), "Not support."

    kernel_size = tensor.shape[2]
    new_kernel_size = kernel_size * 2 - 1
    new_tensor = np.zeros((out_channel, in_channel, new_kernel_size, new_kernel_size),
                          dtype=np.float32)
    source_loc = np.array(range(kernel_size))
    trans_loc = source_loc * 2

    for x, new_x in zip(source_loc, trans_loc):
        for y, new_y in zip(source_loc, trans_loc):
            new_tensor[..., new_x, new_y] = tensor[..., x, y]
    return new_tensor


def main():
    parser = argparse.ArgumentParser(
        description="Quantize weight of model and generate .table, weight/ and bias/")
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

    params = {'obj': "/home/liuzili/workspace/OneDrive_1_2019-5-10/03_benchmark/02_task_specific_benchmarks/02_objdet/jaad_pdet_ssd_tf_graph.pb",
              'seg': "/home/liuzili/workspace/OneDrive_1_2019-5-10/03_benchmark/02_task_specific_benchmarks/01_semseg/mscoco_fcn_padded_lscape_prediction_tf_graph.pb",
              'act': "/home/liuzili/workspace/OneDrive_1_2019-5-10/03_benchmark/02_task_specific_benchmarks/03_actrec/conv_lstm_benchmark_tf_graph.pb",
              'seg3': "/home/liuzili/workspace/OneDrive_1_2019-5-10/seg_dot3.pb",
              'seg4': "/home/liuzili/workspace/OneDrive_1_2019-5-10/seg_dot4.pb"}

    frozen_model_filename = params[args.task]

    graph, _ = load_graph(frozen_model_filename)
    print_ops(graph)

    assert os.path.isfile(args.yml), args.yml
    with open(args.yml) as f:
        quantity_cfg = yaml.load(f)

    init_dir(quantity_cfg)
    weight_quantize(graph, quantity_cfg, args.task)


if __name__ == "__main__":
    main()
