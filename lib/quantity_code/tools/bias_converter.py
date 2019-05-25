import os
import os.path as osp
import numpy as np
import json
from collections import defaultdict

from quantity.tools import BitReader


def file_name(file_dir):
    files_path = []
    for root, dir, files in os.walk(file_dir):
        for name in files:
            file_path = root + "/" + name
            if not file_path.endswith('.json'):
                continue
            files_path.append(file_path)
    return files_path


class BiasConverter:

    def __init__(self, blob_table, weight_table, bias_dir, output_bias_dir):
        """Since the bias bit and output bit should be same, this converter is used
        to align them and rewrite the bias files.
        """
        self._bias_dir = bias_dir
        self._output_bias_dir = output_bias_dir
        self._bit_reader = BitReader(blob_table, weight_table)

    def get_bias_info(self):
        _, bias_bits = self._bit_reader.get_weight_info()
        return bias_bits

    def get_blob_info(self):
        blob_bits = self._bit_reader.get_blob_info()
        return blob_bits

    def get_new_bias_bit(self, bias_bits, blob_bits):
        # TODO blob bit may be larger than bias bit
        new_aligned_bits = {}
        for key in bias_bits.keys():
            bias_bit, blob_bit = bias_bits[key], blob_bits[key]
            new_aligned_bits[key] = min(bias_bit, blob_bit)
            print("{}: {}, {} => {}".format(key, bias_bit, blob_bit, new_aligned_bits[key]))
        return new_aligned_bits

    def generate_new_bitfile(self, bias_bit, new_bias_bits, is_pruned=False):
        files_path = file_name(self._bias_dir)
        for file_path in files_path:
            assert file_path.endswith('_bias.json')
            layer_name = osp.basename(file_path)[:-10]

            assert layer_name in new_bias_bits.keys(), layer_name
            with open(file_path, 'r') as f:
                lines = json.load(f)

            lines = np.array(lines, dtype=np.float32)
            lines = lines / 2 ** bias_bit[layer_name] * 2 ** new_bias_bits[layer_name]
            lines = np.around(lines).astype(np.int8).tolist()

            output_path = osp.join(self._output_bias_dir, osp.basename(file_path))
            with open(output_path, 'w') as f:
                json.dump(lines, f, indent=4)

        npz_file = np.load('workdir/params.npz', allow_pickle=True)
        new_npz_file = defaultdict(dict)
        for layer_name in npz_file:
            weight = npz_file[layer_name][()]['weights']
            bias = npz_file[layer_name][()]['biases'].astype(np.float32)

            if is_pruned:
                layer_name_key = layer_name.replace('/', '_')
            else:
                layer_name_key = layer_name
            bias = bias / 2 ** bias_bit[layer_name_key] * 2 ** new_bias_bits[layer_name_key]
            bias = np.around(bias).astype(np.int32)

            new_npz_file[layer_name] = {'weights': weight, 'biases': bias}

        np.savez('workdir/new_params.npz', **new_npz_file)
        print("Done!")


if __name__ == '__main__':
    converter = BiasConverter('workdir/feat.table', 'workdir/weight.table',
                              'workdir/bias', 'workdir/new_bias')
    bias_bits = converter.get_bias_info()
    blob_bits = converter.get_blob_info()

    empty_set = (set(bias_bits.keys()).difference(set(blob_bits.keys())).union(
        set(blob_bits.keys()).difference(set(bias_bits.keys()))
    ))
    if empty_set != set():
        print("Warning, {} should be empty.".format(empty_set))

    new_bias_bits = converter.get_new_bias_bit(bias_bits, blob_bits)

    converter.generate_new_bitfile(bias_bits, new_bias_bits)
