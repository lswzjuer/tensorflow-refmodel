import argparse
import os
import sys
import os.path as osp
import yaml
from collections import OrderedDict

from quantity.tools import BitReader
from quantity.tools import BiasConverter


def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--yml', dest='yml',
                        default='quantity/configs.yml',
                        help="path to the yml file. the args below are NOT"
                             "needed if yml file is provided.")
    parser.add_argument('--is_pruned', default='false')
    args = parser.parse_args()
    return args


class BiasReWriter:

    def __init__(self,
                 bias_dir,
                 output_bias_dir,
                 weight_file,
                 blob_file):
        """Add the bit info to the prototxt file. The bias and output bit
        is aligned.
        """
        self._weight_file = weight_file
        self._bit_reader = BitReader(weight_table=weight_file, blob_table=blob_file)
        self._bias_converter = BiasConverter(blob_table=blob_file, weight_table=weight_file,
                                             bias_dir=bias_dir, output_bias_dir=output_bias_dir)

    def get_weight_info(self):
        weight_bits, bias_bits = self._bit_reader.get_weight_info()
        return weight_bits, bias_bits

    def get_blob_info(self, is_pruned):
        blob_bits = self._bit_reader.get_blob_info()
        if is_pruned != 'false':
            new_blob_bits = {}
            for k, v in blob_bits.items():
                if (k.startswith('pool') or k.startswith('conv')) and not k.endswith('_conv2d'):
                    new_blob_bits['{}_conv2d'.format(k)] = v
                else:
                    new_blob_bits[k] = v
            blob_bits = new_blob_bits
        return blob_bits

    def bias_output_bit_align(self, bias_bits, blob_bits):
        new_aligned_bits = self._bias_converter.get_new_bias_bit(bias_bits, blob_bits)
        return new_aligned_bits

    def rewrite_bias_dir(self, old_bias_bits, new_bias_bits, is_pruned):
        self._bias_converter.generate_new_bitfile(old_bias_bits, new_bias_bits, is_pruned)

    def rewrite_bias_table(self, old_bias_bits, new_bias_bits):
        with open(self._weight_file, 'r') as f:
            lines = f.readlines()
        new_lines = []
        for line in lines:
            line = line.strip()
            if len(line.split(' ')) > 2:
                new_lines.append(line, '\n')
                continue
            name, bit = line.split(' ')[:2]
            if name[:-5] in old_bias_bits.keys():
                bit = str(new_bias_bits[name[:-5]])
            new_lines.append("{} {}\n".format(name, bit))

        with open(self._weight_file, 'w') as f:
            f.writelines(new_lines)


if __name__ == '__main__':
    args = parse_args()

    assert os.path.isfile(args.yml), args.yml
    with open(args.yml) as f:
        configs = yaml.load(f)

    bias_dir = configs['OUTPUT']['BIAS_DIR']
    output_bias_dir = configs['OUTPUT']['FINAL_BIAS_DIR']
    weight = configs['OUTPUT']['WEIGHT_BIT_TABLE']
    blob = configs['OUTPUT']['FEAT_BIT_TABLE']

    if not osp.exists(output_bias_dir):
        os.makedirs(output_bias_dir)

    rewriter = BiasReWriter(bias_dir, output_bias_dir, weight, blob)
    # weight tabel
    weight_bits, bias_bits = rewriter.get_weight_info()
    blob_bits = rewriter.get_blob_info(args.is_pruned)
    print(bias_bits, '\n', blob_bits)

    empty_set = (set(bias_bits.keys()).difference(set(blob_bits.keys())).union(
        set(blob_bits.keys()).difference(set(bias_bits.keys()))
    ))
    if empty_set != set():
        print("Warning, {} should be empty.".format(empty_set))

    new_aligned_bias = rewriter.bias_output_bit_align(bias_bits, blob_bits)

    # now, bias_bits == blob_bits == new_bias.
    rewriter.rewrite_bias_dir(bias_bits, new_aligned_bias, args.is_pruned)
    rewriter.rewrite_bias_table(bias_bits, new_aligned_bias)
