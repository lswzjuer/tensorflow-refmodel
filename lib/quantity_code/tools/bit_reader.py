class BitReader:

    def __init__(self,
                 blob_table=None,
                 weight_table=None):
        self._blob_table = blob_table
        self._weight_table = weight_table
        self._prefix = None

    @property
    def prefix(self):
        return self._prefix

    def get_blob_info(self):
        assert self._blob_table

        blob_bits = {}
        count_blob = 0
        with open(self._blob_table, 'r') as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            layer_name, bit = line.split(' ')[:2]
            layer_name_list = layer_name.split('/')
            if len(layer_name_list) > 2 and 'VGG' in layer_name:
                prefix = '/'.join(layer_name_list[:-2])
                assert self._prefix is None or self._prefix == prefix, \
                    "{}, {}".format(self._prefix, prefix)
                self._prefix = prefix
                layer_name_list = layer_name_list[-2:]

            layer_name = layer_name_list[0]
            blob_bits[layer_name] = int(eval(bit))
            count_blob += 1
        print("blob count:", count_blob)
        return blob_bits

    def get_weight_info(self):
        assert self._weight_table

        weight_bits, bias_bits = {}, {}
        count_weight, count_bias = 0, 0

        with open(self._weight_table, 'r') as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip()
            contents = line.split(' ')
            layer_name, bit = contents
            bit = int(eval(bit))

            if layer_name.endswith('_kernel'):
                layer_name = layer_name[:-7]
                count_weight += 1
                weight_bits[layer_name] = bit
            elif layer_name.endswith('_bias'):
                layer_name = layer_name[:-5]
                count_bias += 1
                bias_bits[layer_name] = bit
            else:
                print("Unknow layer name {}".format(layer_name))

        print("weight count:", count_weight)
        print("bias count:", count_bias)
        return weight_bits, bias_bits
