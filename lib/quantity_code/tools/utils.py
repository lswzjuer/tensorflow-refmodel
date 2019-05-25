import tensorflow as tf
import numpy as np
import cv2
import os
from collections import OrderedDict, Counter
from copy import deepcopy
import skimage.transform


def load_graph(frozen_graph_filename):
    # We parse the graph_def file
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # We load the graph_def in the default graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name='',
            op_dict=None,
            producer_op_list=None
        )
    return graph, graph_def


def print_ops(graph):
    #get all the operations in tensorflow graph
    for op in graph.get_operations():
        print(op.name, op.values(), op.type)


def get_params(graph, task_type):
    names = []
    for op in graph.get_operations():
        if op.name.endswith('/kernel') or op.name.endswith('/bias'):
            if task_type == 'act' and 'fc6' in op.name:
                # ignore the terrifying part of act-net
                break
            names.append(op.name)

    named_params = {}
    named_shape = {}
    named_type = {}
    with tf.Session(graph=graph) as sess:
        for name in names:
            op = graph.get_operation_by_name(name)
            named_params[name] = sess.run(op.outputs[0])
            named_shape[name] = op.values()[0].shape.as_list()
            named_type[name] = op.type

    return named_params, named_shape, named_type


def build_net(graph_def):
    """
    We need the connection relationship of layers.
    """
    cared_op_type = ['Placeholder', 'Conv2D', 'BiasAdd', 'Relu', 'MaxPool', 'ConcatV2',
                     'Reshape', 'ResizeBilinear', 'Add', 'MatMul']

    net = OrderedDict()
    for i, n in enumerate(graph_def.node):
        if not n.op in cared_op_type:
            continue
        inputs = []
        for inp in n.input:
            if not (inp.endswith('/read') or inp.endswith('/shape') or inp.endswith('/axis') or\
                    inp.endswith('/truediv') or inp.endswith('/size')):
                inputs.append(inp)
        net[n.name] = {'inputs': inputs, 'type': n.op}

    return net


def prune_net_info(net_info, keep_node_list):

    all_op_names = set(list(net_info.keys()))
    prune_op_names = all_op_names - set(keep_node_list)
    keys = list(net_info.keys())[::-1]
    net_info_rev, net_info_new = OrderedDict(), OrderedDict()
    for k in keys:
        net_info_rev[k] = net_info[k]

    def _dfs(name):
        """
        Start from prune op, the input length must be 1.
        """
        info = net_info_rev[name]
        assert len(info['inputs']) <= 1, (info['inputs'], name)
        if len(info['inputs']) == 0:
            return None

        inp = info['inputs'][0]

        if inp in prune_op_names:
            return _dfs(inp)
        else:
            return inp

    for name, info in net_info_rev.items():
        if name in prune_op_names:
            continue

        for ind in range(len(info['inputs'])):
            inp = info['inputs'][ind]
            if inp in prune_op_names:
                info['inputs'][ind] = _dfs(inp)

    for k in net_info.keys():
        if k not in prune_op_names:
            net_info_new[k] = net_info_rev[k]

    return net_info_new


def get_merge_groups(net):
    
    merge_ops = ['ConcatV2', 'Add']
    cared_op_type = ['BiasAdd', 'ResizeBilinear', 'MatMul']

    merge_layer = []
    for name, info in net.items():
        if info['type'] in merge_ops:
            merge_layer.append(name)

    merge_layer.reverse()
    print('merge layers:', merge_layer)

    vis = []

    def _dfs(name):
        if name in vis:
            return []
        vis.append(name)

        info = net[name]
        bottoms = info['inputs']
        names = []

        if len(bottoms) == 0:
            return []

        for bottom in bottoms:
            if net[bottom]['type'] not in cared_op_type:
                names.extend(_dfs(bottom))
            else:
                names.append(bottom)
        return names

    merge_groups = []
    for layer_name in merge_layer:
        b_names = _dfs(layer_name)
        print(layer_name, b_names)
        if b_names:
            merge_groups.append(b_names)

    return merge_groups


def get_activations(graph, img_path, task_type):
    cared_op_type = ['BiasAdd', 'ConcatV2', 'Add', 'ResizeBilinear']

    names = []
    named_op = OrderedDict()
    type_counter = []
    for op in graph.get_operations():
        type_counter.append(op.type)
        if op.type in cared_op_type:
            if task_type == 'act' and 'fc6' in op.values()[0].name:
                # ignore the terrifying part of act-net
                break
            names.append(op.values()[0].name)

    # print(Counter(type_counter))

    for name in names:
        named_op[name] = graph.get_tensor_by_name(name)

    name = 'image' if task_type != 'act' else 'images'
    x = graph.get_tensor_by_name('{}:0'.format(name))

    with tf.device('/gpu:0'):
        with tf.Session(graph=graph) as sess:
            if img_path.endswith('.jpg'):
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                img = preprocess(img, task_id='seg')
            elif img_path.endswith('.npy'):
                img = np.load(img_path, allow_pickle=True)
            else:
                raise NotImplementedError
            feats = sess.run(list(named_op.values()), feed_dict={x: img})

    named_feat = {name: img}
    for name, feat in zip(names, feats):
        if name.endswith(':0'):
            name = name[:-2]
        named_feat[name] = feat

    return named_feat


def preprocess(img, task_id):
    if task_id == 'seg':
        mean = (73.1574881705, 82.9080206596, 72.3900075701)
        std = (44.906197822, 46.1445214188, 45.3104437099)
        pad_shape = (480, 640)
        input_shape = (1080, 1920)

        img = np.array(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), dtype=np.float32)
        img = normalize(img, mean, std)
        img = pad_or_crop(img, output_shape=pad_shape, pad_value=0)
        img = rescale_image(img, list(input_shape) + [3]).astype(img.dtype)[None]
        img = np.pad(img, ((0, 0), (0, 8), (0, 0), (0, 0)), mode='constant')
    else:
        raise NotImplementedError
    return img


def normalize(x, mean=(115.0, 115.0, 115.0), std=(55.0, 55.0, 55.0)):
    x = x.astype(np.float32)
    x[:, :, 0] = (x[:, :, 0] - mean[0]) / std[0]
    x[:, :, 1] = (x[:, :, 1] - mean[1]) / std[1]
    x[:, :, 2] = (x[:, :, 2] - mean[2]) / std[2]
    return x


def pad_or_crop(x, output_shape=(480, 640), pad_value=0):
    output_shape = tuple(output_shape)
    x_shape = x.shape[:2]
    if x_shape == output_shape:
        return x
    diff = np.asarray(output_shape) - np.asarray(x_shape)
    crop = np.maximum(- diff // 2, 0)
    crop = np.stack((crop, np.maximum(- diff - crop, 0)), axis=1)
    if np.any(crop):
        crop[:, 1] = x_shape - crop[:, 1]
        crop = tuple(crop)
        x = x[crop[0][0]:crop[0][1], crop[1][0]:crop[1][1], ...]

    pad = np.maximum(diff // 2, 0)
    pad = np.stack((pad, np.maximum(diff - pad, 0)), axis=1)
    if np.any(pad > 0):
        x = np.pad(x, tuple(pad) + ((0, 0),) * (x.ndim - 2), 'constant', constant_values=pad_value)
    return x


def rescale_image(x, output_shape, order=1):
    output_shape = list(output_shape)
    x = skimage.transform.resize(x, output_shape, order=order, preserve_range=True, mode='reflect', anti_aliasing=False)
    return x
