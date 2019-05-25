import argparse
import tensorflow as tf
import cv2
import os
import numpy as np
from collections import OrderedDict

from quantity.tools import load_graph, print_ops, get_params, get_activations, build_net


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen_model_filename",
                        # default="/home/liuzili/workspace/OneDrive_1_2019-5-10/03_benchmark/02_task_specific_benchmarks/02_objdet/jaad_pdet_ssd_tf_graph.pb",
                        # default="/home/liuzili/workspace/OneDrive_1_2019-5-10/03_benchmark/02_task_specific_benchmarks/01_semseg/mscoco_fcn_padded_lscape_prediction_tf_graph.pb",
                        default="/home/liuzili/workspace/OneDrive_1_2019-5-10/03_benchmark/02_task_specific_benchmarks/03_actrec/conv_lstm_benchmark_tf_graph.pb",
                        type=str,
                        help="Frozen model file to import")
    parser.add_argument('--image', help='image path.', default='/data/liuzili/data/coco/val2017/000000089697.jpg')
    parser.add_argument('--gpu', help='use gpu', default='0')

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ['TF_CUDNN_USE_AUTOTUNE'] = '0'

    graph, graph_def = load_graph(args.frozen_model_filename)
    tf.import_graph_def(graph_def, name='graph')
    summaryWriter = tf.summary.FileWriter('log/', graph)
    print_ops(graph)

    # print(graph.get_operation_by_name('conv3_3_reg/BiasAdd').node_def)
    net = build_net(graph_def)

    # get_activations(graph, args.image)
    raise NotImplementedError
    x = graph.get_tensor_by_name('image:0')
    boxes = graph.get_tensor_by_name('conv5_3/Relu:0')

    w = graph.get_operation_by_name("conv3_3_reg/convolution").outputs[0]
    with tf.device('/gpu:0'):
        with tf.Session(graph=graph) as sess:
            # print(sess.run(w))
            img = cv2.imread(args.image, cv2.IMREAD_COLOR)
            img = np.array(cv2.resize(img, (1920, 1080), interpolation=cv2.INTER_CUBIC),
                           dtype=np.float32)[None]
            print(img.dtype)
            pred_boxs = sess.run([boxes], feed_dict={x: img})[0]
            # print(sess.run(w))
            pass

    # with tf.Session(graph=graph) as sess:
    #     img = cv2.imread(args.image, cv2.IMREAD_COLOR)
    #     img = np.array(cv2.resize(img, (1920, 1080), interpolation=cv2.INTER_CUBIC))[None]
    #     pred_boxs = sess.run([boxes], feed_dict={x: img})[0]
    #     # print(pred_boxs)
    #     # print(pred_boxs.shape)

