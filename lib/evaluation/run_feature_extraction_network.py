import tensorflow as tf
import numpy as np
import evaluation.utils.utils as utils
import os.path as osp
from glob import glob
from time import time
from evaluation.obj_det.annotations import split_from_file
from evaluation.obj_det.preprocessing import get_frames
import argparse

MODELS = ('squeezenet', 'mobilenet', 'small_vgg', 'sparsenet')


def get_input_and_output(model, graph):

    if model == 'squeezenet':
        x = graph.get_tensor_by_name('image:0')
        y = graph.get_tensor_by_name('fire9/concat/concat:0')

        def normalize_image(x):
            x = x.astype(np.float32)
            mean = (103.94, 116.78, 123.68)
            x[:, :, 0] = (x[:, :, 0] - mean[0])
            x[:, :, 1] = (x[:, :, 1] - mean[1])
            x[:, :, 2] = (x[:, :, 2] - mean[2])
            return x
        normalize = normalize_image

    elif model == 'mobilenet':
        x = graph.get_tensor_by_name('image:0')
        y = graph.get_tensor_by_name('residual_12/add:0')

        def normalize_image(x):
            x = x.astype(np.float32)
            x = x / 127.5
            x = x - 1.
            return x
        normalize = normalize_image

    elif model == 'small_vgg':
        x = graph.get_tensor_by_name('image:0')
        y = graph.get_tensor_by_name('conv5_3/Relu:0')

        def normalize_image(x):
            x = x.astype(np.float32)
            mean = (73.1574881705, 82.9080206596, 72.3900075701)
            std = (44.906197822, 46.1445214188, 45.3104437099)
            x[:, :, 0] = (x[:, :, 0] - mean[0]) / std[0]
            x[:, :, 1] = (x[:, :, 1] - mean[1]) / std[1]
            x[:, :, 2] = (x[:, :, 2] - mean[2]) / std[2]
            return x

        normalize = normalize_image

    elif model == 'sparsenet':
        x = graph.get_tensor_by_name('image:0')
        y = graph.get_tensor_by_name('activation_40/Relu:0')

        def normalize_image(x):
            x = x.astype(np.float32)
            mean = (115.0, 115.0, 115.0)
            std = (55.0, 55.0, 55.0)
            x[:, :, 0] = (x[:, :, 0] - mean[0]) / std[0]
            x[:, :, 1] = (x[:, :, 1] - mean[1]) / std[1]
            x[:, :, 2] = (x[:, :, 2] - mean[2]) / std[2]
            return x

        normalize = normalize_image
    else:
        raise RuntimeError('Unknown model {}'.format(model))
    return x, y, normalize


def _infer_model(pb_file):
    for model in MODELS:
        if model in pb_file:
            return model
    raise RuntimeError('Cannot infer model from pb_file string.')


def get_video_list(video_dir, split_file):
    video_files = sorted(glob(osp.join(video_dir, '*.mp4')))
    video_idx = split_from_file([split_file])[0]
    return [video_files[i] for i in video_idx]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Load and run a feature extractor network.')
    parser.add_argument('--pb_file', type=str, help='Full path to tensorflow protobuf model file.')
    parser.add_argument('--video_dir', type=str, help='Path to the JAAD video clips (mp4).')
    parser.add_argument('--split_dir', type=str, help='Path to the folder containing the file test_split.txt.')
    parser.add_argument('--model', type=str, help='Name of the model ({}). If not specified the model is tried to be '
                                                  'inferred from the pb_file path.'.format(MODELS))
    parser.add_argument('--skip', type=int, default=4, help='Frame skip in videos (default is 4).')
    args = parser.parse_args()

    if args.model is None:
        model = _infer_model(args.pb_file)
        print('Model set to {}'.format(model))
    else:
        model = args.model

    video_dir = osp.abspath(args.video_dir)
    split_dir = args.split_dir
    split_file = osp.join(split_dir, 'test_split.txt')
    video_files = get_video_list(video_dir, split_file)
    print('Inferring model {}...'.format(model))

    graph = utils.load_graph(args.pb_file)
    with tf.Session(graph=graph) as sess:
        x, y, normalize = get_input_and_output(model, graph)
        t0 = time()
        for vid in video_files:
            frames, _ = get_frames(vid, skip=args.skip)
            print("Processing {} frames of sequence {}...".format(len(frames), osp.basename(vid)))
            for frame in frames:
                sess.run(y, feed_dict={x: normalize(frame)[np.newaxis, ...]})
        t1 = time()
        print('Total runtime for model {}: {:.01f}s\n'.format(model, t1 - t0))
