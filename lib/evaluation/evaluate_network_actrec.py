import numpy as np
import pickle
import os
import argparse
import utils.utils as utils
import tensorflow as tf
import sys

def add_path(path):
	"""
	This function adds path to python path. 
	"""
	if path not in sys.path:
		sys.path.insert(0,path)

lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../lib'))
add_path(lib_path)

from networks.rcnnnet import RCNN


EXTENSION = '.npy'
LABEL_SCHEME = {'Background': 0, 'standing': 0, 'walking': 1, 'running': 1}

def load_annotations(data_dir, _type='test'):
    """
    Load the JAAD preprocessed annotation from pickle files.

    Parameters
    ----------
    data_dir: string
        Path to the JAAD data folder.

    _type: string
        train, val or test.

    Returns
    -------
    annotations: dict
        The preprocessed annotations as dictionary.

    """
    with open(os.path.join(data_dir, _type + '.pickle'), "rb") as f:
        annotations = pickle.load(f)
    return annotations


def event_recognition_labels(annotations, data_dir, extension=EXTENSION, label_scheme=LABEL_SCHEME):
    """
    Labels for JAAD event recognition.

    Parameters
    ----------
    annotations: dict
        Annotations dictionary (see load_annotations).
    data_dir: string
        Path to the JAAD preprocessed data.

    Returns
    -------
    annotations_list: list
        List of dictionary, where each entry corresponds to a single sample.

    """

    annotations_list = []
    video_name_list = list(annotations["image_data"].keys())

    for video_name in video_name_list:
        video_frames = annotations["image_data"][video_name]["frames"]
        samples = [list(range(1, len(video_frames)))]

        for sample in samples:
            video_annotation = dict()
            video_annotation["video_name"] = video_name
            video_annotation["path"] = data_dir
            video_annotation["file"] = os.path.join(data_dir, video_name + extension)
            video_annotation["sample_id"] = [s for s in sample]
            video_annotation["length"] = len(video_frames)

            s = list(sample)
            last_frame = s[-1]
            first_frame = s[0]
            annotation_idx = annotations["image_data"][video_name]["annotation_id"][last_frame]
            annotation = annotations["annotations"][annotation_idx]
            video_idx = annotation["frame_id"][0]
            video_annotation["metadata"] = annotations["meta_data"][video_idx]
            video_annotation["sample_length"] = last_frame - first_frame
            for k in annotation["labels"].keys():
                video_annotation["sample_" + k] = []
                video_annotation["sample_action_label"] = []
            for frame in s:
                ann_idx = annotations["image_data"][video_name]["annotation_id"][frame]
                for k in annotation["labels"].keys():
                    video_annotation["sample_" + k].append(annotations['annotations'][ann_idx]["labels"][k])
                video_annotation["sample_action_label"].append(label_scheme[video_annotation["sample_action"][-1]])

            annotations_list.append(video_annotation)
    return annotations_list


def load_sample(sample):
    """
    Load input and labels for a single sample.

    Parameters
    ----------
    sample: dict
        Dictionary with input file and label data for a single sample (corresponds to one element of the output list
        from event_recognition_labels)

    Returns
    -------
    image: np.ndarray
        Input image sequence of shape (time_steps, height, width, 3)
    labels: np.ndarray
        Binary target labels for the input sequence of shape (time_steps, )

    """

    metadata = sample['metadata']
    memmapfile = np.memmap(sample['file'], dtype=metadata['dtype'], mode="r", shape=metadata['shape'])
    image = np.array(memmapfile[sample['sample_id']])
    action_labels = np.asarray(sample['sample_action_label'])
    return image, action_labels


def preprocess_input(img):
    """
    Input image preprocessing.

    Parameters
    ----------
    img: np.ndarray
        Input image sequence of shape (time_steps, height, width, 3) or (batch_size, time_steps, height, width, 3)

    Returns
    -------
    img: np.ndarray
        Preprocessed image for the action-recognition model.

    """
    img = img.astype(np.float32)
    img = (img - 127.5) / 127.5
    if img.ndim < 5:  # add batch dimension if necessary
        img = img[np.newaxis, :]
    return img


def _result_to_file(out_path, eval, confmat):
    with open(out_path, 'a') as file:
        print('Storing output in {}.'.format(os.path.realpath(file.name)))
        for k, v in eval.items():
            print('{}: {}'.format(k, v), file=file)
        print('\nConfusion matrix:\n', file=file)
        print(confmat, file=file)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Test the action-recognition network on the JAAD dataset.')
    parser.add_argument('--actrec_dir', type=str, help='Path to main folder of the action-recognition network. '
                                                       'Default: ../02_task_specific_benchmark/03_actrec',
                        default=r'/private/liusongwei/03_actrec')

    parser.add_argument('--pb_file', type=str, help='Full path to .npz  model file. ')
    parser.add_argument('--sub_model', type=str, help='sub model file. ')   
    parser.add_argument('--result_file', type=str, default=None, help='Stores results in result file.')
    args = parser.parse_args()

    actrec_dir = args.actrec_dir
    print('Action-recognition main folder set to: {}'.format(actrec_dir))
    actrec_data_dir = os.path.join(actrec_dir, 'data')

    sub_model_path=args.sub_model

    print('Loading test data from {}'.format(actrec_data_dir))
    annotations = load_annotations(actrec_data_dir, 'test')
    annotations = event_recognition_labels(annotations, actrec_data_dir)

    print('Loading graph ')
    with tf.Graph().as_default() as graph:

        net = RCNN(is_Hardware=True)
        output=net.fusion_graph(sub_model_path)

        #load quantized weights and biases
        config = tf.ConfigProto(allow_soft_placement=True)  
        config.gpu_options.per_process_gpu_memory_fraction = 0.7  
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            # load weights and biases quantized
            net.load(args.pb_file,sess)

            # run
            print('Running test set evaluation.')
            confmat = np.zeros([2, 2], dtype=np.longlong)
            i=0
            for sample in annotations:
                i+=1
                print("image is : {}/{}".format(i,len(annotations)))
                image, action_labels = load_sample(sample)
                image=preprocess_input(image)

                # do sw inference
                pred_score=net.run_sw_demo(sess,image,output)

                #pred_score = sess.run(y, feed_dict={x: preprocess_input(image)})
                pred_label = pred_score.argmax(axis=2)
                confmat = utils.update_confmat(pred_label, action_labels, confmat)

            result = utils.eval_confmat(confmat)

            print(result)
            print(confmat)

            result_file = args.result_file
            if result_file is not None:
                print('S results at {}.'.format(result_file))
                _result_to_file(result_file, result, confmat)

            model_path=r'../tensorflow_model/03_actrec/test_all_model.pb'
            net.save_model(sess,model_path)
