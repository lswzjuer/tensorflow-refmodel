import numpy as np
import pickle
import os
import argparse
import utils as utils
import tensorflow as tf
import sys
from base_layer import *


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


def create_all_graph(image):

    # create quantized sub graph

    # reshape (?,?,120,80,3)->(?,120,80,3)
    # image_sh=tf.shape(image)
    image_reshape=tf.reshape(image,[-1,120,80,3])
    print(image_reshape)

    # quantized input image
    image_reshape=qp(image_reshape,7)
    # conv->pool 
    conv1_1=conv(image_reshape,3, 3, 3,16, 1, 1, name='conv1_1', fl_w=9, fl=7, fl_y=6, rs=10, rate=1)
    conv1_2=conv(conv1_1,3, 3,16, 16, 1, 1, name='conv1_2', fl_w=9, fl=6, fl_y=6, rs=9, rate=1)
    pool1=max_pool(conv1_2,2, 2, 2, 2, name='pool1', padding="VALID")
    print(pool1)

    conv2_1=conv(pool1,3, 3,16, 32, 1, 1, name='conv2_1', fl_w=9, fl=6, fl_y=7, rs=8, rate=1)
    conv2_2=conv(conv2_1,3, 3,32, 32, 1, 1, name='conv2_2', fl_w=9, fl=7, fl_y=7, rs=9, rate=1)
    pool2=max_pool(conv2_2,2, 2, 2, 2, name='pool2', padding="VALID")
    print(pool2)

    conv3_1=conv(pool2,3, 3,32, 64, 1, 1, name='conv3_1', fl_w=10, fl=7, fl_y=7, rs=10, rate=1)
    conv3_2=conv(conv3_1,3, 3, 64,64, 1, 1, name='conv3_2', fl_w=10, fl=7, fl_y=7, rs=10, rate=1)
    conv3_3=conv(conv3_2,3, 3, 64,64, 1, 1, name='conv3_3', fl_w=10, fl=7, fl_y=7, rs=10, rate=1)
    pool3=max_pool(conv3_3,2, 2, 2, 2, name='pool3', padding="VALID")
    print(pool3)

    conv4_1=conv(pool3,3, 3,64, 128, 1, 1, name='conv4_1', fl_w=10,fl=7, fl_y=7, rs=10, rate=1)
    conv4_2=conv(conv4_1,3, 3,128, 128, 1, 1, name='conv4_2', fl_w=10, fl=7, fl_y=6, rs=11, rate=1)
    conv4_3=conv(conv4_2,3, 3,128, 128, 1, 1, name='conv4_3', fl_w=10, fl=6, fl_y=6, rs=10, rate=1)
    pool4=max_pool(conv4_3,2, 2, 2, 2, name='pool4', padding="VALID")
    print(pool4) 

    conv5_1=conv(pool4,3, 3,128, 128, 1, 1, name='conv5_1', fl_w=10, fl=6, fl_y=5, rs=11, rate=1)
    conv5_2=conv(conv5_1,3, 3,128, 128, 1, 1, name='conv5_2', fl_w=10, fl=5, fl_y=5, rs=10, rate=1)
    conv5_3=conv(conv5_2,3, 3, 128,128, 1, 1, name='conv5_3', fl_w=10, fl=5, fl_y=4, rs=11, rate=1)
    pool5=max_pool(conv5_3,2, 2, 2, 2, name='pool5', padding="VALID") 
    print(pool5)
    # dequantized
    pool_output=fp(pool5,4)
    print(pool_output)

    return pool_output

def load_sub_graph(model_path):
    graph_def = tf.GraphDef()
    with open(model_path, "rb") as f:
        graph_def.ParseFromString(f.read())
    return graph_def

def load_weights(data_path,sess):

    data_dict = np.load(data_path).item()
    for op_name in data_dict:
            with tf.variable_scope(op_name, reuse=True):
                    for param_name, data in data_dict[op_name].items():
                            try:
                                    #print("op_name: %s param_name: %s\n" %(op_name, param_name))
                                    #print(data)
                                    var = tf.get_variable(param_name)
                                    sess.run(var.assign(data))
                                    #print_msg("assign pretrain model "+param_name+ " to "+op_name,0)
                            except ValueError:
                                    print_msg("ignore "+"Param: "+str(param_name)+" - OpName: "+str(op_name),3)
                                    if not ignore_missing:
                                            raise
    #refModel_log.print_msg("Model was successfully loaded from "+data_path ,3)
    print("Model was successfully loaded ")


if __name__ == '__main__':

    # all the needed path
    actrec_dir=r'/private/liusongwei/03_actrec'
    submodel_path=r'../tensorflow_model/03_actrec/sub_model.pb'
    pb_file=r'../quantized_model/rcnn/act_params_change.npz' 
    result_file=r'../tensorflow_model/03_actrec/reference_result/quantized_result.txt'
    
    print('Action-recognition main folder set to: {}'.format(actrec_dir))
    actrec_data_dir = os.path.join(actrec_dir, 'data')    

    print('Loading test data from {}'.format(actrec_data_dir))
    annotations = load_annotations(actrec_data_dir, 'test')
    annotations = event_recognition_labels(annotations, actrec_data_dir)


    print('Loading graph ')
    with tf.Graph().as_default() as graph:

        image_=tf.placeholder(dtype=tf.float32,shape=[None,None,120,80,3],name='image_input')
        pool_output=create_all_graph(image_)

        # fusion_graph

        graph_def1=load_sub_graph(submodel_path)
        output,=tf.import_graph_def(graph_def1,input_map={'pool5_out:0':pool_output,
                                                   'images:0':image_},
                                                   return_elements=["new_output:0"],name='')

        all_graph_output=tf.identity(output,"all_output")
        print(all_graph_output)

        #load quantized weights and biases
        config = tf.ConfigProto(allow_soft_placement=True)  
        config.gpu_options.per_process_gpu_memory_fraction = 0.7  
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            # load weights and biases quantized
            load_weights(pb_file,sess)

            # run
            print('Running test set evaluation.')
            confmat = np.zeros([2, 2], dtype=np.longlong)
            i=0
            for sample in annotations:
                i+=1
                print('image is : {}/{}'.format(i,len(annotations)))
                image, action_labels = load_sample(sample)
                image=preprocess_input(image)

                # do sw inference
                pred_score=sess.run(all_graph_output,feed_dict={image_:image})
                print(pred_score)

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