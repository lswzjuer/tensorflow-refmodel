import tensorflow as tf
import argparse
import os
import numpy as np
import skimage.transform
import cv2
import sys

def add_path(path):
	"""
	This function adds path to python path. 
	"""
	if path not in sys.path:
		sys.path.insert(0,path)

lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
add_path(lib_path)

import utils.utils as utils



os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'

MEAN = (73.1574881705, 82.9080206596, 72.3900075701)
STD = (44.906197822, 46.1445214188, 45.3104437099)

NEW_CLASS = np.zeros(100, dtype=np.int16)
NEW_CLASS[1] = 1
NEW_CLASS[3] = 2
NEW_CLASS[4] = 3
NEW_CLASS[5] = 4
NEW_CLASS[6] = 5
NEW_CLASS[7] = 6
NEW_CLASS[8] = 7
NEW_CLASS[[17, 18, 19, 20, 21, 22, 23, 24, 25]] = 8

CLASS_COLOR = list()
CLASS_COLOR.append([0, 0, 0])
CLASS_COLOR.append([220, 20, 60])
CLASS_COLOR.append([0, 0, 142])
CLASS_COLOR.append([0, 0, 230])
CLASS_COLOR.append([0, 140, 0])
CLASS_COLOR.append([0, 60, 100])
CLASS_COLOR.append([0, 80, 100])
CLASS_COLOR.append([0, 0, 70])
CLASS_COLOR.append([200, 60, 20])

CLASS_NAME = ("background", "person", "car", "motorcycle", "airplane", "bus", "train", "truck", "animal")


def rescale_image(x, output_shape, order=1):
    output_shape = list(output_shape)
    x = skimage.transform.resize(x, output_shape, order=order, preserve_range=True, mode='reflect', anti_aliasing=False)
    return x


def normalize(x, mean=MEAN, std=STD):
    x = x.astype(np.float32)
    x[:, :, 0] = (x[:, :, 0] - mean[0]) / std[0]
    x[:, :, 1] = (x[:, :, 1] - mean[1]) / std[1]
    x[:, :, 2] = (x[:, :, 2] - mean[2]) / std[2]
    return x


def convert_label(y, new_class=NEW_CLASS):
    return new_class[y]


def index_to_color(y, class_color=CLASS_COLOR):
    return class_color[y]


def arg_max(x, axis=1):
    return np.argmax(x, axis=axis)


def parse_filenames(file):
    with open(file, 'r') as f:
        image_names = f.readlines()
    image_names = [x.strip("\n") for x in image_names]
    image_names = [os.path.splitext(x)[0] for x in image_names]
    return image_names


def load_sample(images_dir, labels_dir, f_name):

    full_path = os.path.join(images_dir, f_name + '.jpg')
    image = cv2.imread(full_path, -1)
    if image is None:
        raise RuntimeError('Error loading file {}.'.format(full_path))
    
    if len(image.shape) == 3:
        image = np.flip(image, 2)  # to RGB
    else:
        image = image[:, :, np.newaxis]
        image = np.repeat(image, 3, axis=-1)

    label = cv2.imread(os.path.join(labels_dir, f_name + '_gtClasses.png'), -1)
    return image, label


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


def preprocess_for_val(image, label, pad_shape=(480, 640), input_shape=(1080, 1920), label_shape=(1080, 1920)):
    """
    Normalizes images, pads/crops both image and label to pad_shape and then rescales image to input_shape and label to
    label_shape.
    """

    image = normalize(image)
    label = convert_label(label)  # do this before the padding which could introduce -1 labels
    image = pad_or_crop(image, output_shape=pad_shape, pad_value=0)
    label = pad_or_crop(label, output_shape=pad_shape, pad_value=-1)
    if input_shape is not None:
        image = rescale_image(image, list(input_shape) + [3]).astype(image.dtype)
    if label_shape is not None:
        label = rescale_image(label, list(label_shape), order=0).astype(label.dtype)
    image = image[np.newaxis, ...]
    return image, label


def postprocess_output_val(pred, pred_shape=(272, 480), rescale_shape=(1080, 1920)):
    """
    Get output class prediction, reshape prediction to spatial matrix and rescale if required.
    """

    pred = arg_max(pred, axis=-1)
    pred = pred.reshape(pred_shape)
    if rescale_shape is not None:
        pred = rescale_image(pred, output_shape=rescale_shape, order=0)
    return pred



def _get_val_list(val_file, images_dir, labels_dir, out_file=None):
    """

    Identify landscape images that have non-background labels as well.
    """

    image_names = parse_filenames(val_file)
    shp = []
    classes_in_im = []
    for im in image_names:
        image, label = load_sample(images_dir, labels_dir, im)
        label = convert_label(label)
        shp.append(image.shape)
        classes_in_im.append(np.setdiff1d(label, 0))
    is_landscape = [x[0] < x[1] for x in shp]
    has_nb_label = [x.size > 0 for x in classes_in_im]
    landscape_and_nb = np.logical_and(np.asarray(is_landscape), np.asarray(has_nb_label))
    val_names = [x for idx, x in enumerate(image_names) if landscape_and_nb[idx]]

    print('Num landscape images: {}'.format(np.sum(np.asarray(is_landscape))))
    print('Num images with non-background labels: {}'.format(np.sum(np.asarray(has_nb_label))))
    print('Num landscape images with non-background label: {}'.format(np.sum(landscape_and_nb)))

    if out_file:
        with open(out_file, 'x') as file:
            for im in val_names:
                file.write(im + '.jpg\n')
    return val_names


def _get_normalization_mean_std(train_file, images_dir, labels_dir):
    image_names = parse_filenames(train_file)
    mean = np.zeros(3, np.float64)
    std = np.zeros(3, np.float64)
    shp = []
    for name in image_names:
        image, _ = load_sample(images_dir, labels_dir, name)
        image = image.astype(np.float64)
        mean = mean + image.sum(axis=(0, 1))
        std = std + np.sum(image ** 2, axis=(0, 1))
        shp.append(np.prod(image.shape[:2]))
    N = np.sum(shp)
    mean = mean / N
    std = np.sqrt(std / N - (mean ** 2))
    return mean, std


def _result_to_file(out_path, eval, confmat):
    with open(out_path, 'a') as file:
        print('Storing output in {}.'.format(os.path.realpath(file.name)))
        for k, v in eval.items():
            print('{}: {}'.format(k, v), file=file)
        print('\nConfusion matrix:\n', file=file)
        print(confmat, file=file)

def load_sub_graph(model_path):
    graph_def = tf.GraphDef()
    with open(model_path, "rb") as f:
        graph_def.ParseFromString(f.read())
    return graph_def

def quantize_input_image(image,bit_frac):
    sh = image.shape
    image = image.flatten()
    for i in range(len(image)):
        sign=1
        if image[i]<0:
            sign=-1
        image[i]=sign*round(abs(image[i])*pow(2,bit_frac))
    image = image.reshape(sh)
    return image


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Test the semantic segmentation network on a subset of the COCO '
                                                 'dataset.')
    parser.add_argument('--img_dir', type=str, help='Path to coco validation images.', default=None)
    parser.add_argument('--labels_dir', type=str, help='Path to ground truth label images (png).', default=None)
    parser.add_argument('--file_list', type=str, help='Path to file with list of validation images to use.',
                        default=None)
    parser.add_argument('--pb_file', type=str, help='Full path to quantized model file',  default=None)
    parser.add_argument('--result_file', type=str, help='Stores results in result file.', default=None)
    #parser.add_argument('--padded_lscape', action='store_true', help='Evaluation for the graph without padding/cropping')
                                                                     
    args = parser.parse_args()

    model_dir=args.pb_file

    images_dir = args.img_dir
    labels_dir = args.labels_dir
    print('Image directory set to: {}'.format(images_dir))
    print('Label image directory set to: {}'.format(labels_dir))

    file_list = args.file_list
    print('Using evaluation files from list {}'.format(file_list))
    image_names = parse_filenames(file_list)

    print('Running evaluation on {} images.'.format(len(image_names)))
    confmat = np.zeros((9, 9), dtype=np.longlong)
    
    fp_module_path=r'/home/liusongwei/tf-reference-model/lib/fp/fp.so'
    _fp_module = tf.load_op_library(fp_module_path)

    sp_module_path=r'/home/liusongwei/tf-reference-model/lib/saturate/sp.so'
    _sp_module = tf.load_op_library(sp_module_path)
    
    lp_module_path=r'/home/liusongwei/tf-reference-model/lib/lp/lp.so'
    _lp_module = tf.load_op_library(lp_module_path)
    
    print('Loading graph ')
    with tf.Graph().as_default() as graph:

        input_=tf.placeholder(dtype=tf.float32,name="new_input")
        graph_def_=load_sub_graph(model_dir)
        output,=tf.import_graph_def(graph_def_,input_map={'image_input_node/Identity:0':input_},
                                                            return_elements=['new_ground_truth/Reshape:0'])

        new_output=tf.identity(output,name='new_output')
        print(new_output)

        config = tf.ConfigProto(allow_soft_placement=True)  
        config.gpu_options.per_process_gpu_memory_fraction = 0.6 
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:

            if True:
                pred_shape = [272, 480]
            else:
                pred_shape = [270, 480]
            num=0

            for file in image_names[:2000]:
                num+=1
                print("the image nums : {}/{}".format(num,len(image_names[:2000])))
                
                image, label = load_sample(images_dir, labels_dir, file)
                # keshihua
                image, label = preprocess_for_val(image, label)
                # keshihua

                if True:
                    image = np.pad(image, ((0, 0), (0, 8), (0, 0), (0, 0)), mode='constant')
                    
                # do sw inference
                quantized_image=quantize_input_image(image,4)
                pred=sess.run(new_output,feed_dict={input_:quantized_image})

                print(pred.shape)

                # if args.padded_lscape:
                #     pred_pp = pred.reshape(pred_shape + [9, ])[:270, :, :].reshape([-1, 9])

                pred_pp = postprocess_output_val(pred, pred_shape=pred_shape)
                confmat = utils.update_confmat(pred_pp, label, confmat=confmat)
                
            # save test results
            eval = utils.eval_confmat(confmat)
            print(eval)
            print(confmat)
            result_file = args.result_file
            if result_file is not None:
                _result_to_file(result_file, eval, confmat)

            #model_path=r'../../tensorflow_model/01_semseg/new_run_all_model.pb'
            #net.save_model(sess,model_path)

