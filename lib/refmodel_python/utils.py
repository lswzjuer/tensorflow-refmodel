import tensorflow as tf
import numpy as np
from collections import OrderedDict


def load_graph(pb_file):
    """

    Parameters
    ----------
    pb_file: string
        Path to to tensorflow model protobuf file.

    Returns
    -------
    graph: tf.Graph
        The model graph that can be loaded in a session.

    """
    with tf.gfile.GFile(pb_file, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")
    return graph


def update_confmat(prediction, gt, confmat=None, num_classes=None):
    """
    Confusion matrix between flattened prediction and target.
    Rows correspond to predictions, columns to ground truth classes.

    """
    if num_classes is None and confmat is None:
        raise RuntimeError('Either confmat or num_classes must be specified.')
    if confmat is None:
        confmat = np.zeros([num_classes, num_classes], dtype=np.longlong)
    else:
        assert((confmat.ndim == 2) & (confmat.shape[0] == confmat.shape[1]))
    if num_classes is None:
        num_classes = confmat.shape[0]
    temp_matrix, _, _ = np.histogram2d(prediction.flatten(), gt.flatten(), bins=np.arange(-0.5, num_classes, 1))
    confmat = confmat + np.asarray(temp_matrix, dtype=np.longlong)
    return confmat


def eval_confmat(confmat):
    assert ((confmat.ndim == 2) & (confmat.shape[0] == confmat.shape[1]))
    no_classes = confmat.shape[0]
    class_tp = np.zeros([no_classes], dtype=np.int64)  # true positives     tp/(tp+fn)
    class_fp = np.zeros([no_classes], dtype=np.int64)  # false positives    fp/(tn+fp)
    class_tn = np.zeros([no_classes], dtype=np.int64)  # true negatives     tn/(tn+fp)
    class_fn = np.zeros([no_classes], dtype=np.int64)  # false negatives    fn/(tp+fn)
    pixels = np.sum(confmat.flatten())
    for idx in range(no_classes):
        class_tp[idx] = confmat[idx, idx]
        class_fn[idx] = np.sum(confmat[:, idx]) - class_tp[idx]
        class_fp[idx] = np.sum(confmat[idx, :]) - class_tp[idx]
        class_tn[idx] = pixels - (class_tp[idx] + class_fn[idx] + class_fp[idx])
    class_iou = []  # intersection over union
    class_f1 = []  # F1 score
    class_tpr = []  # true positive rate
    class_tnr = []  # true negative rate
    for i in range(no_classes):
        if class_tp[i] == 0:  # to avoid division by zero
            class_iou.append(np.float32(0))
            class_f1.append(np.float32(0))
            class_tpr.append(np.float32(0))
        else:
            class_iou.append(
                (class_tp[i] / (class_tp[i] + class_fn[i] + class_fp[i])).astype(np.float32))
            class_f1.append(
                (2 * class_tp[i] / (2 * class_tp[i] + class_fp[i] + class_fn[i])).astype(np.float32))
            class_tpr.append((class_tp[i] / (class_tp[i] + class_fn[i])).astype(np.float32))
        if class_tn[i] == 0:
            class_tnr.append(np.float32(0))
        else:
            class_tnr.append((class_tn[i] / (class_tn[i] + class_fp[i])).astype(np.float32))

    pixel_acc = (np.sum(class_tp) / pixels).astype(np.float32)  # pixel accuracy
    class_tp_plus_fn = class_tp + class_fn
    class_tp_plus_fn[class_tp_plus_fn == 0] = 1  # to avoid division by zero
    mean_acc = np.mean(np.divide(class_tp, class_tp_plus_fn.astype(np.float32)))
    eval_out = OrderedDict()
    eval_out.__setitem__("Overall Pixel Accuracy", pixel_acc)
    eval_out.__setitem__("Overall Mean Accuracy", mean_acc)
    eval_out.__setitem__("Intersection over Union", class_iou)
    eval_out.__setitem__("F1-Score", class_f1)
    eval_out.__setitem__("True Positive Rate", class_tpr)
    eval_out.__setitem__("True Negative Rate", class_tnr)
    return eval_out
