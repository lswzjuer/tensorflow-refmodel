import numpy as np
import utils.utils as utils
import pickle
import tensorflow as tf
from obj_det.preprocessing import get_frames, normalize
from obj_det.postprocessing import get_bboxes, load_anchors
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import os.path as osp
import argparse
import sys
import os 

def add_path(path):
    """
    This function adds path to python path. 
    """
    if path not in sys.path:
        sys.path.insert(0,path)

lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../lib'))
add_path(lib_path)

from networks.ssdnet import SSD


def _evaluate_predictions(annotations_dir, results):
    """
    Calculate the log-average miss rate for the provided results on the test set.
    """
    gt = COCO(osp.join(annotations_dir, 'test.json'))
    dt = gt.loadRes(results)
    coco_eval = COCOeval(gt, dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    lamr = coco_eval.calculate_lamr()
    params = {'hRng': coco_eval.params.hRng[0], 'vRng': coco_eval.params.vRng[0],
              'iouThr': coco_eval.params.iouThrs[0], 'maxDets': coco_eval.params.maxDets[2],
              'areaRng': coco_eval.params.areaRng[0]}
    return lamr, params


def _result_to_file(out_path, lamr, params):
    with open(out_path, 'a') as file:
        print('Storing output in {}.'.format(osp.realpath(file.name)))
        for k, v in params.items():
            print('{}: {}'.format(k, v), file=file)
        print('Log-average miss rate: {}'.format(lamr), file=file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate the object detection network.')
    parser.add_argument('--annotations_dir', type=str, help='Path to the preprocessed annotations.')
    parser.add_argument('--pb_file', type=str, help='Full path to .npz file.')
    parser.add_argument('--sub_model', type=str, help='Full path to submodel which be fused.')   
    parser.add_argument('--anchor_file', type=str, help='Full path to model anchor file.')
    parser.add_argument('--skip', type=int, default=4, help='Frame skip in videos (default is 4 for evaluation).')
    parser.add_argument('--result_file', type=str, default=None, help='Stores results in result file.')
    args = parser.parse_args()

    nms_params = dict(thr=0.7, top_k=125, mode="max")

    ann_dir = args.annotations_dir
    with open(osp.join(ann_dir, 'test.pckl'), 'rb') as file:
        annotations = pickle.load(file)

    anchors = load_anchors(args.anchor_file)

    skip = args.skip
    sub_model_path=args.sub_model
    model_dir=args.pb_file

    print('Loading graph ')
    with tf.Graph().as_default() as graph:

        # load graph and fusion sub_graph
        net=SSD(is_Hardware=True)
        out_put=net.fusion_graph(sub_model_path)
        results=[]
        
        config = tf.ConfigProto(allow_soft_placement=True)  
        config.gpu_options.per_process_gpu_memory_fraction = 0.6 
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:

            # load quantized weight and biases
            net.load(model_dir,sess)

            fid = 0
            for vcount, vid in enumerate(annotations[:10]):
                print('Processing sequence {}... {}/{}'.format(vid['name'],vcount,len(annotations)))
                frames, _ = get_frames(vid['full_path'], skip=skip)
                i=0
                for frame in frames:
                    i+=1
                    # do sw inference
                    print("image is :{}/{}".format(i,len(frames)))
                    image=normalize(frame)[np.newaxis, ...]
                    
                    pred=net.run_sw_demo(sess,image,out_put)
                    
                    print(pred.shape)

                    boxes, scores = get_bboxes(np.squeeze(pred), anchors, nms_params)
                    for j in range(0, len(boxes)):
                        box = boxes[j].tolist()
                        results.append(
                            dict(image_id=fid, category_id=1, bbox=box, score=scores[j].tolist(), height=box[3]))
                    if not len(boxes):  # if there are no detections
                        results.append(dict(image_id=fid, category_id=0, bbox=[0, 0, 0, 0], score=0))
                    fid += 1

            lamr, params = _evaluate_predictions(ann_dir, results)
            print('Log-average miss rate: {}'.format(lamr))

            result_file = args.result_file
            if result_file is not None:
                _result_to_file(result_file, lamr, params)

            model_path=r'../tensorflow_model/02_objdet/new_all_model.pb'
            net.save_model(sess,model_path)

