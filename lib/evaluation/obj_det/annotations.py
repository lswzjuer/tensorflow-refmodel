import numpy as np
from scipy.io import loadmat
import argparse
import os
import os.path as osp
from glob import glob
from copy import deepcopy
import pickle
import json


def split_from_file(split_files):
    """
    Reads dataset split from files, where each file contains paths to each contained sequence.

    Parameters
    ----------
    split_files: list
        List of filepaths, where each file contains paths to the contained video sequences. Each sequence is
        expected to form its own line in the file.

    Returns
    -------
    list
        Each element is a list of video file indices for each split
    """
    splits = []
    for split_file in split_files:
        splits.append([])
        with open(split_file, 'r') as f:
            for line in f.readlines():
                line = line.strip("\n")
                video_idx = int(osp.basename(osp.splitext(line)[0]).split("_")[1]) - 1  # Correct 1-indexing
                splits[-1].append(video_idx)
    return splits


def parse_bbox_from_vbb(filename):
    """
    Parse MATLAB .vbb object detection annotation files and read position, id, and occlusion
    information for each box in each annotated frame.

    Parameters
    ----------
    filename: str
        Path to .vbb file to be read

    Returns
    ----------
    tuple of list and int
        List of bounding box annotations for each object and each frame containing position,
        object id and (binary) occlusion flag. Also returns number of parsed frames
    """

    vbb = loadmat(filename)

    objLists = vbb['A'][0][0][1][0]
    object_label = [str(v[0]) for v in vbb['A'][0][0][4][0]]

    bbox_dict = dict()
    n_frames = len(objLists)

    for frame_id, obj in enumerate(objLists):
        if obj[0].shape[0] > 0:
            for i in range(obj[0].shape[0]):
                oid = int(obj[0][i][0][0])
                obj_id = object_label[oid - 1]
                if obj_id not in bbox_dict.keys():
                    bbox_dict[obj_id] = dict()
                bbox_coords = np.array(obj[0][i][1][0], dtype=np.float32)
                bbox_coords[2:4] += bbox_coords[0:2]  # Convert ltwh to ltrb
                bbox_coords -= 1  # Convert 1-indexing to 0-indexing
                bbox_id = int(obj[0][i][0][0])
                bbox_occl = float(obj[0][i][3][0])
                bbox_dict[obj_id][frame_id] = dict(pos=bbox_coords, id=bbox_id, occl=bbox_occl)

    return bbox_dict, n_frames


def compute_merged_annotations(annot_full, annot_part):
    # Maps partial occlusions to occlusion value 0.5 and full occlusions to 1.0
    # Necessary due to two independent annotation sets for both occlusion fractions in JAAD
    annot = deepcopy(annot_full)
    for obj in annot_full:
        for fr_idx in annot_full[obj]:
            fr_annot_full = annot_full[obj][fr_idx]
            fr_annot_part = annot_part[obj][fr_idx]
            # Check for annotations with occlusion flag set in partial occlusion annotations
            # and with unset flag in full occlusion annotations
            if fr_annot_part["occl"] == 1.0 and fr_annot_full["occl"] == 0.0:
                annot[obj][fr_idx]["occl"] = 0.5
    return annot


def process_seq_annotations(annotations, n_frames, seq_idx):
    """
    Generates the annotations of a sequence by processing each frame and box. Most annotations are preserved
    in the original format, except for the label, which is extracted from the object name annotation, and
    behavior attributes are distributed to each object and frame instead of defining the independently.
    """
    obj_names = sorted(annotations.keys())
    obj_ids = list(range(len(obj_names)))
    frame_annotations = []
    for fr_idx in range(n_frames):
        frame_entry = dict(frame_idx=fr_idx, sequence=seq_idx, annotations=[])
        bbox_annotations = frame_entry["annotations"]
        for obj_id in obj_ids:
            # Check whether object annotation exists in frame
            this_obj_name = obj_names[obj_id]
            if fr_idx in annotations[this_obj_name]:
                bbox = dict()
                bbox["id"] = obj_id
                bbox["annotation_id"] = annotations[this_obj_name][fr_idx]["id"]
                bbox["box"] = annotations[this_obj_name][fr_idx]["pos"]
                bbox["occluded"] = annotations[this_obj_name][fr_idx]["occl"]
                bbox["obj_name"] = this_obj_name
                # Label scheme documentation on dataset webpage is incomplete, this covers classes
                # car, person and people
                lbl = "car" if "car" in this_obj_name else "people" if "people" in this_obj_name else "person"
                bbox["label"] = lbl

                bbox_annotations.append(bbox)

        frame_annotations.append(frame_entry)
    return frame_annotations


def parse_annotations(video_dir, annotations_dir, split_dir, split_names=('test',), skip=1, out_dir=None):
    video_files = sorted(glob(osp.join(video_dir, '*.mp4')))
    if len(video_files) == 0:
        raise RuntimeError('Could not find any video files in folder {}'.format(video_dir))

    full_annotations_dir = osp.join(annotations_dir, 'JAAD_pedestrian-master', 'vbb_full')
    full_annotations_files = sorted(glob(osp.join(full_annotations_dir, '*.vbb')))

    predefined_splits = list(map(lambda x: osp.join(split_dir, x + "_split.txt"), split_names))
    split_idx = split_from_file(predefined_splits)

    dataset_annotations = []
    for this_split_idx, video_idx in enumerate(split_idx):
        dataset_annotations.append([])
        full_annotation_files_split = [full_annotations_files[i] for i in video_idx]
        video_files_split = [video_files[i] for i in video_idx]

        for seq_idx, this_annot_file in enumerate(full_annotation_files_split):
            this_video_file = video_files_split[seq_idx]
            video_name = osp.basename(osp.splitext(this_annot_file)[0])
            annotations_full, n_frames_annot = parse_bbox_from_vbb(this_annot_file)
            annotations_part, _ = parse_bbox_from_vbb(this_annot_file.replace("vbb_full", "vbb_part"))
            annotations = compute_merged_annotations(annotations_full, annotations_part)
            seq_annotations = process_seq_annotations(annotations, n_frames_annot, seq_idx)
            dataset_annotations[-1].append(dict(name=video_name, name_ext=osp.basename(this_video_file),
                                                full_path=this_video_file, frames=[], frame_skip=skip))
            dataset_annotations[-1][-1]["frames"] = seq_annotations[::skip]

        if out_dir is not None:
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            output_file = osp.join(out_dir, split_names[this_split_idx] + ".pckl")
            with open(output_file, "wb") as file:
                pickle.dump(dataset_annotations[-1], file)

    return dataset_annotations, split_names


def convert_to_json_for_coco(annotations, split_name=None, out_dir=None):
    categories = list()
    categories.append({'id': 1, 'name': 'person'})
    categories.append({'id': 2, 'name': 'car'})
    data = {'images': [], 'annotations': [], 'categories': categories}
    image_id = 0
    ann_id = 0
    for vid in annotations:
        for fr in vid['frames']:
            data['images'].append({'filename': vid['name'], 'frame': fr['frame_idx'], 'id': image_id})
            for ann in fr['annotations']:
                box = ann['box']
                box[2:] = box[2:] - box[:2]
                box = box.tolist()
                width = box[2]
                height = box[3]
                area = width*height

                data['annotations'].append({'bbox': box, 'area': area, 'image_id': image_id, 'id': ann_id,
                                            'occluded': ann['occluded'], 'label': ann['label'],
                                            'category_id': 1 if ann['label'] == 'person' else 2,
                                            'iscrowd': 0, 'height': height, 'width': width,
                                            })
                ann_id = ann_id + 1
            image_id = image_id + 1
    if out_dir is not None:
        with open(osp.join(out_dir, split_name + ".json"), "w") as file:
            json.dump(data, file)
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare the annotations for the object detection task on the JAAD '
                                                 'dataset.')
    parser.add_argument('--video_dir', type=str, help='Path to the JAAD video clips (mp4).')
    parser.add_argument('--annotations_dir', type=str, help='Path to the directory containing the pedestrian and '
                                                            'behavior annotation github repository '
                                                            'JAAD_pedestrian_master.')
    parser.add_argument('--split_dir', type=str, help='Path to the predefined splits of the data into train/val/test.')
    parser.add_argument('--out_dir', type=str, help='Directory where parse annotations are stored.', default=None)
    parser.add_argument('--skip', type=int, default=4, help='Frame skip in videos (default is 4 for evaluation).')
    args = parser.parse_args()

    video_dir = osp.abspath(args.video_dir)
    annotations_dir = args.annotations_dir
    split_dir = args.split_dir
    out_dir = args.out_dir
    skip = args.skip

    if out_dir is not None:
        print('Output is stored in {}'.format(out_dir))

    annotations, split_names = parse_annotations(video_dir, annotations_dir, split_dir, skip=skip, out_dir=out_dir)

    for ann, name in zip(annotations, split_names):
        convert_to_json_for_coco(ann, name, out_dir)
