#!/bin/bash
python evaluate_network_semseg.py  --img_dir=/private/liusongwei/01_semseg/img/val2014  --labels_dir=/private/liusongwei/01_semseg/annotations_converted/instances_val2014 --file_list=/private/liusongwei/01_semseg/coco_val2014_file_list.txt  --pb_file=../quantized_model/fcn8_prune0.3/seg3_params_change.npz --sub_model=../tensorflow_model/01_semseg/three_clip_model.pb  --result_file=../tensorflow_model/01_semseg/reference_result/quantized_prune3_result.txt --choose=1

