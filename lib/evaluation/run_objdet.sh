#!/bin/bash
python evaluate_network_objdet.py --annotations_dir=/private/liusongwei/02_objdet/annotations_processed --anchor_file=../tensorflow_model/02_objdet/utils/encoder_anchors_1080x1920.npy --pb_file=../quantized_model/ssd/obj_params_change.npz --sub_model=../tensorflow_model/02_objdet/sub_model.pb --result_file ../tensorflow_model/02_objdet/reference_result/quantized_result.txt --choose=0
