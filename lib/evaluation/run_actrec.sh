#!/bin/bash
python evaluate_network_actrec.py --actrec_dir=/private/liusongwei/03_actrec --pb_file=../quantized_model/rcnn/act_params_change.npz --sub_model=../tensorflow_model/03_actrec/sub_model.pb --result_file=../tensorflow_model/03_actrec/reference_result/quantized_result.txt


