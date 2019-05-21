# REFERENCE MODEL

(c) FABU America Inc. This is emulator to give a one-to-one hardware output for the hardware accelerator. Framework organization is inspired from the open source implementation of Faster-RCNN from [here](https://github.com/CharlesShang/TFFRCNN). The framework internally uses Tensorflow to do major portion of the computations and then converts the results to a format similar to the hardware output. 


### Requirements: Software
1. Python 2.7.15 (recommended: [Anaconda](https://conda.io/docs/user-guide/install/download.html))
2. Python packages like: `python-opencv`, `easydict`, `matplotlib`, `cv2` etc... 
3. [Tensorflow](https://www.tensorflow.org/). 


### What's New

- [x] PVANet Obstacle network support
- [x] PVANet Traffic Light network support
- [ ] Traffic Lane network support
- [ ] New Layers for SoC (Deconv, Upsampling, etc)
- [x] Proposal Layer similar to hardware
- [ ] Post processing API to visualize the RTL output
- [ ] Support for multi-size images 
- [ ] more hacks :-D 


### Reference Model Architecture
	```
	RefModel
		|-- data
			|-- ob
				|-- raw
					|-- Images ... 
				|-- ref_data_from_caffe
					|-- blobs_obstacle_006.cpk
					|-- pre_obstacle_006.cpk
			|-- tl 
				|-- Images ... 
		|-- lib
			|-- __init__.py
			|-- Makefile
			|-- make.sh
			|-- setup.py
			|-- networks
				|-- __init__.py
				|-- factory.py
				|-- network.py
				|-- pvanet_ob.py
				|-- pvanet_tl.py
			|-- fast_rcnn
				|-- __init__.py 
				|-- bbox_transform.py
				|-- config.py
				|-- nms_wrapper.py
				|-- test.py
			|-- lp
				|-- __init__.py
				|-- lp_op.cc 
				|-- lp_op.py
				|-- lp_op_test.py
				|-- lp.so
				|-- make_ref_hw.sh
			|-- nms
				|-- __init__.py
				|-- cpu_nms.c
				|-- cpu_nms.pyx	
				|--	cpu_nms.so
				|--	gpu_nms.hpp
				|--	gpu_nms.pyx
				|--	nms_kernel.cu

				|-- py_cpu_nms.py
			|-- pucocotools
			|-- roi_pooling_layer
				|--	__init__.py
				|--	roi_pooling_op.cc
				|--	roi_pooling_op_gpu.h
				|--	roi_pooling_op_grad.pyc
				|--	roi_pooling_op.pyc
				|--	roi_pooling.so
				|--	roi_pooling_op_gpu.cu.cc
				|-- roi_pooling_op_grad.py	
				|--	roi_pooling_op.py
				|--	roi_pooling_op_test.py
			|-- rpn_msr				
				|--	generate_anchors.py
				|--	generate.py
				|--	__init__.py
				|--	nms_hw.py
				|--	proposal_layer_tf.py
				|--	sqrootLookUp.py
			|-- utils
				|-- bbox.c	
				|--	blob.py
				|--	box_grid.py
				|--	cython_bbox.so
				|-- fixed_point.py	
				|--	__init__.py
				|-- nms.c	
				|--	timer.py
				|--	bbox.pyx
				|--	cython_nms.so
				|--	nms.py
				|--	nms.pyx
		|-- models
			|--obstacle_model.cpk
			|--tl_model.cpk
		|-- 
			|-- demo.py
			|-- _init_paths.py
			|-- pvatest.py
			|-- refModel.py
		|-- README.md	
	```

### Setup and Run 
1. Load appropriate python module: 	
	```
	module unload python
	module load python\2.7.15
	```
2. ```cd $cnn_emu```
3. ```python refModel --network pvanet_ob --hw_sim 1 --image 000.jpg --resolution_mode 0```

### Running tensorboard: 
1. Set cfg.ENABLE_TENSORBOARD = True
2. Run python refModel (with net and other parameters)
3. run in command prompt to launch tensorboard: tensorboard --logdir=output
4. open browser and go to http://decacs1:6006 
5. Enjoy :-D 
 
