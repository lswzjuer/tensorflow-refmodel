"""
Reference Model
Copyright (c) 2018 MobaiTech Inc 
Author: Abinash Mohanty
"""

import matplotlib.pyplot as plt
import os, sys, cv2
import numpy as np
import argparse
import os.path as osp

np.set_printoptions(threshold=np.inf)

def add_path(path):
	"""
	This function adds path to python path. 
	"""
	if path not in sys.path:
		sys.path.insert(0,path)

# Add the lib directory to system path
lib_path = osp.abspath(osp.join(osp.dirname(__file__), 'lib'))
add_path(lib_path)
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'	# Remove logs from tensorflow


import tensorflow as tf
from networks.factory import get_network
from config import cfg
from utils.fixed_point import getFixedPoint, convert_to_fp 


from utils.parse_args import parse_args
from utils.refModel_log import print_msg

def visualize_output(net, em_image, scale_mode, sim_ops, scale_factor, save_output=False):
	"""
	Function to visualize hw/simulation output. 
	Arguments:
		net: name of the network
		em_image: path to the input image (user is responsible to send in appropriate image (classification vs detection))
		scale_mode: scaling mode (0: highest resolution, 1: mid resolution, 2: TODD)
		sim_ops: Simulation outputs (user is responsible to give all the necessary inputs needed to plot output). As a list.
				 User has to refer the implementation of post_process in lib/networks/xxx.py to give the corrent sim_ops.
		save_output: will save the output to file depending. TODO
	Returns: -- 
	"""	
	print_msg('Running RefModel to visualize sim/hw output',3)
	em_image = cfg.DATA_DIR+str(em_image)
	im = cv2.imread(em_image)
	#feed_dict = net.pre_process(im, scale_mode)
	#scale_factor = feed_dict[net.im_info][0][2]	# TODO send in scale for both X and Y directions 
        #model_path = check_model_path(em_net)
	#net = get_network(em_net, 1)
	net.post_process(im, sim_ops, scale_factor)	



def check_model_path(em_net, net, sess):

	model_path = cfg.MODELS_DIR+str(em_net)+'.npy'

	if not os.path.exists(model_path):
                model_path = cfg.MODELS_DIR+str(em_net)+'_random.npy'
                print_msg("Error: model " + model_path + " does not exist, try to use randomized model " + model_path, 3)
                if not os.path.exists(model_path):
                        status = create_random_model(net, sess, model_path)
                        if status != 0:
                                raise IOError(('Can not create random model @ '+ model_path +'.\n'))
        return model_path


def get_bias(net, model_path):
	"""
	API to return all the bias values in the model. To be used by validation framework to initialize the bias SRAM. 
	Pre-req: The model file (netname.npy) must be present in cfg.MODELS_DIR
	Arguments: 
		net: name of the network. 
	Returns: 
		list containing all the bias values for all the layers in same sequence as define in net._layer_map dict. 
	"""

	print_msg('Getting bias values for the network '+str(model_path)+'...',3)

	#model_path = cfg.MODELS_DIR+str(net)+'.npy'
	#if not os.path.exists(model_path):
	#	raise IOError(('Model not found @ '+ model_path +'.\n'))		

        #model_path = check_model_path(net)
	#net = get_network(str(net), 1)

	num_layers = len(net._layer_map.keys())	# Get the total number of layers in the network
	bias_vals = []
	layers_with_bias_count = 0
	with open(model_path) as fid:
		model = np.load(model_path).item()
		for i in range(num_layers): 
			if net._layer_map[i]['name'] in model: 
				if 'biases' in model[net._layer_map[i]['name']]:
					b = model[net._layer_map[i]['name']]['biases']
					b = convert_to_fp(b)
					bias_vals.append(b)
					layers_with_bias_count += 1	
				else: 
					bias_vals.append([])		
			else: 
				bias_vals.append([])		
			
	print_msg(str(layers_with_bias_count)+' layers were found with bias values',3)
	return bias_vals


def create_random_model(net, sess, random_model_path):
        status = 0
        model = {}
        for layer in net._layer_map.keys():
                name = net._layer_map[layer]['name']
                #print "layer name is " + name
                params = {}
                for i in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name):
                        #print("i.name: %s\n" %(i.name))
                        var_name = ""
                        if (i.name.find(name+"/weights") >= 0):
                                #print("FOUND weights in %s %s\n" %(name, i.name))
                                var_name = "weights"
                        elif (i.name.find(name + "/biases") >= 0):
                                #print("FOUND weights in %s %s\n" %(name, i.name))
                                var_name = "biases"
                                params[var_name] = np.random.randint(-40, 40, size=tuple(dims))
                        if (var_name):
                            shape = tf.shape(i)
                            dims = sess.run(shape)
                            params[var_name] = np.random.randint(-40, 40, size=tuple(dims))
                if (params):
                    model[name] = params
        #TODO check if we can get exceptions 
        np.save(random_model_path, model)
        return status

def get_network_map(net):
	#model_path = cfg.MODELS_DIR+str(net)+'.npy'
	#if not os.path.exists(model_path):
	#	raise IOError(('Model not found @ '+ model_path +'.\n'))		
	# init session
	tf.reset_default_graph()
	sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

	net = get_network(str(net),1)
	return net.get_layer_map()


def get_data_for_validation(em_net, em_image, scale_mode, em_start_layer, em_end_layer):
	"""
	Efficiently get the inputs, outputs, weights of layers given by start and end index
	Arguments: 
		sess: tensorflow session
		net: network to test 
		em_image: path to the input image (user is responsible to send in appropriate image (classification vs detection))
		scale_mode: scaling mode (0: highest resolution, 1: mid resolution, 2: TODD)
		em_start_layer: ID of the start layer
		em_end_layer: ID of the end layer
	Returns:
		List called multi_layer_outputs
		Each element corresponds to a layers to be tested from em_start_layer to em_end_layer 
		for each layer: [layer_output, layer_inputs, parameters]
		to access data for layer I, (x = I - em_start_layer): 
			layer output: 		multi_layer_outputs[x][0] 
			layer inputs (j): 	multi_layer_outputs[x][1][j]	
			layer weights: 		multi_layer_outputs[x][2][0]
			layer biases: 		multi_layer_outputs[x][2][1]
	"""
	print_msg('Running RefModel to get data for verification',3)
	print_msg('Network: '+str(em_net),3)	
	print_msg('Test Image: '+str(em_image),3)	
	print_msg('Resolution Mode: '+str(scale_mode),3)	
	print_msg('Start Layer: '+str(em_start_layer),3)	
	print_msg('End Layer: '+str(em_end_layer),3)	

	# init session
	tf.reset_default_graph()
	sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

	# create network instance and load models parameters 
	net = get_network(str(em_net), 1)

	#model_path = cfg.MODELS_DIR+str(em_net)+'.npy'
    model_path = check_model_path(em_net, net, sess)

	em_image = cfg.DATA_DIR+str(em_image)

	if not os.path.exists(em_image):
		raise IOError(('Image not found @ '+ em_image +'.\n'))	
	
        print_msg('Model Path: '+str(model_path),3)

	if not os.path.exists(model_path):
                model_path = cfg.MODELS_DIR+str(em_net)+'_random.npy'
                print_msg("Error: model " + model_path + " does not exist, try to use randomized model " + model_path, 3)
                if not os.path.exists(model_path):
                        status = create_random_model(net, sess, model_path)
                        if status != 0:
                                raise IOError(('Can not create random model @ '+ model_path +'.\n'))

	sess.run(tf.global_variables_initializer())
	net.load(model_path,sess) 

	im = cv2.imread(em_image)	
	feed_dict = net.pre_process(im, scale_mode)
        print("im_info\n")
        print(net.im_info)
        #print("feed_dict")
        #print(feed_dict)
        #print("feed_dict[net.im_info]\n")
        #print(feed_dict[net.im_info])

        scale_factor = 0.55
        #scale_factor = feed_dict[net.im_info][0][2]	# TODO send in scale for both X and Y directions 
	compute_dict = {}	
	compute_dict_inv = {}	

	for LUT in range(em_start_layer, em_end_layer+1):
		if LUT not in compute_dict:
			compute_dict[LUT] = net.get_output(net._layer_map[LUT]['name'])
			compute_dict_inv[net.get_output(net._layer_map[LUT]['name'])] = LUT
		for LUT_input in net._layer_map[LUT]['inputs']: 
			if LUT_input not in compute_dict:
				compute_dict[LUT_input] = net.get_output(net._layer_map[LUT_input]['name'])
				compute_dict_inv[net.get_output(net._layer_map[LUT_input]['name'])] = LUT_input

	compute_list = 	compute_dict.values()
	op = []
	op = sess.run(compute_list, feed_dict=feed_dict)

	for ii in range(len(compute_list)):
		ID = compute_dict_inv[compute_list[ii]]
		result = op[ii]
		#result = convert_to_fp(result)		
		compute_dict[compute_dict_inv[compute_list[ii]]] = result

	multi_layer_outputs = []
	with open(model_path) as fid:
		model = np.load(model_path).item()
		for LUT in range(em_start_layer, em_end_layer+1):	
			layer_output = []
			layer_inputs = []
			parameters	 = []
			layer_output.append(compute_dict[LUT])
			layer_info = net._layer_map[LUT]
			for LUT_input in layer_info['inputs']:	
				layer_inputs.append(compute_dict[LUT_input])
			layer_output.append(layer_inputs)
			if layer_info['name'] in model: 
				w = None
				b = None
				params = model[layer_info['name']]			
				for key in params.keys():	   
					if key == 'weights':
						w = params[key]
						#w = convert_to_fp(w)	
					elif key == 'biases': 
						b = params[key]
						#b = convert_to_fp(b)	
					else:
						raise KeyError('Unknown Key in model file: %s'%key)   
				if w is not None:	
					parameters.append(w)
				if b is not None:	
					parameters.append(b)			
			multi_layer_outputs.append([layer_output, layer_inputs, parameters])	
	return multi_layer_outputs, net.get_layer_map(), net, scale_factor, model_path


if __name__ == '__main__':

	"""
	Execute this file to run a software demo for supported networks. 
	Command line args: 
		network: name of the network. call Network.factory.list_networks to see the supported networks
		hw_sim: flag to see hardware results (low precision)
		image: path of the image (relative to the cnn_emu_data folder)
		resolution_mode: 0: highest resolution, 1: mid level resolution	
	"""
	args = parse_args()		# Get runtime command line args
	em_network = args.network_name
	em_isHardware = args.isHardware
	em_image = args.image 
	em_image_resolution_mode = args.image_resolution_mode	

	print_msg('Running RefModel in SW standalone mode ... ',3)
	print_msg('Network: '+str(em_network),3)	
	print_msg('Test Image: '+str(em_image),3)	
	print_msg('Resolution Mode: '+str(em_image_resolution_mode),3)	

	#model = cfg.MODELS_DIR+str(em_network)+'.npy'
	em_image = cfg.DATA_DIR+str(em_image)

	# Sanity checks
	if not os.path.exists(em_image):
		raise IOError(('Image not found @ '+ em_image +'.\n'))		
	#if not os.path.exists(model):
	#	raise IOError(('Model not found @ '+ model +'.\n'))	


	# init session
	tf.reset_default_graph()
	sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

	# create network instance and load models parameters 
	net = get_network(str(em_network), em_isHardware)
    model_path = check_model_path(em_network, net, sess)
	sess.run(tf.global_variables_initializer())
	net.load(model_path,sess) 
	net.run_sw_demo(sess, em_image, 0)	

