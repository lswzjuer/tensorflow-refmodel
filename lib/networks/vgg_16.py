"""
Reference Model
Copyright (c) 2018 MobaiTech Inc 
Author: Abinash Mohanty
Description: 
	Class for vgg-16. 
	Input data is Batch_size x 224 x 224 x 3. 
	For inference in our case Batch_size == 1
	Pre-processing function
Revisions:
1. Abinash: Net created (10/30/2018)  
"""

from .network import Network
import tensorflow as tf
from utils.blob import im_list_to_blob
from utils.softmax import softmax
from utils.imageNet_classNames import imageNet_classNames
from config import cfg
import numpy as np
import cv2
from utils.refModel_log import print_msg

class vgg16(Network):
	
	def __init__(self, isHardware=True, trainable=False):
		self.inputs = []
		self.trainable = trainable
		self.isHardware = isHardware
		self.data = tf.placeholder(tf.float32, shape=[None, 224, 224, 3])
		self.layers = dict({'data': self.data})
		self.trainable = trainable
		self.n_classes = 1000
		self.classes = imageNet_classNames()
		self.setup()
		print_msg ('VGG16 Network setup done',0)

	def pre_process(self, im, scale_mode=0):
		"""
		MUST HAVE FUNCTION IN ALL NETWORKS !!!! 
		Pre-processing of the image. De-mean, crop and resize
		Returns the feed dictionary the network is expecting
		For VGG there is just one size image 224x224 at the moment.		
		"""
		print_msg('Pre-processing the image for the network', 4)
		im_orig = im.astype(np.uint8, copy=True)
		im_bgr = cv2.resize(im_orig, (224,224))
		img_mean = np.zeros(shape = im_bgr.shape)
		img_mean[:, :, 0] = cfg.PIXEL_MEANS[0]
		img_mean[:, :, 1] = cfg.PIXEL_MEANS[1]
		img_mean[:, :, 2] = cfg.PIXEL_MEANS[2]
		img_demean = im_bgr-img_mean
		processed_ims = [img_demean]
		blob = im_list_to_blob(processed_ims)		 
		feed_dict = {self.data: blob}
		return feed_dict

	def post_process(self, im, sim_ops, scale_factor=1):
		"""
		MUST HAVE FUNCTION IN ALL NETWORKS !!!! 
		Post-processing of the results from network. This function can be used to visualize data from hardware.  
		"""
		prob = softmax(sim_ops[0][0])
		preds = (np.argsort(prob)[::-1])[0:5]
		# top5 accuary
		for p in preds:
			print_msg(str(self.classes[p]) +' , ' + str(prob[p]), 3)		

	def run_sw_demo(self, sess, img, scale_mode=0):
		"""
		Pure software demo.	
		"""	
		print_msg('Running software only demo ... ', 3)
		im = cv2.imread(img)	
		feed_dict = self.pre_process(im, scale_mode)
		fc8 = sess.run([self.get_output('fc8')], feed_dict=feed_dict)
		# print top5 results
		self.post_process(im, fc8)		# Post process on network output and show result 			

	def create_layer_map(self):
		""" 
		Helper Function for Validation. Dictionary of layer wise parameters. 
		TODO: get correct fl and fl_w values and update this file 
		"""
		self._layer_map[0]={'name':'data',		'inputs':[],	'fl':6,	'fl_w':15}
		self._layer_map[1]={'name':'conv1_1',	'inputs':[0],	'fl':6,	'fl_w':15}
		self._layer_map[2]={'name':'conv1_2',	'inputs':[1],	'fl':6,	'fl_w':15}
		self._layer_map[3]={'name':'pool1',		'inputs':[2],	'fl':6,	'fl_w':15}
		self._layer_map[4]={'name':'conv2_1',	'inputs':[3],	'fl':6,	'fl_w':15}
		self._layer_map[5]={'name':'conv2_2',	'inputs':[4],	'fl':6,	'fl_w':15}
		self._layer_map[6]={'name':'pool2',		'inputs':[5],	'fl':6,	'fl_w':15}
		self._layer_map[7]={'name':'conv3_1',	'inputs':[6],	'fl':6,	'fl_w':15}
		self._layer_map[8]={'name':'conv3_2',	'inputs':[7],	'fl':6,	'fl_w':15}
		self._layer_map[9]={'name':'conv3_3',	'inputs':[8],	'fl':6,	'fl_w':15}
		self._layer_map[10]={'name':'pool3',	'inputs':[9],	'fl':6,	'fl_w':15}
		self._layer_map[11]={'name':'conv4_1',	'inputs':[10],	'fl':6,	'fl_w':15}
		self._layer_map[12]={'name':'conv4_2',	'inputs':[11],	'fl':6,	'fl_w':15}
		self._layer_map[13]={'name':'conv4_3',	'inputs':[12],	'fl':6,	'fl_w':15}
		self._layer_map[14]={'name':'pool4',	'inputs':[13],	'fl':6,	'fl_w':15}
		self._layer_map[15]={'name':'conv5_1',	'inputs':[14],	'fl':6,	'fl_w':15}
		self._layer_map[16]={'name':'conv5_2',	'inputs':[15],	'fl':6,	'fl_w':15}
		self._layer_map[17]={'name':'conv5_3',	'inputs':[16],	'fl':6,	'fl_w':15}
		self._layer_map[18]={'name':'pool5',	'inputs':[17],	'fl':6,	'fl_w':15}
		self._layer_map[19]={'name':'fc6',		'inputs':[18],	'fl':6,	'fl_w':15}
		self._layer_map[20]={'name':'fc7',		'inputs':[19],	'fl':6,	'fl_w':15}
		self._layer_map[21]={'name':'fc8',		'inputs':[20],	'fl':6,	'fl_w':15}

	def setup(self):
		""" 
		TODO: get correct fl values and update this file 
		"""
		(self.feed('data')
		 .conv(3, 3, 64, 1, 1, fl=6, name='conv1_1')
		 .conv(3, 3, 64, 1, 1, fl=6, name='conv1_2')
		 .max_pool(2, 2, 2, 2, padding='VALID', name='pool1')
		 .conv(3, 3, 128, 1, 1, fl=6, name='conv2_1')
		 .conv(3, 3, 128, 1, 1, fl=6, name='conv2_2')
		 .max_pool(2, 2, 2, 2, padding='VALID', name='pool2')
		 .conv(3, 3, 256, 1, 1, fl=6, name='conv3_1')
		 .conv(3, 3, 256, 1, 1, fl=6, name='conv3_2')
		 .conv(3, 3, 256, 1, 1, fl=6, name='conv3_3')
		 .max_pool(2, 2, 2, 2, padding='VALID', name='pool3')
		 .conv(3, 3, 512, 1, 1, fl=6, name='conv4_1')
		 .conv(3, 3, 512, 1, 1, fl=6, name='conv4_2')
		 .conv(3, 3, 512, 1, 1, fl=6, name='conv4_3')
		 .max_pool(2, 2, 2, 2, padding='VALID', name='pool4')
		 .conv(3, 3, 512, 1, 1, fl=6, name='conv5_1')
		 .conv(3, 3, 512, 1, 1, fl=6, name='conv5_2')
		 .conv(3, 3, 512, 1, 1, fl=6, name='conv5_3')
		 .max_pool(2, 2, 2, 2, padding='VALID', name='pool5')	
		 .fc(4096, name='fc6', fl=6)
		 .fc(4096, name='fc7', fl=6)
		 .fc(self.n_classes, name='fc8', fl=6))
