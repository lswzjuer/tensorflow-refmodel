"""
Reference Model
Copyright (c) 2018 MobaiTech Inc 
Author: Abinash Mohanty
Description: 
	Class for vgg-19. 
	Input data is Batch_size x 224 x 224 x 3. 
	For inference in our case Batch_size == 1
	Pre-processing function
"""

from .network import Network
import tensorflow as tf
from utils.blob import im_list_to_blob
from utils.softmax import softmax
from utils.imageNet_classNames import imageNet_classNames
from config import cfg
import numpy as np
import cv2

class vgg19(Network):
	def __init__(self, isHardware=True, trainable=False):
		self.inputs = []
		self.data = tf.placeholder(tf.float32, shape=[None, None, None, 3])
		self.layers = dict({'data': self.data})
		self.trainable = trainable
		self.n_classes = 1000
		self.classes = imageNet_classNames()
		self.setup()

	def pre_process(self, im, scale_mode=0):
		"""
		MUST HAVE FUNCTION IN ALL NETWORKS !!!! 
		Pre-processing of the image. De-mean, crop and resize
		Returns the feed dictionary the network is expecting
		For VGG there is just one size image 224x224 at the moment.		
		"""
		im_orig = im.astype(np.uint8, copy=True)
		im_bgr = cv2.resize(im_orig, (224,224))
		img_mean = np.zeros(shape = img_bgr.shape)
		img_mean[:, :, 0] = cfg.PIXEL_MEANS[0]
		img_mean[:, :, 1] = cfg.PIXEL_MEANS[1]
		img_mean[:, :, 2] = cfg.PIXEL_MEANS[2]
		img_demean = img_bgr-img_mean
		processed_ims = [img_demean]
		blob = im_list_to_blob(processed_ims)		 
		feed_dict = {self.data: blob}
		return feed_dict

	def post_process(self, im, sim_ops, scale_factor=1):
		"""
		MUST HAVE FUNCTION IN ALL NETWORKS !!!! 
		Post-processing of the results from network. This function can be used to visualize data from hardware.  
		"""
		prob = softmax(sim_ops[0])
		preds = (np.argsort(prob)[::-1])[0:5]
		for p in preds:
			print_msg(str(classes[p]) +' , ' + str(prob[p]), 3)		

	def run_sw_demo(self, img, scale_mode=0):
		"""
		Pure software demo.	
		"""	
		print_msg('Running software only demo ... ', 3)
		im = cv2.imread(image_path)	
		feed_dict = self.pre_process(im, scale_mode)
		fc8 = sess.run([self.get_output('fc8')], feed_dict=feed_dict)
		self.post_process(im, [fc8])		# Post process on network output and show result 			

	def create_layer_map(self):
		""" 
		Helper Function for Validation. Dictionary of layer wise parameters. 
		TODO: get correct fl and fl_w values and update this file 
		"""
		pass

	def setup(self):
		pass
