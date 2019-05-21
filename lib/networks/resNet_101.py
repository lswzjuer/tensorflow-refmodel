"""
Reference Model
Copyright (c) 2018 MobaiTech Inc 
Author: Abinash Mohanty
Description: 
	Class for resNet-50. 
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



class resnet101(Network):
	def __init__(self, isHardware=True, trainable=False):
		self.inputs = []
		self.data = tf.placeholder(tf.float32, shape=[None, None, None, 3])
		self.layers = dict({'data':self.data})
		self.trainable = trainable
		self.n_classes = 1000
		self.classes = imageNet_classNames()
		self.create_layer_map()
		self.setup()

	def preProcess(self, im, scale_mode=0):
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

	def post_process(self, im, sim_ops, scale_factor):
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
		fc1000 = sess.run([self.get_output('fc1000')], feed_dict=feed_dict)
		self.post_process(im, [fc1000])		# Post process on network output and show result

	def create_layer_map(self):
		""" 
		Helper Function for Validation. Dictionary of layer wise parameters. 
		TODO: get correct fl and fl_w values and update this file 
		"""
		pass 


	def setup(self):
		(self.feed('data')
			 .conv(7, 7, 64, 2, 2, biased=False, relu=False, name='conv1')
			 .batch_normalization(relu=True, name='bn_conv1')
			 .max_pool(3, 3, 2, 2, name='pool1')
			 .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res2a_branch1')
			 .batch_normalization(name='bn2a_branch1'))

		(self.feed('pool1')
			 .conv(1, 1, 64, 1, 1, biased=False, relu=False, name='res2a_branch2a')
			 .batch_normalization(relu=True, name='bn2a_branch2a')
			 .conv(3, 3, 64, 1, 1, biased=False, relu=False, name='res2a_branch2b')
			 .batch_normalization(relu=True, name='bn2a_branch2b')
			 .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res2a_branch2c')
			 .batch_normalization(name='bn2a_branch2c'))

		(self.feed('bn2a_branch1',
				   'bn2a_branch2c')
			 .add(name='res2a')
			 .relu(name='res2a_relu')
			 .conv(1, 1, 64, 1, 1, biased=False, relu=False, name='res2b_branch2a')
			 .batch_normalization(relu=True, name='bn2b_branch2a')
			 .conv(3, 3, 64, 1, 1, biased=False, relu=False, name='res2b_branch2b')
			 .batch_normalization(relu=True, name='bn2b_branch2b')
			 .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res2b_branch2c')
			 .batch_normalization(name='bn2b_branch2c'))

		(self.feed('res2a_relu',
				   'bn2b_branch2c')
			 .add(name='res2b')
			 .relu(name='res2b_relu')
			 .conv(1, 1, 64, 1, 1, biased=False, relu=False, name='res2c_branch2a')
			 .batch_normalization(relu=True, name='bn2c_branch2a')
			 .conv(3, 3, 64, 1, 1, biased=False, relu=False, name='res2c_branch2b')
			 .batch_normalization(relu=True, name='bn2c_branch2b')
			 .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res2c_branch2c')
			 .batch_normalization(name='bn2c_branch2c'))

		(self.feed('res2b_relu',
				   'bn2c_branch2c')
			 .add(name='res2c')
			 .relu(name='res2c_relu')
			 .conv(1, 1, 512, 2, 2, biased=False, relu=False, name='res3a_branch1')
			 .batch_normalization(name='bn3a_branch1'))

		(self.feed('res2c_relu')
			 .conv(1, 1, 128, 2, 2, biased=False, relu=False, name='res3a_branch2a')
			 .batch_normalization(relu=True, name='bn3a_branch2a')
			 .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='res3a_branch2b')
			 .batch_normalization(relu=True, name='bn3a_branch2b')
			 .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res3a_branch2c')
			 .batch_normalization(name='bn3a_branch2c'))

		(self.feed('bn3a_branch1',
				   'bn3a_branch2c')
			 .add(name='res3a')
			 .relu(name='res3a_relu')
			 .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='res3b1_branch2a')
			 .batch_normalization(relu=True, name='bn3b1_branch2a')
			 .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='res3b1_branch2b')
			 .batch_normalization(relu=True, name='bn3b1_branch2b')
			 .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res3b1_branch2c')
			 .batch_normalization(name='bn3b1_branch2c'))

		(self.feed('res3a_relu',
				   'bn3b1_branch2c')
			 .add(name='res3b1')
			 .relu(name='res3b1_relu')
			 .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='res3b2_branch2a')
			 .batch_normalization(relu=True, name='bn3b2_branch2a')
			 .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='res3b2_branch2b')
			 .batch_normalization(relu=True, name='bn3b2_branch2b')
			 .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res3b2_branch2c')
			 .batch_normalization(name='bn3b2_branch2c'))

		(self.feed('res3b1_relu',
				   'bn3b2_branch2c')
			 .add(name='res3b2')
			 .relu(name='res3b2_relu')
			 .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='res3b3_branch2a')
			 .batch_normalization(relu=True, name='bn3b3_branch2a')
			 .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='res3b3_branch2b')
			 .batch_normalization(relu=True, name='bn3b3_branch2b')
			 .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res3b3_branch2c')
			 .batch_normalization(name='bn3b3_branch2c'))

		(self.feed('res3b2_relu',
				   'bn3b3_branch2c')
			 .add(name='res3b3')
			 .relu(name='res3b3_relu')
			 .conv(1, 1, 1024, 2, 2, biased=False, relu=False, name='res4a_branch1')
			 .batch_normalization(name='bn4a_branch1'))

		(self.feed('res3b3_relu')
			 .conv(1, 1, 256, 2, 2, biased=False, relu=False, name='res4a_branch2a')
			 .batch_normalization(relu=True, name='bn4a_branch2a')
			 .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='res4a_branch2b')
			 .batch_normalization(relu=True, name='bn4a_branch2b')
			 .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4a_branch2c')
			 .batch_normalization(name='bn4a_branch2c'))

		(self.feed('bn4a_branch1',
				   'bn4a_branch2c')
			 .add(name='res4a')
			 .relu(name='res4a_relu')
			 .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b1_branch2a')
			 .batch_normalization(relu=True, name='bn4b1_branch2a')
			 .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='res4b1_branch2b')
			 .batch_normalization(relu=True, name='bn4b1_branch2b')
			 .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b1_branch2c')
			 .batch_normalization(name='bn4b1_branch2c'))

		(self.feed('res4a_relu',
				   'bn4b1_branch2c')
			 .add(name='res4b1')
			 .relu(name='res4b1_relu')
			 .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b2_branch2a')
			 .batch_normalization(relu=True, name='bn4b2_branch2a')
			 .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='res4b2_branch2b')
			 .batch_normalization(relu=True, name='bn4b2_branch2b')
			 .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b2_branch2c')
			 .batch_normalization(name='bn4b2_branch2c'))

		(self.feed('res4b1_relu',
				   'bn4b2_branch2c')
			 .add(name='res4b2')
			 .relu(name='res4b2_relu')
			 .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b3_branch2a')
			 .batch_normalization(relu=True, name='bn4b3_branch2a')
			 .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='res4b3_branch2b')
			 .batch_normalization(relu=True, name='bn4b3_branch2b')
			 .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b3_branch2c')
			 .batch_normalization(name='bn4b3_branch2c'))

		(self.feed('res4b2_relu',
				   'bn4b3_branch2c')
			 .add(name='res4b3')
			 .relu(name='res4b3_relu')
			 .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b4_branch2a')
			 .batch_normalization(relu=True, name='bn4b4_branch2a')
			 .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='res4b4_branch2b')
			 .batch_normalization(relu=True, name='bn4b4_branch2b')
			 .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b4_branch2c')
			 .batch_normalization(name='bn4b4_branch2c'))

		(self.feed('res4b3_relu',
				   'bn4b4_branch2c')
			 .add(name='res4b4')
			 .relu(name='res4b4_relu')
			 .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b5_branch2a')
			 .batch_normalization(relu=True, name='bn4b5_branch2a')
			 .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='res4b5_branch2b')
			 .batch_normalization(relu=True, name='bn4b5_branch2b')
			 .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b5_branch2c')
			 .batch_normalization(name='bn4b5_branch2c'))

		(self.feed('res4b4_relu',
				   'bn4b5_branch2c')
			 .add(name='res4b5')
			 .relu(name='res4b5_relu')
			 .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b6_branch2a')
			 .batch_normalization(relu=True, name='bn4b6_branch2a')
			 .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='res4b6_branch2b')
			 .batch_normalization(relu=True, name='bn4b6_branch2b')
			 .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b6_branch2c')
			 .batch_normalization(name='bn4b6_branch2c'))

		(self.feed('res4b5_relu',
				   'bn4b6_branch2c')
			 .add(name='res4b6')
			 .relu(name='res4b6_relu')
			 .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b7_branch2a')
			 .batch_normalization(relu=True, name='bn4b7_branch2a')
			 .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='res4b7_branch2b')
			 .batch_normalization(relu=True, name='bn4b7_branch2b')
			 .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b7_branch2c')
			 .batch_normalization(name='bn4b7_branch2c'))

		(self.feed('res4b6_relu',
				   'bn4b7_branch2c')
			 .add(name='res4b7')
			 .relu(name='res4b7_relu')
			 .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b8_branch2a')
			 .batch_normalization(relu=True, name='bn4b8_branch2a')
			 .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='res4b8_branch2b')
			 .batch_normalization(relu=True, name='bn4b8_branch2b')
			 .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b8_branch2c')
			 .batch_normalization(name='bn4b8_branch2c'))

		(self.feed('res4b7_relu',
				   'bn4b8_branch2c')
			 .add(name='res4b8')
			 .relu(name='res4b8_relu')
			 .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b9_branch2a')
			 .batch_normalization(relu=True, name='bn4b9_branch2a')
			 .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='res4b9_branch2b')
			 .batch_normalization(relu=True, name='bn4b9_branch2b')
			 .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b9_branch2c')
			 .batch_normalization(name='bn4b9_branch2c'))

		(self.feed('res4b8_relu',
				   'bn4b9_branch2c')
			 .add(name='res4b9')
			 .relu(name='res4b9_relu')
			 .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b10_branch2a')
			 .batch_normalization(relu=True, name='bn4b10_branch2a')
			 .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='res4b10_branch2b')
			 .batch_normalization(relu=True, name='bn4b10_branch2b')
			 .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b10_branch2c')
			 .batch_normalization(name='bn4b10_branch2c'))

		(self.feed('res4b9_relu',
				   'bn4b10_branch2c')
			 .add(name='res4b10')
			 .relu(name='res4b10_relu')
			 .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b11_branch2a')
			 .batch_normalization(relu=True, name='bn4b11_branch2a')
			 .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='res4b11_branch2b')
			 .batch_normalization(relu=True, name='bn4b11_branch2b')
			 .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b11_branch2c')
			 .batch_normalization(name='bn4b11_branch2c'))

		(self.feed('res4b10_relu',
				   'bn4b11_branch2c')
			 .add(name='res4b11')
			 .relu(name='res4b11_relu')
			 .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b12_branch2a')
			 .batch_normalization(relu=True, name='bn4b12_branch2a')
			 .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='res4b12_branch2b')
			 .batch_normalization(relu=True, name='bn4b12_branch2b')
			 .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b12_branch2c')
			 .batch_normalization(name='bn4b12_branch2c'))

		(self.feed('res4b11_relu',
				   'bn4b12_branch2c')
			 .add(name='res4b12')
			 .relu(name='res4b12_relu')
			 .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b13_branch2a')
			 .batch_normalization(relu=True, name='bn4b13_branch2a')
			 .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='res4b13_branch2b')
			 .batch_normalization(relu=True, name='bn4b13_branch2b')
			 .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b13_branch2c')
			 .batch_normalization(name='bn4b13_branch2c'))

		(self.feed('res4b12_relu',
				   'bn4b13_branch2c')
			 .add(name='res4b13')
			 .relu(name='res4b13_relu')
			 .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b14_branch2a')
			 .batch_normalization(relu=True, name='bn4b14_branch2a')
			 .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='res4b14_branch2b')
			 .batch_normalization(relu=True, name='bn4b14_branch2b')
			 .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b14_branch2c')
			 .batch_normalization(name='bn4b14_branch2c'))

		(self.feed('res4b13_relu',
				   'bn4b14_branch2c')
			 .add(name='res4b14')
			 .relu(name='res4b14_relu')
			 .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b15_branch2a')
			 .batch_normalization(relu=True, name='bn4b15_branch2a')
			 .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='res4b15_branch2b')
			 .batch_normalization(relu=True, name='bn4b15_branch2b')
			 .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b15_branch2c')
			 .batch_normalization(name='bn4b15_branch2c'))

		(self.feed('res4b14_relu',
				   'bn4b15_branch2c')
			 .add(name='res4b15')
			 .relu(name='res4b15_relu')
			 .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b16_branch2a')
			 .batch_normalization(relu=True, name='bn4b16_branch2a')
			 .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='res4b16_branch2b')
			 .batch_normalization(relu=True, name='bn4b16_branch2b')
			 .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b16_branch2c')
			 .batch_normalization(name='bn4b16_branch2c'))

		(self.feed('res4b15_relu',
				   'bn4b16_branch2c')
			 .add(name='res4b16')
			 .relu(name='res4b16_relu')
			 .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b17_branch2a')
			 .batch_normalization(relu=True, name='bn4b17_branch2a')
			 .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='res4b17_branch2b')
			 .batch_normalization(relu=True, name='bn4b17_branch2b')
			 .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b17_branch2c')
			 .batch_normalization(name='bn4b17_branch2c'))

		(self.feed('res4b16_relu',
				   'bn4b17_branch2c')
			 .add(name='res4b17')
			 .relu(name='res4b17_relu')
			 .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b18_branch2a')
			 .batch_normalization(relu=True, name='bn4b18_branch2a')
			 .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='res4b18_branch2b')
			 .batch_normalization(relu=True, name='bn4b18_branch2b')
			 .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b18_branch2c')
			 .batch_normalization(name='bn4b18_branch2c'))

		(self.feed('res4b17_relu',
				   'bn4b18_branch2c')
			 .add(name='res4b18')
			 .relu(name='res4b18_relu')
			 .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b19_branch2a')
			 .batch_normalization(relu=True, name='bn4b19_branch2a')
			 .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='res4b19_branch2b')
			 .batch_normalization(relu=True, name='bn4b19_branch2b')
			 .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b19_branch2c')
			 .batch_normalization(name='bn4b19_branch2c'))

		(self.feed('res4b18_relu',
				   'bn4b19_branch2c')
			 .add(name='res4b19')
			 .relu(name='res4b19_relu')
			 .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b20_branch2a')
			 .batch_normalization(relu=True, name='bn4b20_branch2a')
			 .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='res4b20_branch2b')
			 .batch_normalization(relu=True, name='bn4b20_branch2b')
			 .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b20_branch2c')
			 .batch_normalization(name='bn4b20_branch2c'))

		(self.feed('res4b19_relu',
				   'bn4b20_branch2c')
			 .add(name='res4b20')
			 .relu(name='res4b20_relu')
			 .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b21_branch2a')
			 .batch_normalization(relu=True, name='bn4b21_branch2a')
			 .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='res4b21_branch2b')
			 .batch_normalization(relu=True, name='bn4b21_branch2b')
			 .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b21_branch2c')
			 .batch_normalization(name='bn4b21_branch2c'))

		(self.feed('res4b20_relu',
				   'bn4b21_branch2c')
			 .add(name='res4b21')
			 .relu(name='res4b21_relu')
			 .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b22_branch2a')
			 .batch_normalization(relu=True, name='bn4b22_branch2a')
			 .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='res4b22_branch2b')
			 .batch_normalization(relu=True, name='bn4b22_branch2b')
			 .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b22_branch2c')
			 .batch_normalization(name='bn4b22_branch2c'))

		(self.feed('res4b21_relu',
				   'bn4b22_branch2c')
			 .add(name='res4b22')
			 .relu(name='res4b22_relu')
			 .conv(1, 1, 2048, 2, 2, biased=False, relu=False, name='res5a_branch1')
			 .batch_normalization(name='bn5a_branch1'))

		(self.feed('res4b22_relu')
			 .conv(1, 1, 512, 2, 2, biased=False, relu=False, name='res5a_branch2a')
			 .batch_normalization(relu=True, name='bn5a_branch2a')
			 .conv(3, 3, 512, 1, 1, biased=False, relu=False, name='res5a_branch2b')
			 .batch_normalization(relu=True, name='bn5a_branch2b')
			 .conv(1, 1, 2048, 1, 1, biased=False, relu=False, name='res5a_branch2c')
			 .batch_normalization(name='bn5a_branch2c'))

		(self.feed('bn5a_branch1',
				   'bn5a_branch2c')
			 .add(name='res5a')
			 .relu(name='res5a_relu')
			 .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res5b_branch2a')
			 .batch_normalization(relu=True, name='bn5b_branch2a')
			 .conv(3, 3, 512, 1, 1, biased=False, relu=False, name='res5b_branch2b')
			 .batch_normalization(relu=True, name='bn5b_branch2b')
			 .conv(1, 1, 2048, 1, 1, biased=False, relu=False, name='res5b_branch2c')
			 .batch_normalization(name='bn5b_branch2c'))

		(self.feed('res5a_relu',
				   'bn5b_branch2c')
			 .add(name='res5b')
			 .relu(name='res5b_relu')
			 .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res5c_branch2a')
			 .batch_normalization(relu=True, name='bn5c_branch2a')
			 .conv(3, 3, 512, 1, 1, biased=False, relu=False, name='res5c_branch2b')
			 .batch_normalization(relu=True, name='bn5c_branch2b')
			 .conv(1, 1, 2048, 1, 1, biased=False, relu=False, name='res5c_branch2c')
			 .batch_normalization(name='bn5c_branch2c'))

		(self.feed('res5b_relu',
				   'bn5c_branch2c')
			 .add(name='res5c')
			 .relu(name='res5c_relu')
			 .avg_pool(7, 7, 1, 1, padding='VALID', name='pool5')
			 .fc(1000, relu=False, name='fc1000'))
