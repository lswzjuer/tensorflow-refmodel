"""
Reference Model
Copyright (c) 2018 MobaiTech Inc 
Author: Abinash Mohanty
Description: 
	Class for ssdJacintoNetV2 
	Input data is Batch_size x 400 x 640 x 3. 
	For inference in our case Batch_size == 1
	Pre-processing function
"""
from .network import Network
import tensorflow as tf
import numpy as np
import os
import sys

class ssdJacintoNetV2(Network):

	def __init__(self, isHardware=True, trainable=False):
		self.inputs = []
		self.data = tf.placeholder(tf.float32, shape=[None, None, None, 3])
		self.layers = dict({'data': self.data})
		self.trainable = trainable
		self.isHardware=isHardware
		self._layer_map = {}
		self.create_layer_map()
		self.setup()

	def pre_process(self, im, scale_mode=0):
		"""
		MUST HAVE FUNCTION IN ALL NETWORKS !!!! 
		Pre-processing of the image. De-mean, crop and resize
		Returns the feed dictionary the network is expecting
		For VGG there is just one size image 224x224 at the moment.		
		TODO: look into the exact pre-processing needed 
		"""
		im_orig = im.astype(np.uint8, copy=True)
		im_bgr = cv2.resize(im_orig, (640,400))
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
		print ('TODO. POST PROCESS NOT YET IMPLEMENTED')	
	
	def run_sw_demo(self, img, scale_mode=0):
		"""
		Pure software demo.	
		"""	
		print('Running software only demo ... ')
		im = cv2.imread(image_path)	
		feed_dict = self.pre_process(im, scale_mode)
		ctx_output5_relu_mbox_loc, ctx_output5_relu_mbox_conf, ctx_output4_relu_mbox_loc, ctx_output4_relu_mbox_conf, ctx_output3_relu_mbox_loc, ctx_output3_relu_mbox_conf, ctx_output2_relu_mbox_loc, ctx_output2_relu_mbox_conf, ctx_output1_relu_mbox_loc, ctx_output1_relu_mbox_conf, ctx_final = sess.run([self.get_output('ctx_output5_relu_mbox_loc'), self.get_output('ctx_output5_relu_mbox_conf'), self.get_output('ctx_output4_relu_mbox_loc'), self.get_output('ctx_output4_relu_mbox_conf'), self.get_output('ctx_output3_relu_mbox_loc'), self.get_output('ctx_output3_relu_mbox_conf'), self.get_output('ctx_output2_relu_mbox_loc'), self.get_output('ctx_output2_relu_mbox_conf'), self.get_output('ctx_output1_relu_mbox_loc'), self.get_output('ctx_output1_relu_mbox_conf'), self.get_output('ctx_final')], feed_dict=feed_dict)

		# Post process on network output and show result 
		self.post_process(im, [ctx_output5_relu_mbox_loc, ctx_output5_relu_mbox_conf, ctx_output4_relu_mbox_loc, ctx_output4_relu_mbox_conf, ctx_output3_relu_mbox_loc, ctx_output3_relu_mbox_conf, ctx_output2_relu_mbox_loc, ctx_output2_relu_mbox_conf, ctx_output1_relu_mbox_loc, ctx_output1_relu_mbox_conf, ctx_final])


	def create_layer_map(self):
		""" 
		Helper Function for Validation. Dictionary of layer wise parameters. 
		TODO: get correct fl and fl_w values and update this file 
		"""
		self._layer_map[0]={'name':'data',							'inputs':[],	'fl':6,	'fl_w':15}
		self._layer_map[1]={'name':'conv1a',						'inputs':[0],	'fl':6,	'fl_w':15}
		self._layer_map[2]={'name':'conv1b',						'inputs':[1],	'fl':6,	'fl_w':15}
		self._layer_map[3]={'name':'pool1',							'inputs':[2],	'fl':6,	'fl_w':15}
		self._layer_map[4]={'name':'res2a_branch2a',				'inputs':[3],	'fl':6,	'fl_w':15}
		self._layer_map[5]={'name':'res2a_branch2b',				'inputs':[4],	'fl':6,	'fl_w':15}
		self._layer_map[6]={'name':'pool2',							'inputs':[5],	'fl':6,	'fl_w':15}
		self._layer_map[7]={'name':'res3a_branch2a',				'inputs':[6],	'fl':6,	'fl_w':15}
		self._layer_map[8]={'name':'res3a_branch2b',				'inputs':[7],	'fl':6,	'fl_w':15}
		self._layer_map[9]={'name':'pool3',							'inputs':[8],	'fl':6,	'fl_w':15}
		self._layer_map[10]={'name':'res4a_branch2a',				'inputs':[9],	'fl':6,	'fl_w':15}
		self._layer_map[11]={'name':'res4a_branch2b',				'inputs':[10],	'fl':6,	'fl_w':15}
		self._layer_map[12]={'name':'pool4',						'inputs':[11],	'fl':6,	'fl_w':15}
		self._layer_map[13]={'name':'res5a_branch2a',				'inputs':[12],	'fl':6,	'fl_w':15}
		self._layer_map[14]={'name':'res5a_branch2b',				'inputs':[13],	'fl':6,	'fl_w':15}
		self._layer_map[15]={'name':'pool6',						'inputs':[15],	'fl':6,	'fl_w':15}
		self._layer_map[16]={'name':'pool7',						'inputs':[16],	'fl':6,	'fl_w':15}
		self._layer_map[17]={'name':'pool8',						'inputs':[17],	'fl':6,	'fl_w':15}
		self._layer_map[18]={'name':'ctx_output1',					'inputs':[11],	'fl':6,	'fl_w':15}
		self._layer_map[19]={'name':'ctx_output2',					'inputs':[14],	'fl':6,	'fl_w':15}
		self._layer_map[20]={'name':'ctx_output3',					'inputs':[15],	'fl':6,	'fl_w':15}
		self._layer_map[21]={'name':'ctx_output4',					'inputs':[16],	'fl':6,	'fl_w':15}
		self._layer_map[22]={'name':'ctx_output5',					'inputs':[17],	'fl':6,	'fl_w':15}
		self._layer_map[23]={'name':'ctx_output1_relu_mbox_loc',	'inputs':[18],	'fl':6,	'fl_w':15}
		self._layer_map[24]={'name':'ctx_output1_relu_mbox_conf',	'inputs':[18],	'fl':6,	'fl_w':15}
		self._layer_map[25]={'name':'ctx_output2_relu_mbox_loc',	'inputs':[19],	'fl':6,	'fl_w':15}
		self._layer_map[26]={'name':'ctx_output2_relu_mbox_conf',	'inputs':[19],	'fl':6,	'fl_w':15}
		self._layer_map[27]={'name':'ctx_output3_relu_mbox_loc',	'inputs':[20],	'fl':6,	'fl_w':15}
		self._layer_map[28]={'name':'ctx_output3_relu_mbox_conf',	'inputs':[20],	'fl':6,	'fl_w':15}
		self._layer_map[29]={'name':'ctx_output4_relu_mbox_loc',	'inputs':[21],	'fl':6,	'fl_w':15}
		self._layer_map[30]={'name':'ctx_output4_relu_mbox_conf',	'inputs':[21],	'fl':6,	'fl_w':15}
		self._layer_map[31]={'name':'ctx_output5_relu_mbox_loc',	'inputs':[22],	'fl':6,	'fl_w':15}
		self._layer_map[32]={'name':'ctx_output5_relu_mbox_conf',	'inputs':[22],	'fl':6,	'fl_w':15}
		self._layer_map[33]={'name':'out3a',						'inputs':[8],	'fl':6,	'fl_w':15}
		self._layer_map[34]={'name':'ctx_conv1',					'inputs':[33],	'fl':6,	'fl_w':15}
		self._layer_map[35]={'name':'ctx_conv2',					'inputs':[34],	'fl':6,	'fl_w':15}
		self._layer_map[36]={'name':'ctx_conv3',					'inputs':[35],	'fl':6,	'fl_w':15}
		self._layer_map[37]={'name':'ctx_conv4',					'inputs':[36],	'fl':6,	'fl_w':15}
		self._layer_map[38]={'name':'ctx_final',					'inputs':[37],	'fl':6,	'fl_w':15}


	def setup(self):
		(self.feed('data')
			 .conv(5, 5, 32, 2, 2, fl=6, name='conv1a')
			 .conv(3, 3, 32, 1, 1, fl=6, group=4, name='conv1b')
			 .max_pool(2, 2, 2, 2, fl=6, name='pool1')
			 .conv(3, 3, 64, 1, 1, fl=6, name='res2a_branch2a')
			 .conv(3, 3, 64, 1, 1, fl=6, group=4, name='res2a_branch2b')
			 .max_pool(2, 2, 2, 2, fl=6, name='pool2')
			 .conv(3, 3, 128, 1, 1, fl=6, name='res3a_branch2a')
			 .conv(3, 3, 128, 1, 1, fl=6, group=4, name='res3a_branch2b')
			 .max_pool(2, 2, 2, 2, fl=6, name='pool3')
			 .conv(3, 3, 256, 1, 1, fl=6, name='res4a_branch2a')
			 .conv(3, 3, 256, 1, 1, fl=6, group=4, name='res4a_branch2b')
			 .max_pool(2, 2, 2, 2, fl=6, name='pool4')
			 .conv(3, 3, 512, 1, 1, fl=6, name='res5a_branch2a')
			 .conv(3, 3, 512, 1, 1, fl=6, group=4, name='res5a_branch2b')
			 .max_pool(2, 2, 2, 2, fl=6, name='pool6')
			 .max_pool(2, 2, 2, 2, fl=6, name='pool7')
			 .max_pool(2, 2, 2, 2, fl=6, name='pool8')
			 .conv(1, 1, 256, 1, 1, fl=6, name='ctx_output5')
			 .conv(1, 1, 24, 1, 1, fl=6, relu=False, name='ctx_output5_relu_mbox_loc'))

		(self.feed('res4a_branch2b')
			 .conv(1, 1, 256, 1, 1, name='ctx_output1')
			 .conv(1, 1, 24, 1, 1, relu=False, name='ctx_output1_relu_mbox_loc'))

		(self.feed('res5a_branch2b')
			 .conv(1, 1, 256, 1, 1, name='ctx_output2')
			 .conv(1, 1, 24, 1, 1, relu=False, name='ctx_output2_relu_mbox_loc'))

		(self.feed('pool6')
			 .conv(1, 1, 256, 1, 1, name='ctx_output3')
			 .conv(1, 1, 24, 1, 1, relu=False, name='ctx_output3_relu_mbox_loc'))

		(self.feed('pool7')
			 .conv(1, 1, 256, 1, 1, name='ctx_output4')
			 .conv(1, 1, 24, 1, 1, relu=False, name='ctx_output4_relu_mbox_loc'))

		(self.feed('ctx_output1')
			 .conv(1, 1, 54, 1, 1, relu=False, name='ctx_output1_relu_mbox_conf'))

		(self.feed('ctx_output2')
			 .conv(1, 1, 54, 1, 1, relu=False, name='ctx_output2_relu_mbox_conf'))

		(self.feed('ctx_output3')
			 .conv(1, 1, 54, 1, 1, relu=False, name='ctx_output3_relu_mbox_conf'))

		(self.feed('ctx_output4')
			 .conv(1, 1, 54, 1, 1, relu=False, name='ctx_output4_relu_mbox_conf'))

		(self.feed('ctx_output5')
			 .conv(1, 1, 54, 1, 1, relu=False, name='ctx_output5_relu_mbox_conf'))

		(self.feed('res3a_branch2b')
			 .conv(3, 3, 64, 1, 1, group=2, name='out3a')
			 .conv(3, 3, 64, 1, 1, name='ctx_conv1')
			 .conv(3, 3, 64, 1, 1, padding=None, name='ctx_conv2')
			 .conv(3, 3, 64, 1, 1, padding=None, name='ctx_conv3')
			 .conv(3, 3, 64, 1, 1, padding=None, name='ctx_conv4')
			 .conv(3, 3, 6, 1, 1, name='ctx_final'))
