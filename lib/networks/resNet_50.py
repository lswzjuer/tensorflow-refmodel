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
from utils.refModel_log import print_msg

class resnet50(Network):
	
	def __init__(self, isHardware=True, trainable=False):
                cfg.TEST.SCALES = (224,)
                cfg.TEST.MAX_SIZE = 224
		self.inputs = []
		self.trainable = trainable
		self.isHardware = isHardware
		self.data = tf.placeholder(tf.float32, shape=[None, None, None, 3])
		self.im_info = tf.placeholder(tf.float32, shape=[None,3])
		self.keep_prob = tf.placeholder(tf.float32)
		self.layers = dict({'data':self.data, 'im_info':self.im_info})		
		#self.layers = dict({'data':self.data})
		self.trainable = trainable
		self.n_classes = 1000
		self.classes = imageNet_classNames()
		self._layer_map = {}
		self.create_layer_map()
		self.setup()
		print_msg ('ResNet50 Network setup done',0)


	def _get_image_blob(self, im, im_dim):
                """Converts an image into a network input.
                
                Arguments:
                    im (ndarray): a color image in BGR order
                
                Returns:
                    blob (ndarray): a data blob holding an image pyramid
                    im_scale_factors (list): list of image scales (relative to im) used
                        in the image pyramid
                """
                im = im.astype(np.float32, copy=True)
                
                print('Image im.shape = {}'.format(im.shape))
                print(im[0:10,0:10,0])
                print(im[0:10,0:10,1])
                print(im[0:10,0:10,2])
                
                im = im - cfg.PIXEL_MEANS
                im /= cfg.PIXEL_STDS
                
                print('\nAfter substract mean')
                print(im[0:10,0:10,0])
                print(im[0:10,0:10,1])
                print(im[0:10,0:10,2])
                
                im_shape = im.shape
                im_size_min = np.min(im_shape[0:2])
                im_size_max = np.max(im_shape[0:2])
                
                processed_ims = []
                im_scale_factors = []
               
		print('cfg.TEST.SCALES = {},cfg.TEST.MAX_SIZE = {}, im_dim[1] = {}'.format(cfg.TEST.SCALES,cfg.TEST.MAX_SIZE,im_dim[1]))
		TEST_MAX_SIZE = im_dim[1]

                for target_size in cfg.TEST.SCALES:
                    im_scale = float(target_size) / float(im_size_min)
                    # Prevent the biggest axis from being more than MAX_SIZE
                    if np.round(im_scale * im_size_max) > TEST_MAX_SIZE:
                        im_scale = float(TEST_MAX_SIZE) / float(im_size_max)
                    multiple = cfg.TEST.SCALE_MULTIPLE_OF
                    if multiple > 1:
                        im_scale_x = np.floor(im.shape[1] * im_scale / multiple) * multiple / im.shape[1]
                        im_scale_y = np.floor(im.shape[0] * im_scale / multiple) * multiple / im.shape[0]
                        print('im_scale = {}, multiple = {}'.format(im_scale,multiple))
                        print('im.shape[0] = {}, im.shape[1] = {}'.format(im.shape[0],im.shape[1]))
                        print('im_scale_x = {}, im_scale_y = {}\n'.format(im_scale_x,im_scale_y))
                        im = cv2.resize(im, None, None, fx=im_scale_x, fy=im_scale_y, interpolation=cv2.INTER_LINEAR)
                
                    else:
                        im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
                        print('im_scale = {}\n'.format(im_scale))
                    im_scale_factors.append(im_scale)
                    processed_ims.append(im)
                
                print('\nAfter resize')
                #print(im[0:10,0:10,0])
                #print(im[0:10,0:10,1])
                #print(im[0:10,0:10,2])
                
                # Create a blob to hold the input images
                
                blob = im_list_to_blob(processed_ims)
                
                blob = blob.astype(np.int32, copy=True)
                blob = blob.astype(np.float32, copy=True)
                blob = np.clip(blob,-128,127)
                
                print(blob[0,0:10,0:10,0])
                print(blob[0,0:10,0:10,1])
                print(blob[0,0:10,0:10,2])
                #np.savetxt('a0_processed_ims0.csv',blob[0,:,:,0],fmt='%d,')
                #np.savetxt('a0_processed_ims1.csv',blob[0,:,:,1],fmt='%d,')
                #np.savetxt('a0_processed_ims2.csv',blob[0,:,:,2],fmt='%d,')
                print('blob.shape = {}'.format(blob.shape))
                
                return blob, np.array(im_scale_factors)

	def pre_process(self, im, scale_mode=0):
		"""
		MUST HAVE FUNCTION IN ALL NETWORKS !!!! 
		Pre-processing of the image. De-mean, crop and resize
		Returns the feed dictionary the network is expecting
		For VGG there is just one size image 224x224 at the moment.		
		"""
		#im_orig = im.astype(np.uint8, copy=True)
		#im_bgr = cv2.resize(im_orig, (224,224))
		#img_mean = np.zeros(shape = im_bgr.shape)
		#img_mean[:, :, 0] = cfg.PIXEL_MEANS[0]
		#img_mean[:, :, 1] = cfg.PIXEL_MEANS[1]
		#img_mean[:, :, 2] = cfg.PIXEL_MEANS[2]
		#img_demean = im_bgr-img_mean
		#processed_ims = [img_demean]
		#blob = im_list_to_blob(processed_ims)		 
		#feed_dict = {self.data: blob}
		#return feed_dict
                xmin = 500
                ymin = 100
                width = 224
                height = 224

		im = im[ymin:ymin+height,xmin:xmin+width]

		blobs = {'data' : None, 'rois' : None}
                im_dim = [224,224]
		blobs['data'], im_scales = self._get_image_blob(im, im_dim)	
		blobs['im_info'] = np.array([[blobs['data'].shape[1], blobs['data'].shape[2], im_scales[0]]],dtype=np.float32)
		feed_dict={self.data: blobs['data'], self.im_info: blobs['im_info'], self.keep_prob: 1.0}
		return feed_dict

	def post_process(self, im, sim_ops, scale_factor):
		"""
		MUST HAVE FUNCTION IN ALL NETWORKS !!!! 
		Post-processing of the results from network. This function can be used to visualize data from hardware.  
		"""
		prob = softmax(sim_ops[0][0])
		preds = (np.argsort(prob)[::-1])[0:5]
		for p in preds:
		    print_msg(str(classes[p]) +' , ' + str(prob[p]), 3)		
	
	def run_sw_demo(self, sess, img, scale_mode=0):
		"""
		Pure software demo.
		"""	
		print_msg('Running software only demo ... ', 3)
		im = cv2.imread(img)	
		feed_dict = self.pre_process(im, scale_mode)
		fc1000 = sess.run([self.get_output('fc1000')], feed_dict=feed_dict)
		self.post_process(im, fc1000)		# Post process on network output and show result

	def create_layer_map(self):
		""" 
		Helper Function for Validation. Dictionary of layer wise parameters. 
		TODO: get correct fl and fl_w values and update this file 
		"""
		self._layer_map[ 0]={'name':'data',		'inputs':[],		'fl':6,	'fl_w':15, 'type':'data'}
		self._layer_map[ 1]={'name':'conv1',		'inputs':[0],		'fl':6,	'fl_w':15,'type':'conv'}
		self._layer_map[ 2]={'name':'pool1',		'inputs':[1],		'fl':6,	'fl_w':15,'type':'plmx'}
		self._layer_map[ 3]={'name':'res2a_branch1',	'inputs':[2],		'fl':6,	'fl_w':15,'type':'conv'}
		self._layer_map[ 4]={'name':'res2a_branch2a',	'inputs':[2],		'fl':6,	'fl_w':15,'type':'conv'}
		self._layer_map[ 5]={'name':'res2a_branch2b',	'inputs':[4],		'fl':6,	'fl_w':15,'type':'conv'}
		self._layer_map[ 6]={'name':'res2a_branch2c',	'inputs':[5],		'fl':6,	'fl_w':15,'type':'conv'}
		self._layer_map[ 7]={'name':'res2a',			'inputs':[3,6],		'fl':6,	'fl_w':15,'type':'ewis'}
		self._layer_map[ 8]={'name':'res2b_branch2a',	'inputs':[7],		'fl':6,	'fl_w':15,'type':'conv'}
		self._layer_map[ 9]={'name':'res2b_branch2b',	'inputs':[8],		'fl':6,	'fl_w':15,'type':'conv'}
		self._layer_map[10]={'name':'res2b_branch2c',	'inputs':[9],		'fl':6,	'fl_w':15,'type':'conv'}
		self._layer_map[11]={'name':'res2b',			'inputs':[7],		'fl':6,	'fl_w':15,'type':'ewis'}
		self._layer_map[12]={'name':'res2c_branch2a',	'inputs':[11],		'fl':6,	'fl_w':15,'type':'conv'}
		self._layer_map[13]={'name':'res2c_branch2b',	'inputs':[12],		'fl':6,	'fl_w':15,'type':'conv'}
		self._layer_map[14]={'name':'res2c_branch2c',	'inputs':[13],		'fl':6,	'fl_w':15,'type':'conv'}
		self._layer_map[15]={'name':'res2c',			'inputs':[11,13],	'fl':6,	'fl_w':15,'type':'ewis'}
		self._layer_map[16]={'name':'res3a_branch1',	'inputs':[15],		'fl':6,	'fl_w':15,'type':'conv'}
		self._layer_map[17]={'name':'res3a_branch2a',	'inputs':[15],		'fl':6,	'fl_w':15,'type':'conv'}
		self._layer_map[18]={'name':'res3a_branch2b',	'inputs':[17],		'fl':6,	'fl_w':15,'type':'conv'}
		self._layer_map[19]={'name':'res3a_branch2c',	'inputs':[18],		'fl':6,	'fl_w':15,'type':'conv'}
		self._layer_map[20]={'name':'res3a',			'inputs':[16,19],	'fl':6,	'fl_w':15,'type':'ewis'}
		self._layer_map[21]={'name':'res3b_branch2a',	'inputs':[20],		'fl':6,	'fl_w':15,'type':'conv'}
		self._layer_map[22]={'name':'res3b_branch2b',	'inputs':[21],		'fl':6,	'fl_w':15,'type':'conv'}
		self._layer_map[23]={'name':'res3b_branch2c',	'inputs':[22],		'fl':6,	'fl_w':15,'type':'conv'}
		self._layer_map[24]={'name':'res3b',			'inputs':[20,23],	'fl':6,	'fl_w':15,'type':'ewis'}
		self._layer_map[25]={'name':'res3c_branch2a',	'inputs':[24],		'fl':6,	'fl_w':15,'type':'conv'}
		self._layer_map[26]={'name':'res3c_branch2b',	'inputs':[25],		'fl':6,	'fl_w':15,'type':'conv'}
		self._layer_map[27]={'name':'res3c_branch2c',	'inputs':[26],		'fl':6,	'fl_w':15,'type':'conv'}
		self._layer_map[28]={'name':'res3c',			'inputs':[24,27],	'fl':6,	'fl_w':15,'type':'ewis'}
		self._layer_map[29]={'name':'res3d_branch2a',	'inputs':[28],		'fl':6,	'fl_w':15,'type':'conv'}
		self._layer_map[30]={'name':'res3d_branch2b',	'inputs':[29],		'fl':6,	'fl_w':15,'type':'conv'}
		self._layer_map[31]={'name':'res3d_branch2c',	'inputs':[30],		'fl':6,	'fl_w':15,'type':'conv'}
		self._layer_map[32]={'name':'res3d',			'inputs':[28,31],	'fl':6,	'fl_w':15,'type':'ewis'}
		self._layer_map[33]={'name':'res4a_branch1',	'inputs':[32],		'fl':6,	'fl_w':15,'type':'conv'}
		self._layer_map[34]={'name':'res4a_branch2a',	'inputs':[32],		'fl':6,	'fl_w':15,'type':'conv'}
		self._layer_map[35]={'name':'res4a_branch2b',	'inputs':[34],		'fl':6,	'fl_w':15,'type':'conv'}
		self._layer_map[36]={'name':'res4a_branch2c',	'inputs':[35],		'fl':6,	'fl_w':15,'type':'conv'}
		self._layer_map[37]={'name':'res4a',			'inputs':[33,36],	'fl':6,	'fl_w':15,'type':'ewis'}
		self._layer_map[38]={'name':'res4b_branch2a',	'inputs':[37],		'fl':6,	'fl_w':15,'type':'conv'}
		self._layer_map[39]={'name':'res4b_branch2b',	'inputs':[38],		'fl':6,	'fl_w':15,'type':'conv'}
		self._layer_map[40]={'name':'res4b_branch2c',	'inputs':[39],		'fl':6,	'fl_w':15,'type':'conv'}
		self._layer_map[41]={'name':'res4b',			'inputs':[37,40],	'fl':6,	'fl_w':15,'type':'ewis'}
		self._layer_map[42]={'name':'res4c_branch2a',	'inputs':[41],		'fl':6,	'fl_w':15,'type':'conv'}
		self._layer_map[43]={'name':'res4c_branch2b',	'inputs':[42],		'fl':6,	'fl_w':15,'type':'conv'}
		self._layer_map[44]={'name':'res4c_branch2c',	'inputs':[43],		'fl':6,	'fl_w':15,'type':'conv'}
		self._layer_map[45]={'name':'res4c',			'inputs':[41,44],	'fl':6,	'fl_w':15,'type':'ewis'}
		self._layer_map[46]={'name':'res4d_branch2a',	'inputs':[45],		'fl':6,	'fl_w':15,'type':'conv'}
		self._layer_map[47]={'name':'res4d_branch2b',	'inputs':[46],		'fl':6,	'fl_w':15,'type':'conv'}
		self._layer_map[48]={'name':'res4d_branch2c',	'inputs':[47],		'fl':6,	'fl_w':15,'type':'conv'}
		self._layer_map[49]={'name':'res4d',			'inputs':[45,48],	'fl':6,	'fl_w':15,'type':'ewis'}
		self._layer_map[50]={'name':'res4e_branch2a',	'inputs':[49],		'fl':6,	'fl_w':15,'type':'conv'}
		self._layer_map[51]={'name':'res4e_branch2b',	'inputs':[50],		'fl':6,	'fl_w':15,'type':'conv'}
		self._layer_map[52]={'name':'res4e_branch2c',	'inputs':[51],		'fl':6,	'fl_w':15,'type':'conv'}
		self._layer_map[53]={'name':'res4e',			'inputs':[49,52],	'fl':6,	'fl_w':15,'type':'ewis'}
		self._layer_map[54]={'name':'res4f_branch2a',	'inputs':[53],		'fl':6,	'fl_w':15,'type':'conv'}
		self._layer_map[55]={'name':'res4f_branch2b',	'inputs':[54],		'fl':6,	'fl_w':15,'type':'conv'}
		self._layer_map[56]={'name':'res4f_branch2c',	'inputs':[55],		'fl':6,	'fl_w':15,'type':'conv'}
		self._layer_map[57]={'name':'res4f',			'inputs':[53,56],		'fl':6,	'fl_w':15,'type':'ewis'}
		self._layer_map[58]={'name':'res5a_branch1',	'inputs':[57],		'fl':6,	'fl_w':15,'type':'conv'}
		self._layer_map[59]={'name':'res5a_branch2a',	'inputs':[57],		'fl':6,	'fl_w':15,'type':'conv'}
		self._layer_map[60]={'name':'res5a_branch2b',	'inputs':[59],		'fl':6,	'fl_w':15,'type':'conv'}
		self._layer_map[61]={'name':'res5a_branch2c',	'inputs':[60],		'fl':6,	'fl_w':15,'type':'conv'}
		self._layer_map[62]={'name':'res5a',			'inputs':[58,61],		'fl':6,	'fl_w':15,'type':'ewis'}
		self._layer_map[63]={'name':'res5b_branch2a',	'inputs':[62],		'fl':6,	'fl_w':15,'type':'conv'}
		self._layer_map[64]={'name':'res5b_branch2b',	'inputs':[63],		'fl':6,	'fl_w':15,'type':'conv'}
		self._layer_map[65]={'name':'res5b_branch2c',	'inputs':[64],		'fl':6,	'fl_w':15,'type':'conv'}
		self._layer_map[66]={'name':'res5b',			'inputs':[62,65],		'fl':6,	'fl_w':15,'type':'ewis'}
		self._layer_map[67]={'name':'res5c_branch2a',	'inputs':[66],		'fl':6,	'fl_w':15,'type':'conv'}
		self._layer_map[68]={'name':'res5c_branch2b',	'inputs':[67],		'fl':6,	'fl_w':15,'type':'conv'}
		self._layer_map[69]={'name':'res5c_branch2c',	'inputs':[68],		'fl':6,	'fl_w':15,'type':'conv'}
		self._layer_map[70]={'name':'res5c',			'inputs':[66,69],		'fl':6,	'fl_w':15,'type':'ewis'}
		self._layer_map[71]={'name':'pool5',			'inputs':[70],		'fl':6,	'fl_w':15,'type':'gapl'}
		self._layer_map[72]={'name':'fc1000',			'inputs':[71],		'fl':6,	'fl_w':15,'type':'fcon'}

	def setup(self):
		"""
		Network defination and graph implementation.	
		TODO: Commented batch_normalization for merging. Use the merge batch normalization script to do this. 
		TODO: get correct fl and fl_w values and update this file 
		"""	
		(self.feed('data')
			 .conv(7, 7, 64, 2, 2, relu=False, name='conv1')
			 #.batch_normalizationg(relu=True, name='bn_conv1')
			 .max_pool(3, 3, 2, 2, name='pool1')
			 .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res2a_branch1'))
			 #.batch_normalizationg(name='bn2a_branch1'))

		(self.feed('pool1')
			 .conv(1, 1, 64, 1, 1, biased=False, relu=False, name='res2a_branch2a')
			 ##.batch_normalizationg(relu=True, name='bn2a_branch2a')
			 .conv(3, 3, 64, 1, 1, biased=False, relu=False, name='res2a_branch2b')
			 #.batch_normalizationg(relu=True, name='bn2a_branch2b')
			 .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res2a_branch2c'))
			 #.batch_normalizationg(name='bn2a_branch2c'))

		(self.feed('res2a_branch1','res2a_branch2c')
			 .add(name='res2a', relu=True)
			 .conv(1, 1, 64, 1, 1, biased=False, relu=False, name='res2b_branch2a')
			 #.batch_normalizationg(relu=True, name='bn2b_branch2a')
			 .conv(3, 3, 64, 1, 1, biased=False, relu=False, name='res2b_branch2b')
			 #.batch_normalizationg(relu=True, name='bn2b_branch2b')
			 .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res2b_branch2c'))
			 #.batch_normalizationg(name='bn2b_branch2c'))

		(self.feed('res2a','res2b_branch2c')
			 .add(name='res2b', relu=True)
			 .conv(1, 1, 64, 1, 1, biased=False, relu=False, name='res2c_branch2a')
			 #.batch_normalizationg(relu=True, name='bn2c_branch2a')
			 .conv(3, 3, 64, 1, 1, biased=False, relu=False, name='res2c_branch2b')
			 #.batch_normalizationg(relu=True, name='bn2c_branch2b')
			 .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res2c_branch2c'))
			 #.batch_normalizationg(name='bn2c_branch2c'))

		(self.feed('res2b','res2c_branch2c')
			 .add(name='res2c')
			 .conv(1, 1, 512, 2, 2, biased=False, relu=False, name='res3a_branch1'))
			 #.batch_normalizationg(name='bn3a_branch1'))

		(self.feed('res2c')
			 .conv(1, 1, 128, 2, 2, biased=False, relu=False, name='res3a_branch2a')
			 #.batch_normalizationg(relu=True, name='bn3a_branch2a')
			 .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='res3a_branch2b')
			 #.batch_normalizationg(relu=True, name='bn3a_branch2b')
			 .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res3a_branch2c'))
			 #.batch_normalizationg(name='bn3a_branch2c'))

		(self.feed('res3a_branch1','res3a_branch2c')
			 .add(name='res3a')
			 .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='res3b_branch2a')
			 #.batch_normalizationg(relu=True, name='bn3b_branch2a')
			 .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='res3b_branch2b')
			 #.batch_normalizationg(relu=True, name='bn3b_branch2b')
			 .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res3b_branch2c'))
			 #.batch_normalizationg(name='bn3b_branch2c'))

		(self.feed('res3a','res3b_branch2c')
			 .add(name='res3b')
			 .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='res3c_branch2a')
			 #.batch_normalizationg(relu=True, name='bn3c_branch2a')
			 .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='res3c_branch2b')
			 #.batch_normalizationg(relu=True, name='bn3c_branch2b')
			 .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res3c_branch2c'))
			 #.batch_normalizationg(name='bn3c_branch2c'))

		(self.feed('res3b','res3c_branch2c')
			 .add(name='res3c')
			 .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='res3d_branch2a')
			 #.batch_normalizationg(relu=True, name='bn3d_branch2a')
			 .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='res3d_branch2b')
			 #.batch_normalizationg(relu=True, name='bn3d_branch2b')
			 .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res3d_branch2c'))
			 #.batch_normalizationg(name='bn3d_branch2c'))

		(self.feed('res3c','res3d_branch2c')
			 .add(name='res3d')
			 .conv(1, 1, 1024, 2, 2, biased=False, relu=False, name='res4a_branch1'))
			 #.batch_normalizationg(name='bn4a_branch1'))

		(self.feed('res3d')
			 .conv(1, 1, 256, 2, 2, biased=False, relu=False, name='res4a_branch2a')
			 #.batch_normalizationg(relu=True, name='bn4a_branch2a')
			 .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='res4a_branch2b')
			 #.batch_normalizationg(relu=True, name='bn4a_branch2b')
			 .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4a_branch2c'))
			 #.batch_normalizationg(name='bn4a_branch2c'))

		(self.feed('res4a_branch1','res4a_branch2c')
			 .add(name='res4a')
			 .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b_branch2a')
			 #.batch_normalizationg(relu=True, name='bn4b_branch2a')
			 .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='res4b_branch2b')
			 #.batch_normalizationg(relu=True, name='bn4b_branch2b')
			 .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b_branch2c'))
			 #.batch_normalizationg(name='bn4b_branch2c'))

		(self.feed('res4a','res4b_branch2c')
			 .add(name='res4b')
			 .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4c_branch2a')
			 #.batch_normalizationg(relu=True, name='bn4c_branch2a')
			 .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='res4c_branch2b')
			 #.batch_normalizationg(relu=True, name='bn4c_branch2b')
			 .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4c_branch2c'))
			 #.batch_normalizationg(name='bn4c_branch2c'))

		(self.feed('res4b','res4c_branch2c')
			 .add(name='res4c')
			 .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4d_branch2a')
			 #.batch_normalizationg(relu=True, name='bn4d_branch2a')
			 .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='res4d_branch2b')
			 #.batch_normalizationg(relu=True, name='bn4d_branch2b')
			 .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4d_branch2c'))
			 #.batch_normalizationg(name='bn4d_branch2c'))

		(self.feed('res4c','res4d_branch2c')
			 .add(name='res4d')
			 .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4e_branch2a')
			 #.batch_normalizationg(relu=True, name='bn4e_branch2a')
			 .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='res4e_branch2b')
			 #.batch_normalizationg(relu=True, name='bn4e_branch2b')
			 .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4e_branch2c'))
			 #.batch_normalizationg(name='bn4e_branch2c'))

		(self.feed('res4d','res4e_branch2c')
			 .add(name='res4e')
			 .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4f_branch2a')
			 #.batch_normalizationg(relu=True, name='bn4f_branch2a')
			 .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='res4f_branch2b')
			 #.batch_normalizationg(relu=True, name='bn4f_branch2b')
			 .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4f_branch2c'))
			 #.batch_normalizationg(name='bn4f_branch2c'))

		(self.feed('res4e','res4f_branch2c')
			 .add(name='res4f')
			 .conv(1, 1, 2048, 2, 2, biased=False, relu=False, name='res5a_branch1'))
			 #.batch_normalizationg(name='bn5a_branch1'))

		(self.feed('res4f')
			 .conv(1, 1, 512, 2, 2, biased=False, relu=False, name='res5a_branch2a')
			 #.batch_normalizationg(relu=True, name='bn5a_branch2a')
			 .conv(3, 3, 512, 1, 1, biased=False, relu=False, name='res5a_branch2b')
			 #.batch_normalizationg(relu=True, name='bn5a_branch2b')
			 .conv(1, 1, 2048, 1, 1, biased=False, relu=False, name='res5a_branch2c'))
			 #.batch_normalizationg(name='bn5a_branch2c'))

		(self.feed('res5a_branch1','res5a_branch2c')
			 .add(name='res5a')
			 .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res5b_branch2a')
			 #.batch_normalizationg(relu=True, name='bn5b_branch2a')
			 .conv(3, 3, 512, 1, 1, biased=False, relu=False, name='res5b_branch2b')
			 #.batch_normalizationg(relu=True, name='bn5b_branch2b')
			 .conv(1, 1, 2048, 1, 1, biased=False, relu=False, name='res5b_branch2c'))
			 #.batch_normalizationg(name='bn5b_branch2c'))

		(self.feed('res5a','res5b_branch2c')
			 .add(name='res5b')
			 .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res5c_branch2a')
			 #.batch_normalizationg(relu=True, name='bn5c_branch2a')
			 .conv(3, 3, 512, 1, 1, biased=False, relu=False, name='res5c_branch2b')
			 #.batch_normalizationg(relu=True, name='bn5c_branch2b')
			 .conv(1, 1, 2048, 1, 1, biased=False, relu=False, name='res5c_branch2c'))
			 #.batch_normalizationg(name='bn5c_branch2c'))

		(self.feed('res5b','res5c_branch2c')
			 .add(name='res5c')
			 .avg_pool(7, 7, 1, 1, padding='VALID', name='pool5'))
			 #.fc(num_out=1000, relu=False, name='fc1000'))
