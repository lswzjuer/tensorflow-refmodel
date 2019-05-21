"""
Reference Model
Copyright (c) 2018 MobaiTech Inc 
Author: Abinash Mohanty
"""

from .network import Network
import tensorflow as tf
from config import cfg
from utils.blob import im_list_to_blob
from utils.bbox_transform import bbox_transform_inv
from utils.refModel_log import print_msg
from utils.nms_wrapper import nms
import matplotlib.pyplot as plt
import numpy as np
import cv2
import json

class pvanet_roip8(Network):

	def build_json(self, name, npa):
		temp = npa.tolist()
		fname = name + ".json"
		f = open(fname,'w')
		f.write(name + ' shape: ' + str(npa.shape) + "\n")
		json.dump(temp, f, separators=(',', ':'), sort_keys=True, indent=4)
		f.close()

	def __init__(self, isHardware=True, trainable=False):
                cfg.TEST.SCALES = (1200,)
                cfg.TEST.MAX_SIZE = 1056

		self.inputs = []
		self.trainable = trainable
		self.isHardware = isHardware
		self.data = tf.placeholder(tf.float32, shape=[None, None, None, 3])
		self.im_info = tf.placeholder(tf.float32, shape=[None,3])
		self.keep_prob = tf.placeholder(tf.float32)
		self.layers = dict({'data':self.data, 'im_info':self.im_info})		
		self.classes = ("BG","car","van","bus","truck","person","person-sitting","bicycle","motor","open-tricycle","close-tricycle","forklift","crane","large-block")
		self._layer_map = {}
		self.create_layer_map()
		self.setup()

	def _clip_boxes(self, boxes, im_shape):
		"""
		Clip boxes to image boundaries.
		"""
		boxes[:, 0::4] = np.maximum(boxes[:, 0::4], 0)
		boxes[:, 1::4] = np.maximum(boxes[:, 1::4], 0)
		boxes[:, 2::4] = np.minimum(boxes[:, 2::4], im_shape[1] - 1)
		boxes[:, 3::4] = np.minimum(boxes[:, 3::4], im_shape[0] - 1)
		return boxes

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
                
                im = im - cfg.PIXEL_MEANS
                im /= cfg.PIXEL_STDS
                
                print('\nAfter substract mean')
                #print(im[0:10,0:10,0])
                #print(im[0:10,0:10,1])
                #print(im[0:10,0:10,2])
                
                im_shape = im.shape
                im_size_min = np.min(im_shape[0:2])
                im_size_max = np.max(im_shape[0:2])
                print("im_size_min: %d, im_size_max: %d\n" %(im_size_min,im_size_max))
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
                
                #print('\nAfter resize')
                #print(im[0:10,0:10,0])
                #print(im[0:10,0:10,1])
                #print(im[0:10,0:10,2])
                
                # Create a blob to hold the input images
                
                blob = im_list_to_blob(processed_ims)
                
                blob = blob.astype(np.int32, copy=True)
                blob = blob.astype(np.float32, copy=True)

                self.build_json("blob_pre_clip",blob)

                blob = np.clip(blob,-128,127)

                self.build_json("blob_post_clip",blob)
                
                #print(blob[0,0:10,0:10,0])
                #print(blob[0,0:10,0:10,1])
                #print(blob[0,0:10,0:10,2])
                #np.savetxt('a0_processed_ims0.csv',blob[0,:,:,0],fmt='%d,')
                #np.savetxt('a0_processed_ims1.csv',blob[0,:,:,1],fmt='%d,')
                #np.savetxt('a0_processed_ims2.csv',blob[0,:,:,2],fmt='%d,')
                print('blob.shape = {}'.format(blob.shape))
                print('im.shape = {}'.format(im.shape))
                
                return blob, np.array(im_scale_factors)

	def pre_process(self, im, scale_mode=0):
		"""
		MUST HAVE FUNCTION IN ALL NETWORKS !!!! 
		Pre-processing of the image. De-mean, crop and resize
		Supports 3 scaling factors at the moment. 
		Returns the feed dictionary the network is expecting
		"""
		blobs = {'data' : None, 'rois' : None}
		if scale_mode == 0:
			im_dim = [640,1056]
		elif scale_mode == 1:
			im_dim = [364, 602]
		else:	
			im_dim = [640,1056]
		blobs['data'], im_scales = self._get_image_blob(im, im_dim)	
		blobs['im_info'] = np.array([[blobs['data'].shape[1], blobs['data'].shape[2], im_scales[0]]],dtype=np.float32)
		feed_dict={self.data: blobs['data'], self.im_info: blobs['im_info'], self.keep_prob: 1.0}
		return feed_dict

	def post_process(self, im, sim_ops, scale_factor=1):
		"""
		MUST HAVE FUNCTION IN ALL NETWORKS !!!! 
		Post-processing of the results from network. This function can be used to visualize data from hardware.  
		self.post_process(im, [cls_score, cls_prob, bbox_pred, rois], scale_factor)	
		"""
                print("cls_score:\n")
                print(sim_ops[0])
                print("cls_prob:\n")
                print(sim_ops[1])
                print("bbox_pred:\n")
                print(sim_ops[2])
                print("rois:\n")
                print(sim_ops[3])
                print("scale_factor:\n")
                print(scale_factor)

		im = im[:, :, (2, 1, 0)]

		cls_score = sim_ops[0]
		cls_score = convert_to_float_py(cls_score, self._layer_map[78]['fl'])

		cls_prob = sim_ops[1]

		bbox_pred = sim_ops[2]
		bbox_pred = convert_to_float_py(bbox_pred, self._layer_map[79]['fl'])

		rois = sim_ops[3]
		boxes = rois[:, 1:5] / scale_factor

		# ABINASH ONLY FOR DEBUG DELETE IT 
		scores = cls_prob
		#scores = cls_score
		box_deltas = bbox_pred
		pred_boxes = bbox_transform_inv(boxes, box_deltas, False)
		pred_boxes = self._clip_boxes(pred_boxes, im.shape)	

		fig, ax = plt.subplots(figsize=(12, 12))
		ax.imshow(im, aspect='equal')
		CONF_THRESH = 0.6
		NMS_THRESH = 0.4
                self.build_json("scores",scores)
                self.build_json("box_deltas",box_deltas)
                self.build_json("pred_boxes",pred_boxes)
                self.build_json("boxes_pred",bbox_pred)
                self.build_json("cls_prob",cls_prob)
		for cls_ind, cls in enumerate(self.classes[1:]):
			cls_ind += 1  # because we skipped background
			cls_boxes = pred_boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
			cls_scores = scores[:, cls_ind]
			dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
                        self.build_json(("dets_pre_keep_"+str(cls_ind)),dets)
			keep = nms(dets, NMS_THRESH)
			dets = dets[keep, :]

                        self.build_json(("cls_boxes_"+str(cls_ind)),cls_boxes)
                        self.build_json(("cls_scores_"+str(cls_ind)),cls_scores)
                        print("POST_PROCESS: classes, cls_ind: %d, keep: \n" %(cls_ind))
                        print(keep)
                        self.build_json(("dets_post_keep_"+str(cls_ind)),dets)
			self._vis_detections(im, cls, dets, ax, thresh=CONF_THRESH)
		plt.show()		
				
	def _vis_detections(self, im, class_name, dets, ax, thresh=0.5):
		"""
		Draw detected bounding boxes.
		"""
		inds = np.where(dets[:, -1] >= thresh)[0]
		if len(inds) == 0:
			print_msg('No detections found',2)
			return
		for i in inds:
			bbox = dets[i, :4]
			score = dets[i, -1]
			ax.add_patch(
				plt.Rectangle((bbox[0], bbox[1]),
							  bbox[2] - bbox[0],
							  bbox[3] - bbox[1], fill=False,
							  edgecolor='red', linewidth=3.5)
			)
			ax.text(bbox[0], bbox[1] - 2,'{:s} {:.3f}'.format(class_name, score),bbox=dict(facecolor='blue', alpha=0.5),fontsize=14, color='white')
		
		ax.set_title(('{} detections with p({} | box) >= {:.1f}').format(class_name, class_name,thresh),fontsize=14)
		plt.axis('off')
		plt.tight_layout()
		plt.draw()

	def run_sw_demo(self, sess, image_path, scale_mode=0): 
		"""
		Pure software demo. 
		"""
		print_msg('Running software only demo ... ', 3)
		# Load Image and pre-process
		im = cv2.imread(image_path)	
		feed_dict = self.pre_process(im, scale_mode)
		scale_factor = feed_dict[self.im_info][0][2]	# TODO send in scale for both X and Y directions 

		merged = tf.summary.merge_all()
		tfboard_writer = tf.summary.FileWriter(cfg.OUTPUT_DIR, sess.graph)
		# Run network with the session to get proper outputs
		summary, cls_score, cls_prob, bbox_pred, rois = sess.run([merged, self.get_output('cls_score_fabu'), 		
								 self.get_output('cls_prob'), self.get_output('bbox_pred_fabu'), 
							  	 self.get_output('proposal')], feed_dict=feed_dict)

		tfboard_writer.add_summary(summary)
		# Post process on network output and show result 
		self.post_process(im, [cls_score, cls_prob, bbox_pred, rois], scale_factor)	
		
	def get_final_outputs(self):
		return self.get_output('cls_score_fabu') 		
	
	def create_layer_map(self):
		""" 
		Helper Function for Validation. Dictionary of layer wise parameters 
		"""
		self._layer_map[0]=	{'name':'data',					'inputs':[],					'fl':7,'type':'data'} 
		self._layer_map[1]=	{'name':'conv1',				'inputs':[0],					'fl':6, 'fl_w':16,'type':'conv'}
		self._layer_map[2]=	{'name':'conv2',				'inputs':[1],					'fl':6, 'fl_w':14,'type':'conv'}
		self._layer_map[3]=	{'name':'conv3',				'inputs':[2],					'fl':6, 'fl_w':16,'type':'conv'}
		# Inception 3a ~~~~~~~~~~~~~~~~~~~~~~~~
		self._layer_map[4]=	{'name':'inc3a/pool1',			'inputs':[3],					 'fl':6,'type':'plmx'}
		self._layer_map[5]=	{'name':'inc3a/conv1',			'inputs':[4],					 'fl':6, 'fl_w':15,'type':'conv'}
		self._layer_map[6]=	{'name':'inc3a/conv3_1',		'inputs':[3],					 'fl':6, 'fl_w':14,'type':'conv'}
		self._layer_map[7]=	{'name':'inc3a/conv3_2',		'inputs':[6],					 'fl':6, 'fl_w':15,'type':'conv'}
		self._layer_map[8]=	{'name':'inc3a/conv5_1',		'inputs':[3],					 'fl':6, 'fl_w':14,'type':'conv'}
		self._layer_map[9]=	{'name':'inc3a/conv5_2',		'inputs':[8],					 'fl':6, 'fl_w':15,'type':'conv'}
		self._layer_map[10]={'name':'inc3a/conv5_3',		'inputs':[9],					 'fl':6, 'fl_w':15,'type':'conv'}
		# Inception 3b ~~~~~~~~~~~~~~~~~~~~~~~~
		self._layer_map[11]={'name':'inc3b/conv1',			'inputs':[5,7,10], 				 'fl':6, 'fl_w':14,'type':'conv'}
		self._layer_map[12]={'name':'inc3b/conv3_1',		'inputs':[5,7,10], 				 'fl':6, 'fl_w':14,'type':'conv'}
		self._layer_map[13]={'name':'inc3b/conv3_2',		'inputs':[12], 					 'fl':6, 'fl_w':15,'type':'conv'}
		self._layer_map[14]={'name':'inc3b/conv5_1',		'inputs':[5,7,10], 				 'fl':6, 'fl_w':15,'type':'conv'}
		self._layer_map[15]={'name':'inc3b/conv5_2',		'inputs':[14], 					 'fl':6, 'fl_w':14,'type':'conv'}
		self._layer_map[16]={'name':'inc3b/conv5_3',		'inputs':[15],					 'fl':6, 'fl_w':15,'type':'conv'}
		# Inception 3c ~~~~~~~~~~~~~~~~~~~~~~~~
		self._layer_map[17]={'name':'inc3c/conv1',			'inputs':[11,13,16],		 	 'fl':6, 'fl_w':14,'type':'conv'}
		self._layer_map[18]={'name':'inc3c/conv3_1',		'inputs':[11,13,16],		 	 'fl':6, 'fl_w':15,'type':'conv'}
		self._layer_map[19]={'name':'inc3c/conv3_2',		'inputs':[18], 				 	 'fl':6, 'fl_w':15,'type':'conv'}
		self._layer_map[20]={'name':'inc3c/conv5_1',		'inputs':[11,13,16],		 	 'fl':6, 'fl_w':14,'type':'conv'}
		self._layer_map[21]={'name':'inc3c/conv5_2',		'inputs':[20], 				 	 'fl':6, 'fl_w':13,'type':'conv'}
		self._layer_map[22]={'name':'inc3c/conv5_3',		'inputs':[21],				 	 'fl':6, 'fl_w':15,'type':'conv'}
		# Inception 3d ~~~~~~~~~~~~~~~~~~~~~~~~
		self._layer_map[23]={'name':'inc3d/conv1',			'inputs':[17,19,22],		 	 'fl':6, 'fl_w':15,'type':'conv'}
		self._layer_map[24]={'name':'inc3d/conv3_1',		'inputs':[17,19,22],		 	 'fl':6, 'fl_w':14,'type':'conv'}
		self._layer_map[25]={'name':'inc3d/conv3_2',		'inputs':[24],				 	 'fl':6, 'fl_w':15,'type':'conv'}
		self._layer_map[26]={'name':'inc3d/conv5_1',		'inputs':[17,19,22],		 	 'fl':6, 'fl_w':15,'type':'conv'}
		self._layer_map[27]={'name':'inc3d/conv5_2',		'inputs':[26],				 	 'fl':6, 'fl_w':14,'type':'conv'}
		self._layer_map[28]={'name':'inc3d/conv5_3',		'inputs':[27],				 	 'fl':6, 'fl_w':16,'type':'conv'}
		# Inception 3e ~~~~~~~~~~~~~~~~~~~~~~~~
		self._layer_map[29]={'name':'inc3e/conv1',			'inputs':[23,25,28],		 	 'fl':6, 'fl_w':14,'type':'conv'}
		self._layer_map[30]={'name':'inc3e/conv3_1',		'inputs':[23,25,28],		 	 'fl':6, 'fl_w':15,'type':'conv'}
		self._layer_map[31]={'name':'inc3e/conv3_2',		'inputs':[30],				 	 'fl':6, 'fl_w':15,'type':'conv'}
		self._layer_map[32]={'name':'inc3e/conv5_1',		'inputs':[23,25,28],		 	 'fl':6, 'fl_w':15,'type':'conv'}
		self._layer_map[33]={'name':'inc3e/conv5_2',		'inputs':[32],				 	 'fl':6, 'fl_w':14,'type':'conv'}
		self._layer_map[34]={'name':'inc3e/conv5_3',		'inputs':[33],				 	 'fl':6, 'fl_w':15,'type':'conv'}
		# Inception 4a ~~~~~~~~~~~~~~~~~~~~~~~~
		self._layer_map[35]={'name':'inc4a/pool1',			'inputs':[29,31,34],		 	 'fl':6,'type':'plmx'}
		self._layer_map[36]={'name':'inc4a/conv1',			'inputs':[35], 				 	 'fl':6, 'fl_w':15,'type':'conv'}
		self._layer_map[37]={'name':'inc4a/conv3_1',		'inputs':[29,31,34],		 	 'fl':6, 'fl_w':15,'type':'conv'}
		self._layer_map[38]={'name':'inc4a/conv3_2',		'inputs':[37],				 	 'fl':6, 'fl_w':15,'type':'conv'}
		self._layer_map[39]={'name':'inc4a/conv5_1',		'inputs':[29,31,34],		 	 'fl':6, 'fl_w':16,'type':'conv'}
		self._layer_map[40]={'name':'inc4a/conv5_2',		'inputs':[39],				 	 'fl':6, 'fl_w':15,'type':'conv'}
		self._layer_map[41]={'name':'inc4a/conv5_3',		'inputs':[40],				 	 'fl':6, 'fl_w':16,'type':'conv'}
		# Inception 4b ~~~~~~~~~~~~~~~~~~~~~~~~
		self._layer_map[42]={'name':'inc4b/conv1',			'inputs':[36,38,41],		 	 'fl':6, 'fl_w':15,'type':'conv'}
		self._layer_map[43]={'name':'inc4b/conv3_1',		'inputs':[36,38,41],		 	 'fl':6, 'fl_w':15,'type':'conv'}
		self._layer_map[44]={'name':'inc4b/conv3_2',		'inputs':[43],				 	 'fl':6, 'fl_w':15,'type':'conv'}
		self._layer_map[45]={'name':'inc4b/conv5_1',		'inputs':[36,38,41],		 	 'fl':6, 'fl_w':16,'type':'conv'}
		self._layer_map[46]={'name':'inc4b/conv5_2',		'inputs':[45],				 	 'fl':6, 'fl_w':15,'type':'conv'}
		self._layer_map[47]={'name':'inc4b/conv5_3',		'inputs':[46],				 	 'fl':6, 'fl_w':15,'type':'conv'}
		# Inception 4c ~~~~~~~~~~~~~~~~~~~~~~~~
		self._layer_map[48]={'name':'inc4c/conv1',			'inputs':[42,44,47],		 	 'fl':6, 'fl_w':15,'type':'conv'}
		self._layer_map[49]={'name':'inc4c/conv3_1',		'inputs':[42,44,47],		 	 'fl':6, 'fl_w':15,'type':'conv'}
		self._layer_map[50]={'name':'inc4c/conv3_2',		'inputs':[49],				 	 'fl':6, 'fl_w':15,'type':'conv'}
		self._layer_map[51]={'name':'inc4c/conv5_1',		'inputs':[42,44,47],		 	 'fl':6, 'fl_w':15,'type':'conv'}
		self._layer_map[52]={'name':'inc4c/conv5_2',		'inputs':[51],				 	 'fl':6, 'fl_w':15,'type':'conv'}
		self._layer_map[53]={'name':'inc4c/conv5_3',		'inputs':[52],				 	 'fl':6, 'fl_w':14,'type':'conv'}
		# Inception 4d ~~~~~~~~~~~~~~~~~~~~~~~~
		self._layer_map[54]={'name':'inc4d/conv1',			'inputs':[48,50,53],		 	 'fl':6, 'fl_w':15,'type':'conv'}
		self._layer_map[55]={'name':'inc4d/conv3_1',		'inputs':[48,50,53],		 	 'fl':6, 'fl_w':14,'type':'conv'}
		self._layer_map[56]={'name':'inc4d/conv3_2',		'inputs':[55], 				 	 'fl':6, 'fl_w':15,'type':'conv'}
		self._layer_map[57]={'name':'inc4d/conv5_1',		'inputs':[48,50,53],		 	 'fl':6, 'fl_w':16,'type':'conv'}
		self._layer_map[58]={'name':'inc4d/conv5_2',		'inputs':[57],				 	 'fl':6, 'fl_w':15,'type':'conv'}
		self._layer_map[59]={'name':'inc4d/conv5_3',		'inputs':[58],				 	 'fl':6, 'fl_w':15,'type':'conv'}
		# Inception 4e ~~~~~~~~~~~~~~~~~~~~~~~~
		self._layer_map[60]={'name':'inc4e/conv1',			'inputs':[54,56,59],		 	 'fl':6, 'fl_w':15,'type':'conv'}
		self._layer_map[61]={'name':'inc4e/conv3_1',		'inputs':[54,56,59],		 	 'fl':6, 'fl_w':15,'type':'conv'}
		self._layer_map[62]={'name':'inc4e/conv3_2',		'inputs':[61],				 	 'fl':6, 'fl_w':15,'type':'conv'}
		self._layer_map[63]={'name':'inc4e/conv5_1',		'inputs':[54,56,59],		 	 'fl':6, 'fl_w':14,'type':'conv'}
		self._layer_map[64]={'name':'inc4e/conv5_2',		'inputs':[63],				 	 'fl':6, 'fl_w':14,'type':'conv'}
		self._layer_map[65]={'name':'inc4e/conv5_3',		'inputs':[64],				 	 'fl':6, 'fl_w':15,'type':'conv'}
		# Downsample ~~~~~~~~~~~~~~~~~~~~~~~~~~~
		self._layer_map[66]={'name':'downsample', 			'inputs':[3], 					 'fl':6,'type':'plmx'}
		self._layer_map[67]={'name':'convf', 				'inputs':[66,29,31,34,60,62,65], 'fl':8, 'fl_w':16,'type':'conv'}					
		self._layer_map[68]={'name':'convf_up2',			'inputs':[67],					 'fl':8, 'fl_w':16,'type':'decv'}	# TODO: fl and fl_w values
		self._layer_map[69]={'name':'rpn_conv1', 			'inputs':[67], 					 'fl':10, 'fl_w':17,'type':'conv'}

		self._layer_map[70]={'name':'rpn_cls_score_fabu',	'inputs':[69], 					 'fl':10, 'fl_w':17,'type':'conv'}
		self._layer_map[71]={'name':'rpn_bbox_pred_fabu', 	'inputs':[69], 					 'fl':13, 'fl_w':18,'type':'conv'}
		self._layer_map[72]={'name':'proposal', 			'inputs':[70,71], 				 'fl':4,'type':'prop'}							
		self._layer_map[73]={'name':'roi_pool_conv5', 		'inputs':[68,72], 				 'fl':8,'type':'roip'}							
		self._layer_map[74]={'name':'fc6_L', 				'inputs':[73],					 'fl':6, 'fl_w':17,'type':'fcon'}
		self._layer_map[75]={'name':'fc6_U', 				'inputs':[74],					 'fl':7, 'fl_w':17,'type':'fcon'}
		self._layer_map[76]={'name':'fc7_L', 				'inputs':[75],					 'fl':7, 'fl_w':17,'type':'fcon'}
		self._layer_map[77]={'name':'fc7_U', 				'inputs':[76],					 'fl':9, 'fl_w':17,'type':'fcon'}
		self._layer_map[78]={'name':'cls_score_fabu', 		'inputs':[77],					 'fl':9, 'fl_w':17,'type':'fcon'}
		self._layer_map[79]={'name':'bbox_pred_fabu', 		'inputs':[77],					 'fl':9, 'fl_w':17,'type':'fcon'}

	def setup(self):
		anchor_scales = [1.5, 3.0, 6.0, 9.0, 16.0, 32.0, 48.0] 
		base_size = 10
		ratios = [0.333, 0.5, 0.667, 1.0, 1.5, 2.0, 3.0]
		pre_nms_top_n = 2000  
		_feat_stride = [16, ]
		num_stddev = cfg.NUM_STDDEV_OB	
                max_nms_topN = cfg.MAX_NMS_TOP_N_OB

		(self.feed('data')
			 .pad(1,name='conv1_pad')
			 .conv(4,4,32,2,2,name='conv1', fl=6, padding='VALID')
			 .pad(1,name='conv2_pad')
			 .conv(3,3,48,2,2,name='conv2', fl=6, padding='VALID')
			 .pad(1,name='conv3_pad')
			 .conv(3,3,96,2,2,name='conv3', fl=6, padding='VALID'))

		# Inception 3a ~~
		(self.feed('conv3')
			 .pad_v1(1,1,name='inc31fpool1/pad')
			 .max_pool(3,3,2,2,name='inc3a/pool1',padding='VALID')
			 .conv(1,1,96,1,1,name='inc3a/conv1', fl=6, padding='VALID'))
		(self.feed('conv3')
			 .conv(1,1,16,1,1,name='inc3a/conv3_1', fl=6, padding='VALID')
			 .pad(1,name='inc3a/conv3_2_pad')
			 .conv(3,3,64,2,2,name='inc3a/conv3_2', fl=6, padding='VALID'))
		(self.feed('conv3')
			 .conv(1,1,16,1,1,name='inc3a/conv5_1', fl=6, padding='VALID')
			 .pad(1,name='inc3a/conv5_2_pad')
			 .conv(3,3,32,1,1,name='inc3a/conv5_2', fl=6, padding='VALID')
			 .pad(1,name='inc3a/conv5_3_pad')
			 .conv(3,3,32,2,2,name='inc3a/conv5_3', fl=6, padding='VALID'))
		(self.feed('inc3a/conv1', 'inc3a/conv3_2','inc3a/conv5_3')
			 .concat(axis=-1,name='inc3a'))

		# Inception 3b ~~ !
		(self.feed('inc3a')
			 .conv(1,1,96,1,1,name='inc3b/conv1', fl=6, padding='VALID'))
		(self.feed('inc3a')
			 .conv(1,1,16,1,1,name='inc3b/conv3_1', fl=6, padding='VALID')
			 .pad(1,name='inc3b/conv3_2_pad')
			 .conv(3,3,64,1,1,name='inc3b/conv3_2', fl=6, padding='VALID'))
		(self.feed('inc3a')
			 .conv(1,1,16,1,1,name='inc3b/conv5_1', fl=6, padding='VALID')
			 .pad(1,name='inc3b/conv5_2_pad')
			 .conv(3,3,32,1,1,name='inc3b/conv5_2', fl=6, padding='VALID')
			 .pad(1,name='inc3b/conv5_3_pad')
			 .conv(3,3,32,1,1,name='inc3b/conv5_3', fl=6, padding='VALID'))
		(self.feed('inc3b/conv1', 'inc3b/conv3_2','inc3b/conv5_3')
			 .concat(axis=-1,name='inc3b'))

		# Inception 3c ~~ !
		(self.feed('inc3b')
			 .conv(1,1,96,1,1,name='inc3c/conv1', fl=6, padding='VALID'))
		(self.feed('inc3b')
			 .conv(1,1,16,1,1,name='inc3c/conv3_1', fl=6, padding='VALID')
			 .pad(1,name='inc3c/conv3_2_pad')
			 .conv(3,3,64,1,1,name='inc3c/conv3_2', fl=6, padding='VALID'))
		(self.feed('inc3b')
			 .conv(1,1,16,1,1,name='inc3c/conv5_1', fl=6, padding='VALID')
			 .pad(1,name='inc3cfconv5_2_pad')
			 .conv(3,3,32,1,1,name='inc3c/conv5_2', fl=6, padding='VALID')
			 .pad(1,name='inc3c/conv5_3_pad')
			 .conv(3,3,32,1,1,name='inc3c/conv5_3', fl=6, padding='VALID'))
		(self.feed('inc3c/conv1', 'inc3c/conv3_2','inc3c/conv5_3')
			 .concat(axis=-1,name='inc3c'))

		# Inception 3d ~~ !
		(self.feed('inc3c')
			 .conv(1,1,96,1,1,name='inc3d/conv1', fl=6, padding='VALID'))
		(self.feed('inc3c')
			 .conv(1,1,16,1,1,name='inc3d/conv3_1', fl=6, padding='VALID')
			 .pad(1,name='inc3d/conv3_3_pad')
			 .conv(3,3,64,1,1,name='inc3d/conv3_2', fl=6, padding='VALID'))
		(self.feed('inc3c')
			 .conv(1,1,16,1,1,name='inc3d/conv5_1', fl=6, padding='VALID')
			 .pad(1,name='inc3d/conv5_3_pad')
			 .conv(3,3,32,1,1,name='inc3d/conv5_2', fl=6, padding='VALID')
			 .pad(1,name='inc3d/conv5_3_pad')
			 .conv(3,3,32,1,1,name='inc3d/conv5_3', fl=6, padding='VALID'))
		(self.feed('inc3d/conv1', 'inc3d/conv3_2','inc3d/conv5_3')
			 .concat(axis=-1,name='inc3d'))

		# Inception 3e ~~ !
		(self.feed('inc3d')
			 .conv(1,1,96,1,1,name='inc3e/conv1', fl=6, padding='VALID'))
		(self.feed('inc3d')
			 .conv(1,1,16,1,1,name='inc3e/conv3_1', fl=6, padding='VALID')
			 .pad(1,name='inc3e/conv3_2_pad')
			 .conv(3,3,64,1,1,name='inc3e/conv3_2', fl=6, padding='VALID'))
		(self.feed('inc3d')
			 .conv(1,1,16,1,1,name='inc3e/conv5_1', fl=6, padding='VALID')
			 .pad(1,name='inc3e/conv5_2_pad')
			 .conv(3,3,32,1,1,name='inc3e/conv5_2', fl=6, padding='VALID')
			 .pad(1,name='inc3e/conv5_3_pad')
			 .conv(3,3,32,1,1,name='inc3e/conv5_3', fl=6, padding='VALID'))
		(self.feed('inc3e/conv1', 'inc3e/conv3_2','inc3e/conv5_3')
			 .concat(axis=-1,name='inc3e'))

		# Inception 4a ~~
		(self.feed('inc3e')
			 .pad(1,name='inc4a/pool1_pad')
			 .max_pool(3,3,1,1,name='inc4a/pool1',padding='VALID')
			 .conv(1,1,128,1,1,name='inc4a/conv1', fl=6, padding='VALID'))
		(self.feed('inc3e')
			 .conv(1,1,32,1,1,name='inc4a/conv3_1', fl=6, padding='VALID')
			 .pad(2,name='inc4a/conv3_2_pad')
			 .conv(5,5,96,1,1,name='inc4a/conv3_2', fl=6, padding='VALID'))
		(self.feed('inc3e')
			 .conv(1,1,16,1,1,name='inc4a/conv5_1', fl=6, padding='VALID')
			 .pad(2,name='inc4a/conv5_2_pad')
			 .conv(5,5,32,1,1,name='inc4a/conv5_2', fl=6, padding='VALID')
			 .pad(2,name='inc4a/conv5_3_pad')
			 .conv(5,5,32,1,1,name='inc4a/conv5_3', fl=6, padding='VALID'))
		(self.feed('inc4a/conv1', 'inc4a/conv3_2','inc4a/conv5_3')
			 .concat(axis=-1,name='inc4a'))

		# Inception 4b ~~
		(self.feed('inc4a')
			 .conv(1,1,128,1,1,name='inc4b/conv1', fl=6, padding='VALID'))
		(self.feed('inc4a')
			 .conv(1,1,32,1,1,name='inc4b/conv3_1', fl=6, padding='VALID')
			 .pad(2,name='inc4b/conv3_2_pad')
			 .conv(5,5,96,1,1,name='inc4b/conv3_2', fl=6, padding='VALID'))
		(self.feed('inc4a')
			 .conv(1,1,16,1,1,name='inc4b/conv5_1', fl=6, padding='VALID')
			 .pad(2,name='inc4b/conv5_2_pad')
			 .conv(5,5,32,1,1,name='inc4b/conv5_2', fl=6, padding='VALID')
			 .pad(2,name='inc4b/conv5_3_pad')
			 .conv(5,5,32,1,1,name='inc4b/conv5_3', fl=6, padding='VALID'))
		(self.feed('inc4b/conv1', 'inc4b/conv3_2','inc4b/conv5_3')
			 .concat(axis=-1,name='inc4b'))

		# Inception 4c ~~
		(self.feed('inc4b')
			 .conv(1,1,128,1,1,name='inc4c/conv1', fl=6, padding='VALID'))
		(self.feed('inc4b')
			 .conv(1,1,32,1,1,name='inc4c/conv3_1', fl=6, padding='VALID')
			 .pad(2,name='inc4c/conv3_2_pad')
			 .conv(5,5,96,1,1,name='inc4c/conv3_2', fl=6, padding='VALID'))
		(self.feed('inc4b')
			 .conv(1,1,16,1,1,name='inc4c/conv5_1', fl=6, padding='VALID')
			 .pad(2,name='inc4c/conv5_2_pad')
			 .conv(5,5,32,1,1,name='inc4c/conv5_2', fl=6, padding='VALID')
			 .pad(2,name='inc4c/conv5_3_pad')
			 .conv(5,5,32,1,1,name='inc4c/conv5_3', fl=6, padding='VALID'))
		(self.feed('inc4c/conv1', 'inc4c/conv3_2','inc4c/conv5_3')
			 .concat(axis=-1,name='inc4c'))

		# Inception 4d ~~
		(self.feed('inc4c')
			 .conv(1,1,128,1,1,name='inc4d/conv1', fl=6, padding='VALID'))
		(self.feed('inc4c')
			 .conv(1,1,32,1,1,name='inc4d/conv3_1', fl=6, padding='VALID')
			 .pad(2,name='inc4d/conv3_2_pad')
			 .conv(5,5,96,1,1,name='inc4d/conv3_2', fl=6, padding='VALID'))
		(self.feed('inc4c')
			 .conv(1,1,16,1,1,name='inc4d/conv5_1', fl=6, padding='VALID')
			 .pad(2,name='inc4d/conv5_2_pad')
			 .conv(5,5,32,1,1,name='inc4d/conv5_2', fl=6, padding='VALID')
			 .pad(2,name='inc4d/conv5_3_pad')
			 .conv(5,5,32,1,1,name='inc4d/conv5_3', fl=6, padding='VALID'))
		(self.feed('inc4d/conv1', 'inc4d/conv3_2','inc4d/conv5_3')
			 .concat(axis=-1,name='inc4d'))

		# Inception 4e ~~
		(self.feed('inc4d')
			 .conv(1,1,128,1,1,name='inc4e/conv1', fl=6, padding='VALID'))
		(self.feed('inc4d')
			 .conv(1,1,32,1,1,name='inc4e/conv3_1', fl=6, padding='VALID')
			 .pad(2,name='inc4e/conv3_2_pad')
			 .conv(5,5,96,1,1,name='inc4e/conv3_2', fl=6, padding='VALID'))
		(self.feed('inc4d')
			 .conv(1,1,16,1,1,name='inc4e/conv5_1', fl=6, padding='VALID')
			 .pad(2,name='inc4e/conv5_2_pad')
			 .conv(5,5,32,1,1,name='inc4e/conv5_2', fl=6, padding='VALID')
			 .pad(2,name='inc4e/conv5_3_pad')
			 .conv(5,5,32,1,1,name='inc4e/conv5_3', fl=6, padding='VALID'))
		(self.feed('inc4e/conv1', 'inc4e/conv3_2','inc4e/conv5_3')
			 .concat(axis=-1,name='inc4e'))

		(self.feed('conv3')
			 .pad_v1(1,1,name='inc31/pool1/pad')
			 .max_pool(3,3,2,2,name='downsample', padding='VALID'))

		(self.feed('downsample','inc3e','inc4e',)
			 .concat(axis=-1,name='concat')
			 .conv(1,1,256,1,1,name='convf', fl=8 ,padding='VALID')
                         #.nearest_neighbor_layer(name='convf_up2',stride=2))	
			 .upconv(shape=[1,80,132,256], c_o=256, name='convf_up2', ksize=4, stride=2, biased=False, relu=False, padding='SAME'))	# Deconv layer !!! 	

		(self.feed('convf')
			 .conv(1,1,256,1,1,name='rpn_conv1', fl=10, padding='VALID')
			 .conv(1,1,98,1,1,name='rpn_cls_score_fabu', relu=False, fl=10, padding='VALID')
			 .dummy(name='rpn_cls_prob_reshape'))

		(self.feed('rpn_conv1')
			 .conv(1,1,196,1,1,name='rpn_bbox_pred_fabu',relu=False, fl=13, padding='VALID'))

		(self.feed('rpn_cls_prob_reshape', 'rpn_bbox_pred_fabu', 'im_info')
			 .proposal_layer(_feat_stride, anchor_scales, 'TEST', base_size, ratios, pre_nms_top_n, max_nms_topN, name='proposal', num_stddev=num_stddev, fl_cls_prob=3, fl_bbox_pred=6))

		(self.feed('convf_up2','proposal')
			 .roi_pool(6,6,0.125, name='roi_pool_conv5')
			 .fc(512,name='fc6_L', fl=6, relu=False)
			 .fc(4096,name='fc6_U', fl=7)
			 .fc(128,name='fc7_L',fl=7, relu=False)
		  	 .fc(4096,name='fc7_U', fl=9)
			 .fc(56,name='bbox_pred_fabu',fl=9 ,relu=False))

		(self.feed('fc7_U')
			 .fc(14,name='cls_score_fabu',fl=9, relu=False)
			 .softmax(name='cls_prob'))
