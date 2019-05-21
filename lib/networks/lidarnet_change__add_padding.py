# -*- coding: utf-8 -*-
# @Author: liusongwei
# @Date:   2019-05-09 20:21:09
# @Last Modified by:   liusongwei
# @Last Modified time: 2019-05-09 21:14:14



# lidar net 
from .network import Network
import tensorflow as tf
import numpy as np
import os
import sys

class LIDAR(Network):

    def __init__(self, isHardware=True, trainable=False):
        '''Simply execute the inference operation '''

        self.inputs = []
        self.trainable = trainable
        self.isHardware = isHardware
        # self.data = tf.placeholder(tf.float32, shape=[None, None, None, 3])
        self.data = tf.placeholder(tf.float32, shape=[None, 1088, 800, 8])
        # self.im_info = tf.placeholder(tf.float32, shape=[None, 3])
        # self.keep_prob = tf.placeholder(tf.float32)
        # self.layers = dict({'data': self.data, 'im_info': self.im_info})
        self.layers=dict({'data':self.data})
        self._layer_map = {}
        self.create_layer_map()
        self.setup()


    def pre_process(self, image_path, scale_mode=0):

        """
        MUST HAVE FUNCTION IN ALL NETWORKS !!!!
        Pre-processing of the image. De-mean, crop and resize
        Returns the feed dictionary the network is expecting
        For VGG there is just one size image 224x224 at the moment.
        """

        print('Loading the images')
        if not os.path.exists(image_path):
            raise KeyError('the image path not exists ')
        else:
            full_path = []
            files = os.listdir(image_path)
            for file in files:
                full_path.append(os.path.join(image_path, file))
            image = np.load(full_path[0])
            # TODO:image preprocessing operation as you need
            image=np.transpose(image,(0,2,3,1))
        if self.isHardware==False:
            return image
        else:
            return self._quantize_based_frac(image,self._layer_map[0]['fl'])

    def post_process(self):
        pass


    def _quantize_based_frac(self,num,frac):
        sh = num.shape
        num = num.flatten()
        for i in range(len(num)):
            sign=1
            if num[i]<0:
                sign=-1
            num[i]=sign*round(abs(num[i])*pow(2,frac))
        num = num.reshape(sh)
        return num

    def quantize_data(self,data_path,origin_data_path,quantize_data_path):
        '''
        input:
        data_path: the XX.npy file include all the weights and biases 
        which are not be quantized
        return:
        quantized_.npy which include all the quantized weights and biases
        note: THE net.create_layer_map() MUST HAVE BEEN DONE 
        '''
        # change upsampling layer data format in XX.npy from list to dict
        data_dict = np.load(data_path).item()
        namelist = data_dict.keys()
        for name in namelist:
                # if type(data_dict[name]).__name__ == 'dict':
                #         print(' dict is : {}  shape is : {}'.format(name, data_dict[name]['weights'].shape))
                if 'upsampling' in name:
                        if type(data_dict[name]).__name__ != 'list':
                                raise ValueError('type is not list ')
                        new_dict = {}
                        new_dict['weights'] = np.transpose(data_dict[name][0], (2, 3, 1, 0))
                        new_dict['biases'] = data_dict[name][1]
                        # print(new_dict['weights'].shape)
                        # print(new_dict['biases'].shape)
                        data_dict[name] = new_dict
        # the file chenged( upsampling layer's data format form list to dict )
        with open(origin_data_path,'wb') as of:
            np.save(of,data_dict)

        # quantize based _layer_map() information
        all_layer_map_id=self._layer_map.keys()
        for name in namelist:
            flag=False
            for _id in all_layer_map_id:
                if name == self._layer_map[_id]['name']:
                    if self._layer_map[_id]['type'] not in ['conv','deconv']:
                        raise KeyError("layer: "+name+" is a wrong type in layer map")
                    weight_frac_bit=self._layer_map[_id]['fl_w']
                    # we set the bias's frac_bit equal to output's frac_bit
                    bias_frac_bit=self._layer_map[_id]['fl_y']
                    data_dict[name]['weights']=self._quantize_based_frac(data_dict[name]['weights'],weight_frac_bit)
                    data_dict[name]['biases']=self._quantize_based_frac(data_dict[name]['biases'],bias_frac_bit)
                    flag=True
                    print("layer:" +name+" be quantized")
            if not flag:
                raise KeyError('layer: '+name+' not in _layer_map')
        print('quantizing  over')
        with open(quantize_data_path,'wb') as qf:
            np.save(qf,data_dict)


    def run_sw_demo(self,sess,image_path):

        """
        Pure software demo.
        """
        # load one test image
        image=self.pre_process(image_path)
        output_nodes,fly_lists=self.get_output_node()
        feed_dict={self.data:image}
        outputs=sess.run(output_nodes,feed_dict=feed_dict)
        return outputs,fly_lists
     
    def run_test_node(self,sess,image_path):
        # one test node as you set by name
        name='resnetv1pyr1_hybridsequential2_conv0_fwd'
        test_node=tf.get_default_graph().get_tensor_by_name(
            name+'/BiasAdd:0')
        # get the right fl_y of name
        fl_y=None
        all_layer_map_id=self._layer_map.keys()
        for _id in all_layer_map_id:
            if name == self._layer_map[_id]['name']:
                fl_y=self._layer_map[_id]['fl_y']
        if fl_y is None:
            raise KeyError('rs is wrong ')
        # get the result
        image=self.pre_process(image_path)
        feed_dict={self.data:image}
        output=sess.run(test_node,feed_dict=feed_dict)
        return output,fl_y

    def save_pb_model(self,sess,output_node,output_path):
        const_graph = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def,
                                                                   output_node_names=[output_node])

        # 训练完成之后保存模型为.pb文件
        with tf.gfile.FastGFile(output_path, mode='wb') as f:
            f.write(const_graph.SerializeToString())

    def show_all_node(self):
        # review the graph node
        tensor_node_list = [tensor for tensor in tf.get_default_graph().as_graph_def().node]
        for tensor in tensor_node_list:
            print('{}'.format(tensor.name), '\n')

    def get_output_node(self):
        '''make the output node list '''
        cls_head_8 = tf.get_default_graph().get_tensor_by_name(
            'resnetv1pyr1_cls_head_8hybridsequential2_conv0_fwd/BiasAdd:0')
        reg_head_8 = tf.get_default_graph().get_tensor_by_name(
            'resnetv1pyr1_reg_head_8hybridsequential2_conv0_fwd/BiasAdd:0')
        cls_head_4 = tf.get_default_graph().get_tensor_by_name(
            'resnetv1pyr1_cls_head_4hybridsequential2_conv0_fwd/BiasAdd:0')
        reg_head_4 = tf.get_default_graph().get_tensor_by_name(
            'resnetv1pyr1_reg_head_4hybridsequential2_conv0_fwd/BiasAdd:0')
        cls_head_16 = tf.get_default_graph().get_tensor_by_name(
            'resnetv1pyr1_cls_head_16hybridsequential2_conv0_fwd/BiasAdd:0')
        reg_head_16 = tf.get_default_graph().get_tensor_by_name(
            'resnetv1pyr1_reg_head_16hybridsequential2_conv0_fwd/BiasAdd:0')
        output_list=[reg_head_4,reg_head_8,reg_head_16,cls_head_4,cls_head_8,cls_head_16]
        fl_y_list=[6,6,5,2,2,2]
        return output_list,fl_y_list


    def create_layer_map(self):
        # TODO : the inputs information is wrong, please update it if you use

        #..............
        self._layer_map[0]= {'name':'data',   'inputs':[],        'fl': 7            ,'type':'data'}
        self._layer_map[1]= {'name':'resnetv1pyr1_hybridsequential0_conv0_fwd',   'inputs':[0],        'fl_w': 2, 'fl':7, 'fl_y':2, 'rs':7, 'type':'conv'}         
        self._layer_map[2]= {'name':'resnetv1pyr1_hybridsequential1_conv0_fwd',   'inputs':[1],        'fl_w': 8, 'fl':2, 'fl_y':3, 'rs':7, 'type':'conv'} 
        self._layer_map[3]= {'name':'resnetv1pyr1_hybridsequential2_conv0_fwd',   'inputs':[2],        'fl_w': 8, 'fl':3, 'fl_y':3, 'rs':8, 'type':'conv'} 
        self._layer_map[4]= {'name':'pool0_fwd',   'inputs':[3],       'fl':3, 'fl_y':3,                'type':'max_pool'} 
        self._layer_map[5]= {'name':'resnetv1pyr1_stage_4bottleneckv10_hybridsequential0_conv0_fwd',   'inputs':[4],        'fl_w': 8, 'fl':3, 'fl_y':3, 'rs':8, 'type':'conv'} 
        self._layer_map[6]= {'name':'resnetv1pyr1_stage_4bottleneckv10_hybridsequential1_conv0_fwd',   'inputs':[5],        'fl_w': 9, 'fl':3, 'fl_y':3, 'rs':9, 'type':'conv'} 
        self._layer_map[7]= {'name':'resnetv1pyr1_stage_4bottleneckv10_hybridsequential2_conv0_fwd',   'inputs':[6],        'fl_w': 7, 'fl':3, 'fl_y':3, 'rs':7, 'type':'conv'} 
        
        #............
        self._layer_map[8]= {'name':'resnetv1pyr1_stage_4_shortcuthybridsequential0_conv0_fwd',   'inputs':[4],        'fl_w': 8, 'fl':3, 'fl_y':3, 'rs':8, 'type':'conv'} 

        #............
        self._layer_map[9]= {'name':'resnetv1pyr1_stage_4bottleneckv10__plus0', 'inputs':[7,8],              'fl':3,    'fl_y':3,           'type':'add'} 
        self._layer_map[10]= {'name':'resnetv1pyr1_stage_4bottleneckv10_relu2_fwd', 'inputs':[9],  'type':'relu'} 
        self._layer_map[11]= {'name':'resnetv1pyr1_stage_4bottleneckv11_hybridsequential0_conv0_fwd',   'inputs':[10],        'fl_w': 9, 'fl':3, 'fl_y':3, 'rs':9, 'type':'conv'} 
        self._layer_map[12]= {'name':'resnetv1pyr1_stage_4bottleneckv11_hybridsequential1_conv0_fwd',   'inputs':[11],        'fl_w': 9, 'fl':3, 'fl_y':3, 'rs':9, 'type':'conv'} 
        self._layer_map[13]= {'name':'resnetv1pyr1_stage_4bottleneckv11_hybridsequential2_conv0_fwd',   'inputs':[12],        'fl_w': 7, 'fl':3, 'fl_y':3, 'rs':7, 'type':'conv'} 

        #.............
        self._layer_map[14]= {'name':'resnetv1pyr1_stage_4bottleneckv11__plus0', 'inputs':[10,13],              'fl':3,    'fl_y':3,          'type':'add'} 
        self._layer_map[15]= {'name':'resnetv1pyr1_stage_4bottleneckv11_relu2_fwd', 'inputs':[14],  'type':'relu'} 
        self._layer_map[16]= {'name':'resnetv1pyr1_stage_4bottleneckv12_hybridsequential0_conv0_fwd',   'inputs':[15],        'fl_w': 9, 'fl':3, 'fl_y':3, 'rs':9, 'type':'conv'} 
        self._layer_map[17]= {'name':'resnetv1pyr1_stage_4bottleneckv12_hybridsequential1_conv0_fwd',   'inputs':[16],        'fl_w': 9, 'fl':3, 'fl_y':3, 'rs':9, 'type':'conv'} 
        self._layer_map[18]= {'name':'resnetv1pyr1_stage_4bottleneckv12_hybridsequential2_conv0_fwd',   'inputs':[17],        'fl_w': 7, 'fl':3, 'fl_y':3, 'rs':7, 'type':'conv'} 

        #.............
        self._layer_map[19]= {'name':'resnetv1pyr1_stage_4bottleneckv12__plus0', 'inputs':[15,18],              'fl':3,    'fl_y':3,         'type':'add'} 
        self._layer_map[20]= {'name':'resnetv1pyr1_stage_4bottleneckv12_relu2_fwd', 'inputs':[19],  'type':'relu'} 
        self._layer_map[21]= {'name':'resnetv1pyr1_stage_8bottleneckv10_hybridsequential0_conv0_fwd',   'inputs':[20],        'fl_w': 9, 'fl':3, 'fl_y':3, 'rs':9, 'type':'conv'} 
        self._layer_map[22]= {'name':'resnetv1pyr1_stage_8bottleneckv10_hybridsequential1_conv0_fwd',   'inputs':[21],        'fl_w': 10, 'fl':3, 'fl_y':3, 'rs':10, 'type':'conv'} 
        self._layer_map[23]= {'name':'resnetv1pyr1_stage_8bottleneckv10_hybridsequential2_conv0_fwd',   'inputs':[22],        'fl_w': 8, 'fl':3, 'fl_y':3, 'rs':8, 'type':'conv'} 

        #.............
        self._layer_map[24]= {'name':'resnetv1pyr1_stage_8_shortcutpool0_fwd',   'inputs':[20], 'fl':3, 'fl_y':3, 'type':'avg_pool'} 
        self._layer_map[25]= {'name':'resnetv1pyr1_stage_8_shortcuthybridsequential0_conv0_fwd',   'inputs':[24],        'fl_w': 8, 'fl':3, 'fl_y':3, 'rs':8, 'type':'conv'} 

        #.............
        self._layer_map[26]= {'name':'resnetv1pyr1_stage_8bottleneckv10__plus0', 'inputs':[23,25],               'fl':3,    'fl_y':3,        'type':'add'} 
        self._layer_map[27]= {'name':'resnetv1pyr1_stage_8bottleneckv10_relu2_fwd', 'inputs':[26],  'type':'relu'} 
        self._layer_map[28]= {'name':'resnetv1pyr1_stage_8bottleneckv11_hybridsequential0_conv0_fwd',   'inputs':[27],        'fl_w': 10, 'fl':3, 'fl_y':3, 'rs':10, 'type':'conv'} 
        self._layer_map[29]= {'name':'resnetv1pyr1_stage_8bottleneckv11_hybridsequential1_conv0_fwd',   'inputs':[28],        'fl_w': 9, 'fl':3, 'fl_y':3, 'rs':9, 'type':'conv'} 
        self._layer_map[30]= {'name':'resnetv1pyr1_stage_8bottleneckv11_hybridsequential2_conv0_fwd',   'inputs':[29],        'fl_w': 7, 'fl':3, 'fl_y':3, 'rs':7, 'type':'conv'} 

        #..............
        self._layer_map[31]= {'name':'resnetv1pyr1_stage_8bottleneckv11__plus0', 'inputs':[27,30],               'fl':3,    'fl_y':3,        'type':'add'} 
        self._layer_map[32]= {'name':'resnetv1pyr1_stage_8bottleneckv11_relu2_fwd', 'inputs':[31],  'type':'relu'} 
        self._layer_map[33]= {'name':'resnetv1pyr1_stage_8bottleneckv12_hybridsequential0_conv0_fwd',   'inputs':[32],        'fl_w': 10, 'fl':3, 'fl_y':3, 'rs':10, 'type':'conv'} 
        self._layer_map[34]= {'name':'resnetv1pyr1_stage_8bottleneckv12_hybridsequential1_conv0_fwd',   'inputs':[33],        'fl_w': 9, 'fl':3, 'fl_y':3, 'rs':9, 'type':'conv'} 
        self._layer_map[35]= {'name':'resnetv1pyr1_stage_8bottleneckv12_hybridsequential2_conv0_fwd',   'inputs':[34],        'fl_w': 7, 'fl':3, 'fl_y':3, 'rs':7, 'type':'conv'} 

        #..............
        self._layer_map[36]= {'name':'resnetv1pyr1_stage_8bottleneckv12__plus0', 'inputs':[32,35],              'fl':3,    'fl_y':3,         'type':'add'} 
        self._layer_map[37]= {'name':'resnetv1pyr1_stage_8bottleneckv12_relu2_fwd', 'inputs':[36],  'type':'relu'} 
        self._layer_map[38]= {'name':'resnetv1pyr1_stage_8bottleneckv13_hybridsequential0_conv0_fwd',   'inputs':[37],        'fl_w': 10, 'fl':3, 'fl_y':3, 'rs':10, 'type':'conv'} 
        self._layer_map[39]= {'name':'resnetv1pyr1_stage_8bottleneckv13_hybridsequential1_conv0_fwd',   'inputs':[38],        'fl_w': 9, 'fl':3, 'fl_y':3, 'rs':9,   'type':'conv'} 
        self._layer_map[40]= {'name':'resnetv1pyr1_stage_8bottleneckv13_hybridsequential2_conv0_fwd',   'inputs':[39],        'fl_w': 7, 'fl':3, 'fl_y':3, 'rs':7,   'type':'conv'} 

        #............
        self._layer_map[41]= {'name':'resnetv1pyr1_stage_8bottleneckv13__plus0', 'inputs':[37,40],              'fl':3,    'fl_y':3,         'type':'add'} 
        self._layer_map[42]= {'name':'resnetv1pyr1_stage_8bottleneckv13_relu2_fwd', 'inputs':[41],  'type':'relu'} 
        self._layer_map[43]= {'name':'resnetv1pyr1_hybridsequential6_conv0_fwd',   'inputs':[42],        'fl_w': 10, 'fl':3, 'fl_y':3, 'rs':10, 'type':'conv'} 
  

        #.............
        self._layer_map[44]= {'name':'resnetv1pyr1_stage_16bottleneckv10_hybridsequential0_conv0_fwd', 'inputs':[42],           'fl_w': 10, 'fl':3, 'fl_y':3, 'rs':10, 'type':'conv'} 
        self._layer_map[45]= {'name':'resnetv1pyr1_stage_16bottleneckv10_hybridsequential1_conv0_fwd', 'inputs':[44],           'fl_w': 11, 'fl':3, 'fl_y':4, 'rs':10, 'type':'conv'} 
        self._layer_map[46]= {'name':'resnetv1pyr1_stage_16bottleneckv10_hybridsequential2_conv0_fwd',   'inputs':[45],         'fl_w': 9, 'fl':4, 'fl_y':4, 'rs':9, 'type':'conv'} 
  
        #.............
        self._layer_map[47]= {'name':'resnetv1pyr1_stage_16_shortcutpool0_fwd',   'inputs':[42],  'fl':3, 'fl_y':3, 'type':'avg_pool'} 
        self._layer_map[48]= {'name':'resnetv1pyr1_stage_16_shortcuthybridsequential0_conv0_fwd',   'inputs':[47],        'fl_w': 9, 'fl':3, 'fl_y':4, 'rs':8, 'type':'conv'} 

        #..............
        self._layer_map[49]= {'name':'resnetv1pyr1_stage_16bottleneckv10__plus0', 'inputs':[46,48],              'fl':4,    'fl_y':4,         'type':'add'} 
        self._layer_map[50]= {'name':'resnetv1pyr1_stage_16bottleneckv10_relu2_fwd', 'inputs':[49],  'type':'relu'} 
        self._layer_map[51]= {'name':'resnetv1pyr1_stage_16bottleneckv11_hybridsequential0_conv0_fwd',   'inputs':[50],        'fl_w': 10, 'fl':4, 'fl_y':3, 'rs':11, 'type':'conv'} 
        self._layer_map[52]= {'name':'resnetv1pyr1_stage_16bottleneckv11_hybridsequential1_conv0_fwd',   'inputs':[51],        'fl_w': 10, 'fl':3, 'fl_y':4, 'rs':9,   'type':'conv'} 
        self._layer_map[53]= {'name':'resnetv1pyr1_stage_16bottleneckv11_hybridsequential2_conv0_fwd',   'inputs':[52],        'fl_w': 8, 'fl':4, 'fl_y':4, 'rs':8,   'type':'conv'} 


        #................
        self._layer_map[54]= {'name':'resnetv1pyr1_stage_16bottleneckv11__plus0', 'inputs':[50,53],              'fl':4,    'fl_y':4,         'type':'add'} 
        self._layer_map[55]= {'name':'resnetv1pyr1_stage_16bottleneckv11_relu2_fwd', 'inputs':[54],  'type':'relu'} 
        self._layer_map[56]= {'name':'resnetv1pyr1_stage_16bottleneckv12_hybridsequential0_conv0_fwd',   'inputs':[55],        'fl_w': 10, 'fl':4, 'fl_y':3, 'rs':11, 'type':'conv'} 
        self._layer_map[57]= {'name':'resnetv1pyr1_stage_16bottleneckv12_hybridsequential1_conv0_fwd',   'inputs':[56],        'fl_w': 10, 'fl':3, 'fl_y':3, 'rs':10,   'type':'conv'} 
        self._layer_map[58]= {'name':'resnetv1pyr1_stage_16bottleneckv12_hybridsequential2_conv0_fwd',   'inputs':[57],        'fl_w': 8, 'fl':3, 'fl_y':4, 'rs':7,   'type':'conv'} 

        #.........
        self._layer_map[59]= {'name':'resnetv1pyr1_stage_16bottleneckv12__plus0', 'inputs':[55,58],              'fl':4,    'fl_y':4,         'type':'add'} 
        self._layer_map[60]= {'name':'resnetv1pyr1_stage_16bottleneckv12_relu2_fwd', 'inputs':[59],  'type':'relu'} 
        self._layer_map[61]= {'name':'resnetv1pyr1_hybridsequential4_conv0_fwd',   'inputs':[60],        'fl_w': 11, 'fl':4, 'fl_y':4, 'rs':11, 'type':'conv'} 
   

        #.................
        self._layer_map[62]= {'name':'resnetv1pyr1_stage_16_dbottleneckv10_hybridsequential0_conv0_fwd',   'inputs':[60],        'fl_w': 10, 'fl':4, 'fl_y':3, 'rs':11, 'type':'conv'} 
        self._layer_map[63]= {'name':'resnetv1pyr1_stage_16_dbottleneckv10_hybridsequential1_conv0_fwd',   'inputs':[62],        'fl_w': 9, 'fl':3, 'fl_y':4, 'rs':8,   'type':'conv'} 
        self._layer_map[64]= {'name':'resnetv1pyr1_stage_16_dbottleneckv10_hybridsequential2_conv0_fwd',   'inputs':[63],        'fl_w': 8, 'fl':4, 'fl_y':4, 'rs':8,   'type':'conv'} 


        #...............
        self._layer_map[65]= {'name':'resnetv1pyr1_stage_16_d_shortcuthybridsequential0_conv0_fwd',   'inputs':[60],            'fl_w': 10, 'fl':4, 'fl_y':4, 'rs':10,   'type':'conv'} 
       
        #................
        self._layer_map[66]= {'name':'resnetv1pyr1_stage_16_dbottleneckv10__plus0', 'inputs':[64,65],              'fl':4,    'fl_y':4,         'type':'add'} 
        self._layer_map[67]= {'name':'resnetv1pyr1_stage_16_dbottleneckv10_relu2_fwd', 'inputs':[66],  'type':'relu'} 
        self._layer_map[68]= {'name':'resnetv1pyr1_stage_16_dbottleneckv11_hybridsequential0_conv0_fwd',   'inputs':[67],        'fl_w': 10, 'fl':4, 'fl_y':3, 'rs':11, 'type':'conv'} 
        self._layer_map[69]= {'name':'resnetv1pyr1_stage_16_dbottleneckv11_hybridsequential1_conv0_fwd',   'inputs':[68],        'fl_w': 9, 'fl':3, 'fl_y':4, 'rs':8,   'type':'conv'} 
        self._layer_map[70]= {'name':'resnetv1pyr1_stage_16_dbottleneckv11_hybridsequential2_conv0_fwd',   'inputs':[69],        'fl_w': 8, 'fl':4, 'fl_y':4, 'rs':8,   'type':'conv'} 


        #..................
        self._layer_map[71]= {'name':'resnetv1pyr1_stage_16_dbottleneckv11__plus0', 'inputs':[67,70],               'fl':4,    'fl_y':4,        'type':'add'} 
        self._layer_map[72]= {'name':'resnetv1pyr1_stage_16_dbottleneckv11_relu2_fwd', 'inputs':[71],  'type':'relu'} 
        self._layer_map[73]= {'name':'resnetv1pyr1_stage_16_dbottleneckv12_hybridsequential0_conv0_fwd',   'inputs':[72],        'fl_w': 10, 'fl':4, 'fl_y':3, 'rs':11, 'type':'conv'} 
        self._layer_map[74]= {'name':'resnetv1pyr1_stage_16_dbottleneckv12_hybridsequential1_conv0_fwd',   'inputs':[73],        'fl_w': 9, 'fl':3, 'fl_y':3, 'rs':9,   'type':'conv'} 
        self._layer_map[75]= {'name':'resnetv1pyr1_stage_16_dbottleneckv12_hybridsequential2_conv0_fwd',   'inputs':[74],        'fl_w': 8, 'fl':3, 'fl_y':4, 'rs':7,   'type':'conv'} 
       
        #..................
        self._layer_map[76]= {'name':'resnetv1pyr1_stage_16_dbottleneckv12__plus0',     'inputs':[72,75],              'fl':4,    'fl_y':4,         'type':'add'} 
        self._layer_map[77]= {'name':'resnetv1pyr1_stage_16_dbottleneckv12_relu2_fwd',  'inputs':[76],  'type':'relu'} 
        self._layer_map[78]= {'name':'resnetv1pyr1_hybridsequential3_conv0_fwd',         'inputs':[77], 'fl_w': 10, 'fl':4, 'fl_y':4, 'rs':10, 'type':'conv'} 
   
        #..................
        self._layer_map[79]= {'name':'resnetv1pyr1__plus0',                       'inputs':[61,78],                'fl':4,    'fl_y':4,           'type':'add'} 
        self._layer_map[80]= {'name':'resnetv1pyr1_hybridsequential5_conv0_fwd',  'inputs':[79],        'fl_w': 12, 'fl':4, 'fl_y':3, 'rs':13, 'type':'conv'} 
        self._layer_map[81]= {'name':'resnetv1pyr1_upsampling0',                  'inputs':[80],        'fl_w': 0, 'fl':3, 'fl_y':3, 'rs':0, 'type':'deconv'}  # TODO
        
        #..................
        self._layer_map[82]= {'name':'resnetv1pyr1__plus1',                       'inputs':[43,81],                'fl':3,    'fl_y':3,           'type':'add'} 
        self._layer_map[83]= {'name':'resnetv1pyr1_hybridsequential7_conv0_fwd',  'inputs':[82],        'fl_w': 13, 'fl':3, 'fl_y':3, 'rs':13, 'type':'conv'} 
        self._layer_map[84]= {'name':'resnetv1pyr1_upsampling1',                  'inputs':[83],        'fl_w': 0, 'fl':3, 'fl_y':3, 'rs':0, 'type':'deconv'}     #TODO
   
        #..................
        self._layer_map[85]= {'name':'resnetv1pyr1_hybridsequential8_conv0_fwd',            'inputs':[20],        'fl_w': 10, 'fl':3, 'fl_y':3, 'rs':10, 'type':'conv'}   
          

        #............output........
        self._layer_map[86]= {'name':'resnetv1pyr1__plus2',   'inputs':[84,85],                  'fl':3,    'fl_y':3,            'type':'add'} 
        self._layer_map[87]= {'name':'resnetv1pyr1_hybridsequential9_conv0_fwd',   'inputs':[86],                  'fl_w': 11, 'fl':3,'fl_y':3, 'rs':11,   'type':'conv'} 
        self._layer_map[88]= {'name':'resnetv1pyr1_share_head_4hybridsequential0_conv0_fwd',   'inputs':[87],      'fl_w': 8, 'fl':3, 'fl_y':2, 'rs':9,   'type':'conv'} 
        self._layer_map[89]= {'name':'resnetv1pyr1_cls_head_4hybridsequential0_conv0_fwd',   'inputs':[88],        'fl_w': 8, 'fl':2, 'fl_y':2, 'rs':8,   'type':'conv'} 
        self._layer_map[90]= {'name':'resnetv1pyr1_cls_head_4hybridsequential1_conv0_fwd',   'inputs':[89],        'fl_w': 8, 'fl':2, 'fl_y':2, 'rs':8,   'type':'conv'} 
        self._layer_map[91]= {'name':'resnetv1pyr1_cls_head_4hybridsequential2_conv0_fwd',   'inputs':[90],        'fl_w': 8, 'fl':2, 'fl_y':2, 'rs':8,   'type':'conv'} 

        #............output........
        self._layer_map[92]= {'name':'resnetv1pyr1_share_head_8hybridsequential0_conv0_fwd',   'inputs':[83],      'fl_w': 8, 'fl':3, 'fl_y':3, 'rs':8,   'type':'conv'} 
        self._layer_map[93]= {'name':'resnetv1pyr1_cls_head_8hybridsequential0_conv0_fwd',   'inputs':[92],        'fl_w': 8, 'fl':3, 'fl_y':3, 'rs':8,   'type':'conv'} 
        self._layer_map[94]= {'name':'resnetv1pyr1_cls_head_8hybridsequential1_conv0_fwd',   'inputs':[93],        'fl_w': 8, 'fl':3, 'fl_y':3, 'rs':8,   'type':'conv'} 
        self._layer_map[95]= {'name':'resnetv1pyr1_cls_head_8hybridsequential2_conv0_fwd',   'inputs':[94],        'fl_w': 8, 'fl':3, 'fl_y':2, 'rs':9,   'type':'conv'} 

        #............output........
        self._layer_map[96]= {'name':'resnetv1pyr1_share_head_16hybridsequential0_conv0_fwd',   'inputs':[80],      'fl_w': 9, 'fl':4, 'fl_y':1, 'rs':12,   'type':'conv'} 
        self._layer_map[97]= {'name':'resnetv1pyr1_cls_head_16hybridsequential0_conv0_fwd',   'inputs':[96],        'fl_w': 9, 'fl':1, 'fl_y':1, 'rs':9,   'type':'conv'} 
        self._layer_map[98]= {'name':'resnetv1pyr1_cls_head_16hybridsequential1_conv0_fwd',   'inputs':[97],        'fl_w': 9, 'fl':1, 'fl_y':1, 'rs':9,   'type':'conv'} 
        self._layer_map[99]= {'name':'resnetv1pyr1_cls_head_16hybridsequential2_conv0_fwd',   'inputs':[98],        'fl_w': 8, 'fl':1, 'fl_y':2, 'rs':7,   'type':'conv'} 
  
        #............output........
        self._layer_map[100]= {'name':'resnetv1pyr1_reg_head_4hybridsequential0_conv0_fwd',   'inputs':[88],        'fl_w': 8, 'fl':2, 'fl_y':3, 'rs':7,   'type':'conv'} 
        self._layer_map[101]= {'name':'resnetv1pyr1_reg_head_4hybridsequential1_conv0_fwd',   'inputs':[100],       'fl_w': 8, 'fl':3, 'fl_y':5, 'rs':6,   'type':'conv'} 
        self._layer_map[102]= {'name':'resnetv1pyr1_reg_head_4hybridsequential2_conv0_fwd',   'inputs':[101],       'fl_w': 7, 'fl':5, 'fl_y':6, 'rs':6,   'type':'conv'} 
  
        #............output........
        self._layer_map[103]= {'name':'resnetv1pyr1_reg_head_8hybridsequential0_conv0_fwd',   'inputs':[92],        'fl_w': 9, 'fl':3, 'fl_y':4, 'rs':8,   'type':'conv'} 
        self._layer_map[104]= {'name':'resnetv1pyr1_reg_head_8hybridsequential1_conv0_fwd',   'inputs':[103],       'fl_w': 9, 'fl':4, 'fl_y':4, 'rs':9,   'type':'conv'} 
        self._layer_map[105]= {'name':'resnetv1pyr1_reg_head_8hybridsequential2_conv0_fwd',   'inputs':[104],       'fl_w': 8, 'fl':4, 'fl_y':6, 'rs':6,   'type':'conv'} 
  
        #............output........
        self._layer_map[106]= {'name':'resnetv1pyr1_reg_head_16hybridsequential0_conv0_fwd',   'inputs':[96],        'fl_w': 10, 'fl':1, 'fl_y':1, 'rs':10,   'type':'conv'} 
        self._layer_map[107]= {'name':'resnetv1pyr1_reg_head_16hybridsequential1_conv0_fwd',   'inputs':[106],       'fl_w': 12, 'fl':1, 'fl_y':3, 'rs':10,   'type':'conv'} 
        self._layer_map[108]= {'name':'resnetv1pyr1_reg_head_16hybridsequential2_conv0_fwd',   'inputs':[107],       'fl_w': 10, 'fl':3, 'fl_y':5, 'rs':8,   'type':'conv'} 
  

    def setup(self):
        '''calculate the bit_rs '''
        (self.feed('data')
        	 .pad(1,name='')		#	p=1,s=2
             .conv(3, 3, 32, 2, 2, name='resnetv1pyr1_hybridsequential0_conv0_fwd',fl_w=2,fl=7,fl_y=2,rs=7) 
             .pad(1,name='')
             .conv(3, 3, 32, 1, 1, name='resnetv1pyr1_hybridsequential1_conv0_fwd',fl_w=8,fl=2,fl_y=3,rs=7) 
             .pad(1,name='')
             .conv(3, 3, 64, 1, 1, name='resnetv1pyr1_hybridsequential2_conv0_fwd',fl_w=8,fl=3,fl_y=3,rs=8)
             .pad_v1(1,name='')		#
             .max_pool(3, 3, 2, 2, name='pool0_fwd',fl=3,fl_y=3)

             .conv(1, 1, 32, 1, 1, name='resnetv1pyr1_stage_4bottleneckv10_hybridsequential0_conv0_fwd',fl_w=8,fl=3,fl_y=3,rs=8)
             .pad(1,name='')
             .conv(3, 3, 32, 1, 1, name='resnetv1pyr1_stage_4bottleneckv10_hybridsequential1_conv0_fwd',fl_w=9,fl=3,fl_y=3,rs=9)
             .conv(1, 1, 256, 1, 1, relu=False, name='resnetv1pyr1_stage_4bottleneckv10_hybridsequential2_conv0_fwd',fl_w=7,fl=3,fl_y=3,rs=7))

        (self.feed('pool0_fwd')
             .conv(1, 1, 256, 1, 1, relu=False, name='resnetv1pyr1_stage_4_shortcuthybridsequential0_conv0_fwd',fl_w=8,fl=3,fl_y=3,rs=8))

        (self.feed('resnetv1pyr1_stage_4bottleneckv10_hybridsequential2_conv0_fwd', 
                   'resnetv1pyr1_stage_4_shortcuthybridsequential0_conv0_fwd')
             .add(name='resnetv1pyr1_stage_4bottleneckv10__plus0',fl=3,fl_y=3)
             .relu(name='resnetv1pyr1_stage_4bottleneckv10_relu2_fwd')
             .conv(1, 1, 32, 1, 1, name='resnetv1pyr1_stage_4bottleneckv11_hybridsequential0_conv0_fwd',fl_w=9,fl=3,fl_y=3,rs=9)
             .pad(1,name='')
             .conv(3, 3, 32, 1, 1, name='resnetv1pyr1_stage_4bottleneckv11_hybridsequential1_conv0_fwd',fl_w=9,fl=3,fl_y=3,rs=9)
             .conv(1, 1, 256, 1, 1, relu=False, name='resnetv1pyr1_stage_4bottleneckv11_hybridsequential2_conv0_fwd',fl_w=7,fl=3,fl_y=3,rs=7))

        (self.feed('resnetv1pyr1_stage_4bottleneckv11_hybridsequential2_conv0_fwd', 
                   'resnetv1pyr1_stage_4bottleneckv10_relu2_fwd')
             .add(name='resnetv1pyr1_stage_4bottleneckv11__plus0',fl=3,fl_y=3)
             .relu(name='resnetv1pyr1_stage_4bottleneckv11_relu2_fwd')
             .conv(1, 1, 32, 1, 1, name='resnetv1pyr1_stage_4bottleneckv12_hybridsequential0_conv0_fwd',fl_w=9,fl=3,fl_y=3,rs=9)
             .pad(1,name='')
             .conv(3, 3, 32, 1, 1, name='resnetv1pyr1_stage_4bottleneckv12_hybridsequential1_conv0_fwd',fl_w=9,fl=3,fl_y=3,rs=9)
             .conv(1, 1, 256, 1, 1, relu=False, name='resnetv1pyr1_stage_4bottleneckv12_hybridsequential2_conv0_fwd',fl_w=7,fl=3,fl_y=3,rs=7))

        (self.feed('resnetv1pyr1_stage_4bottleneckv12_hybridsequential2_conv0_fwd', 
                   'resnetv1pyr1_stage_4bottleneckv11_relu2_fwd')
             .add(name='resnetv1pyr1_stage_4bottleneckv12__plus0',fl=3,fl_y=3)
             .relu(name='resnetv1pyr1_stage_4bottleneckv12_relu2_fwd')
             .pad()		#p=0,s=2
             .conv(1, 1, 128, 2, 2, name='resnetv1pyr1_stage_8bottleneckv10_hybridsequential0_conv0_fwd',fl_w=9,fl=3,fl_y=3,rs=9)
             .pad(1,name='')
             .conv(3, 3, 128, 1, 1, name='resnetv1pyr1_stage_8bottleneckv10_hybridsequential1_conv0_fwd',fl_w=10,fl=3,fl_y=3,rs=10)
             .conv(1, 1, 512, 1, 1, relu=False, name='resnetv1pyr1_stage_8bottleneckv10_hybridsequential2_conv0_fwd',fl_w=8,fl=3,fl_y=3,rs=8))

        (self.feed('resnetv1pyr1_stage_4bottleneckv12_relu2_fwd')
             .pad_v1()	#TODO
             .avg_pool(2, 2, 2, 2, name='resnetv1pyr1_stage_8_shortcutpool0_fwd',fl=3,fl_y=3)
             .conv(1, 1, 512, 1, 1, relu=False, name='resnetv1pyr1_stage_8_shortcuthybridsequential0_conv0_fwd',fl_w=8,fl=3,fl_y=3,rs=8))

        (self.feed('resnetv1pyr1_stage_8bottleneckv10_hybridsequential2_conv0_fwd', 
                   'resnetv1pyr1_stage_8_shortcuthybridsequential0_conv0_fwd')
             .add(name='resnetv1pyr1_stage_8bottleneckv10__plus0',fl=3,fl_y=3)
             .relu(name='resnetv1pyr1_stage_8bottleneckv10_relu2_fwd')
             .conv(1, 1, 64, 1, 1, name='resnetv1pyr1_stage_8bottleneckv11_hybridsequential0_conv0_fwd',fl_w=10,fl=3,fl_y=3,rs=10)
             .pad(1,name='')             
             .conv(3, 3, 64, 1, 1, name='resnetv1pyr1_stage_8bottleneckv11_hybridsequential1_conv0_fwd',fl_w=9,fl=3,fl_y=3,rs=9)
             .conv(1, 1, 512, 1, 1, relu=False, name='resnetv1pyr1_stage_8bottleneckv11_hybridsequential2_conv0_fwd',fl_w=7,fl=3,fl_y=3,rs=7))

        (self.feed('resnetv1pyr1_stage_8bottleneckv11_hybridsequential2_conv0_fwd', 
                   'resnetv1pyr1_stage_8bottleneckv10_relu2_fwd')
             .add(name='resnetv1pyr1_stage_8bottleneckv11__plus0',fl=3,fl_y=3)
             .relu(name='resnetv1pyr1_stage_8bottleneckv11_relu2_fwd')
             .conv(1, 1, 64, 1, 1, name='resnetv1pyr1_stage_8bottleneckv12_hybridsequential0_conv0_fwd',fl_w=10,fl=3,fl_y=3,rs=10)
             .pad(1,name='')             
             .conv(3, 3, 64, 1, 1, name='resnetv1pyr1_stage_8bottleneckv12_hybridsequential1_conv0_fwd',fl_w=9,fl=3,fl_y=3,rs=9)
             .conv(1, 1, 512, 1, 1, relu=False, name='resnetv1pyr1_stage_8bottleneckv12_hybridsequential2_conv0_fwd',fl_w=7,fl=3,fl_y=3,rs=7))

        (self.feed('resnetv1pyr1_stage_8bottleneckv12_hybridsequential2_conv0_fwd', 
                   'resnetv1pyr1_stage_8bottleneckv11_relu2_fwd')
             .add(name='resnetv1pyr1_stage_8bottleneckv12__plus0',fl=3,fl_y=3)
             .relu(name='resnetv1pyr1_stage_8bottleneckv12_relu2_fwd')
             .conv(1, 1, 64, 1, 1, name='resnetv1pyr1_stage_8bottleneckv13_hybridsequential0_conv0_fwd',fl_w=10,fl=3,fl_y=3,rs=10)
             .pad(1,name='')            
             .conv(3, 3, 64, 1, 1, name='resnetv1pyr1_stage_8bottleneckv13_hybridsequential1_conv0_fwd',fl_w=9,fl=3,fl_y=3,rs=9)
             .conv(1, 1, 512, 1, 1, relu=False, name='resnetv1pyr1_stage_8bottleneckv13_hybridsequential2_conv0_fwd',fl_w=7,fl=3,fl_y=3,rs=7))

        (self.feed('resnetv1pyr1_stage_8bottleneckv13_hybridsequential2_conv0_fwd', 
                   'resnetv1pyr1_stage_8bottleneckv12_relu2_fwd')
             .add(name='resnetv1pyr1_stage_8bottleneckv13__plus0',fl=3,fl_y=3)  
             .relu(name='resnetv1pyr1_stage_8bottleneckv13_relu2_fwd')
             .conv(1, 1, 128, 1, 1, relu=False, name='resnetv1pyr1_hybridsequential6_conv0_fwd',fl_w=10,fl=3,fl_y=3,rs=10))

        (self.feed('resnetv1pyr1_stage_8bottleneckv13_relu2_fwd')
             .conv(1, 1, 256, 2, 2, name='resnetv1pyr1_stage_16bottleneckv10_hybridsequential0_conv0_fwd',fl_w=10,fl=3,fl_y=3,rs=10)
             .pad(1,name='')
             .conv(3, 3, 256, 1, 1, name='resnetv1pyr1_stage_16bottleneckv10_hybridsequential1_conv0_fwd',fl_w=11,fl=3,fl_y=4,rs=10)
             .conv(1, 1, 1024, 1, 1, relu=False, name='resnetv1pyr1_stage_16bottleneckv10_hybridsequential2_conv0_fwd',fl_w=9,fl=4,fl_y=4,rs=9))

        (self.feed('resnetv1pyr1_stage_8bottleneckv13_relu2_fwd')
        	 .pad_v1()		#TODO
             .avg_pool(2, 2, 2, 2, name='resnetv1pyr1_stage_16_shortcutpool0_fwd',fl=3,fl_y=3)
             .conv(1, 1, 1024, 1, 1, relu=False, name='resnetv1pyr1_stage_16_shortcuthybridsequential0_conv0_fwd',fl_w=9,fl=3,fl_y=4,rs=8))

        (self.feed('resnetv1pyr1_stage_16bottleneckv10_hybridsequential2_conv0_fwd', 
                   'resnetv1pyr1_stage_16_shortcuthybridsequential0_conv0_fwd')
             .add(name='resnetv1pyr1_stage_16bottleneckv10__plus0',fl=4,fl_y=4)     
             .relu(name='resnetv1pyr1_stage_16bottleneckv10_relu2_fwd')
             .conv(1, 1, 128, 1, 1, name='resnetv1pyr1_stage_16bottleneckv11_hybridsequential0_conv0_fwd',fl_w=10,fl=4,fl_y=3,rs=11)
             .pad(1,name='')
             .conv(3, 3, 128, 1, 1, name='resnetv1pyr1_stage_16bottleneckv11_hybridsequential1_conv0_fwd',fl_w=10,fl=3,fl_y=4,rs=9)
             .conv(1, 1, 1024, 1, 1, relu=False, name='resnetv1pyr1_stage_16bottleneckv11_hybridsequential2_conv0_fwd',fl_w=8,fl=4,fl_y=4,rs=8))

        (self.feed('resnetv1pyr1_stage_16bottleneckv11_hybridsequential2_conv0_fwd', 
                   'resnetv1pyr1_stage_16bottleneckv10_relu2_fwd')
             .add(name='resnetv1pyr1_stage_16bottleneckv11__plus0',fl=4,fl_y=4)
             .relu(name='resnetv1pyr1_stage_16bottleneckv11_relu2_fwd')
             .conv(1, 1, 128, 1, 1, name='resnetv1pyr1_stage_16bottleneckv12_hybridsequential0_conv0_fwd',fl_w=10,fl=4,fl_y=3,rs=11)
             .pad(1,name='')
             .conv(3, 3, 128, 1, 1, name='resnetv1pyr1_stage_16bottleneckv12_hybridsequential1_conv0_fwd',fl_w=10,fl=3,fl_y=3,rs=10)
             .conv(1, 1, 1024, 1, 1, relu=False, name='resnetv1pyr1_stage_16bottleneckv12_hybridsequential2_conv0_fwd',fl_w=8,fl=3,fl_y=4,rs=7))


        (self.feed('resnetv1pyr1_stage_16bottleneckv12_hybridsequential2_conv0_fwd', 
                   'resnetv1pyr1_stage_16bottleneckv11_relu2_fwd')
             .add(name='resnetv1pyr1_stage_16bottleneckv12__plus0',fl=4,fl_y=4)
             .relu(name='resnetv1pyr1_stage_16bottleneckv12_relu2_fwd')
             .conv(1, 1, 256, 1, 1, relu=False, name='resnetv1pyr1_hybridsequential4_conv0_fwd',fl_w=11,fl=4,fl_y=4,rs=11))

        (self.feed('resnetv1pyr1_stage_16bottleneckv12_relu2_fwd')
             .conv(1, 1, 128, 1, 1, name='resnetv1pyr1_stage_16_dbottleneckv10_hybridsequential0_conv0_fwd',fl_w=10,fl=4,fl_y=3,rs=11)
             .pad()		# p=2,d=2
             .conv(3, 3, 128, 1, 1, name='resnetv1pyr1_stage_16_dbottleneckv10_hybridsequential1_conv0_fwd',fl_w=9,fl=3,fl_y=4,rs=8)
             .conv(1, 1, 1024, 1, 1, relu=False, name='resnetv1pyr1_stage_16_dbottleneckv10_hybridsequential2_conv0_fwd',fl_w=8,fl=4,fl_y=4,rs=8))

        (self.feed('resnetv1pyr1_stage_16bottleneckv12_relu2_fwd')
             .conv(1, 1, 1024, 1, 1, relu=False, name='resnetv1pyr1_stage_16_d_shortcuthybridsequential0_conv0_fwd',fl_w=10,fl=4,fl_y=4,rs=10))

        (self.feed('resnetv1pyr1_stage_16_dbottleneckv10_hybridsequential2_conv0_fwd', 
                   'resnetv1pyr1_stage_16_d_shortcuthybridsequential0_conv0_fwd')
             .add(name='resnetv1pyr1_stage_16_dbottleneckv10__plus0',fl=4,fl_y=4)
             .relu(name='resnetv1pyr1_stage_16_dbottleneckv10_relu2_fwd')
             .conv(1, 1, 128, 1, 1, name='resnetv1pyr1_stage_16_dbottleneckv11_hybridsequential0_conv0_fwd',fl_w=10,fl=4,fl_y=3,rs=11)
             .pad()	   #p=2, d=2
             .conv(3, 3, 128, 1, 1, name='resnetv1pyr1_stage_16_dbottleneckv11_hybridsequential1_conv0_fwd',fl_w=9,fl=3,fl_y=4,rs=8)
             .conv(1, 1, 1024, 1, 1, relu=False, name='resnetv1pyr1_stage_16_dbottleneckv11_hybridsequential2_conv0_fwd',fl_w=8,fl=4,fl_y=4,rs=8))

        (self.feed('resnetv1pyr1_stage_16_dbottleneckv11_hybridsequential2_conv0_fwd', 
                   'resnetv1pyr1_stage_16_dbottleneckv10_relu2_fwd')
             .add(name='resnetv1pyr1_stage_16_dbottleneckv11__plus0',fl=4,fl_y=4)
             .relu(name='resnetv1pyr1_stage_16_dbottleneckv11_relu2_fwd')
             .conv(1, 1, 128, 1, 1, name='resnetv1pyr1_stage_16_dbottleneckv12_hybridsequential0_conv0_fwd',fl_w=10,fl=4,fl_y=3,rs=11)
             .pad()   #p=2,d=2
             .conv(3, 3, 128, 1, 1,  name='resnetv1pyr1_stage_16_dbottleneckv12_hybridsequential1_conv0_fwd',fl_w=9,fl=3,fl_y=3,rs=9)
             .conv(1, 1, 1024, 1, 1, relu=False, name='resnetv1pyr1_stage_16_dbottleneckv12_hybridsequential2_conv0_fwd',fl_w=8,fl=3,fl_y=4,rs=7))

        (self.feed('resnetv1pyr1_stage_16_dbottleneckv12_hybridsequential2_conv0_fwd', 
                   'resnetv1pyr1_stage_16_dbottleneckv11_relu2_fwd')
             .add(name='resnetv1pyr1_stage_16_dbottleneckv12__plus0',fl=4,fl_y=4)  
             .relu(name='resnetv1pyr1_stage_16_dbottleneckv12_relu2_fwd')
             .conv(1, 1, 256, 1, 1, relu=False, name='resnetv1pyr1_hybridsequential3_conv0_fwd',fl_w=10,fl=4,fl_y=4,rs=10))

        (self.feed('resnetv1pyr1_hybridsequential4_conv0_fwd', 
                   'resnetv1pyr1_hybridsequential3_conv0_fwd')
             .add(name='resnetv1pyr1__plus0',fl=4,fl_y=4)
             .pad(1,name='')         
             .conv(3, 3, 128, 1, 1, relu=False, name='resnetv1pyr1_hybridsequential5_conv0_fwd',fl_w=12,fl=4,fl_y=3,rs=13)
             .pad()		#p=0,s=2
             .deconv(2, 2, 128, 2, 2, relu=False, name='resnetv1pyr1_upsampling0',fl_w=0,fl=3,fl_y=3,rs=0))         #TODO X=3,Y=3

        (self.feed('resnetv1pyr1_hybridsequential6_conv0_fwd', 
                   'resnetv1pyr1_upsampling0')
             .add(name='resnetv1pyr1__plus1',fl=3,fl_y=3)
             .pad(1,name='')      
             .conv(3, 3, 64, 1, 1, relu=False, name='resnetv1pyr1_hybridsequential7_conv0_fwd',fl_w=13,fl=3,fl_y=3,rs=13)
             .pad()		#p=0, s=2
             .deconv(2, 2, 64, 2, 2, relu=False, name='resnetv1pyr1_upsampling1',fl_w=0,fl=3,fl_y=3,rs=0))         #TODO X=3,Y=3

        (self.feed('resnetv1pyr1_stage_4bottleneckv12_relu2_fwd')
             .conv(1, 1, 64, 1, 1, relu=False, name='resnetv1pyr1_hybridsequential8_conv0_fwd',fl_w=10,fl=3,fl_y=3,rs=10))

        (self.feed('resnetv1pyr1_upsampling1', 
                   'resnetv1pyr1_hybridsequential8_conv0_fwd')
             .add(name='resnetv1pyr1__plus2',fl=3,fl_y=3)
             .pad(1,name='')
             .conv(3, 3, 64, 1, 1, relu=False, name='resnetv1pyr1_hybridsequential9_conv0_fwd',fl_w=11,fl=3,fl_y=3,rs=11)
             .pad(1,name='')
             .conv(3, 3, 64, 1, 1, name='resnetv1pyr1_share_head_4hybridsequential0_conv0_fwd',fl_w=8,fl=3,fl_y=2,rs=9)
             .pad(1,name='')
             .conv(3, 3, 64, 1, 1, name='resnetv1pyr1_cls_head_4hybridsequential0_conv0_fwd',fl_w=8,fl=2,fl_y=2,rs=8)
             .pad(1,name='')
             .conv(3, 3, 64, 1, 1, name='resnetv1pyr1_cls_head_4hybridsequential1_conv0_fwd',fl_w=8,fl=2,fl_y=2,rs=8)
             .conv(1, 1, 4, 1, 1, relu=False, name='resnetv1pyr1_cls_head_4hybridsequential2_conv0_fwd',fl_w=8,fl=2,fl_y=2,rs=8))

        (self.feed('resnetv1pyr1_hybridsequential7_conv0_fwd')
             .pad(1,name='')
             .conv(3, 3, 128, 1, 1, name='resnetv1pyr1_share_head_8hybridsequential0_conv0_fwd',fl_w=8,fl=3,fl_y=3,rs=8)
             .pad(1,name='')
             .conv(3, 3, 128, 1, 1, name='resnetv1pyr1_cls_head_8hybridsequential0_conv0_fwd',fl_w=8,fl=3,fl_y=3,rs=8)
             .pad(1,name='')
             .conv(3, 3, 64, 1, 1, name='resnetv1pyr1_cls_head_8hybridsequential1_conv0_fwd',fl_w=8,fl=3,fl_y=3,rs=8)
             .conv(1, 1, 4, 1, 1, relu=False, name='resnetv1pyr1_cls_head_8hybridsequential2_conv0_fwd',fl_w=8,fl=3,fl_y=2,rs=9))

        (self.feed('resnetv1pyr1_hybridsequential5_conv0_fwd')
             .pad(1,name='')
             .conv(3, 3, 256, 1, 1, name='resnetv1pyr1_share_head_16hybridsequential0_conv0_fwd',fl_w=9,fl=3,fl_y=1,rs=11)
             .pad(1,name='')
             .conv(3, 3, 128, 1, 1, name='resnetv1pyr1_cls_head_16hybridsequential0_conv0_fwd',fl_w=9,fl=1,fl_y=1,rs=9)
             .pad(1,name='')
             .conv(3, 3, 64, 1, 1, name='resnetv1pyr1_cls_head_16hybridsequential1_conv0_fwd',fl_w=9,fl=1,fl_y=1,rs=9)
             .conv(1, 1, 4, 1, 1, relu=False, name='resnetv1pyr1_cls_head_16hybridsequential2_conv0_fwd',fl_w=8,fl=1,fl_y=2,rs=7))

        (self.feed('resnetv1pyr1_share_head_4hybridsequential0_conv0_fwd')
             .pad(1,name='')
             .conv(3, 3, 64, 1, 1, name='resnetv1pyr1_reg_head_4hybridsequential0_conv0_fwd',fl_w=8,fl=2,fl_y=3,rs=7)
             .pad(1,name='')
             .conv(3, 3, 64, 1, 1, name='resnetv1pyr1_reg_head_4hybridsequential1_conv0_fwd',fl_w=8,fl=3,fl_y=5,rs=6)
             .conv(1, 1, 28, 1, 1, relu=False, name='resnetv1pyr1_reg_head_4hybridsequential2_conv0_fwd',fl_w=7,fl=5,fl_y=6,rs=6))

        (self.feed('resnetv1pyr1_share_head_8hybridsequential0_conv0_fwd')
             .pad(1,name='')
             .conv(3, 3, 128, 1, 1, name='resnetv1pyr1_reg_head_8hybridsequential0_conv0_fwd',fl_w=9,fl=3,fl_y=4,rs=8)
             .pad(1,name='')
             .conv(3, 3, 128, 1, 1, name='resnetv1pyr1_reg_head_8hybridsequential1_conv0_fwd',fl_w=9,fl=4,fl_y=4,rs=9)
             .conv(1, 1, 28, 1, 1, relu=False, name='resnetv1pyr1_reg_head_8hybridsequential2_conv0_fwd',fl_w=8,fl=4,fl_y=6,rs=6))

        (self.feed('resnetv1pyr1_share_head_16hybridsequential0_conv0_fwd')
             .pad(1,name='')             
             .conv(3, 3, 128, 1, 1, name='resnetv1pyr1_reg_head_16hybridsequential0_conv0_fwd',fl_w=10,fl=1,fl_y=1,rs=10)
             .pad(1,name='')             
             .conv(3, 3, 128, 1, 1, name='resnetv1pyr1_reg_head_16hybridsequential1_conv0_fwd',fl_w=12,fl=1,fl_y=3,rs=10)
             .conv(1, 1, 28, 1, 1, relu=False, name='resnetv1pyr1_reg_head_16hybridsequential2_conv0_fwd',fl_w=10,fl=3,fl_y=5,rs=8))


if __name__=="__main__":
    # some parameters
    em_network='lidar'
    em_isHardware=False
    image_path=r'../utils/convert_model/examples/lidar/model'
    model_path=r'../utils/convert_model/examples/lidar/test_data'

    # init session
    tf.reset_default_graph()
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

    # create network instance and load models parameters
    net = get_network(em_network, em_isHardware)
    sess.run(tf.global_variables_initializer())
    net.load(model_path, sess)
    net.run_sw_demo(sess, image_path)

