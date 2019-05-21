# -*- coding: utf-8 -*-
# @Author: liusongwei
# @Date:   2019-05-13 11:38:04
# @Last Modified by:   liusongwei
# @Last Modified time: 2019-05-21 12:38:43

from .network import Network
import tensorflow as tf
import numpy as np
import os
import sys

class RCNN(Network):
    """docstring for rcnn"""

    def __init__(self, is_Hardware,is_trainable=False):

        self.inputs = []
        self.trainable = is_trainable
        self.isHardware = is_Hardware
        # self.data = tf.placeholder(tf.float32, shape=[None, None, None, 3])
        self.data = tf.placeholder(tf.float32, shape=[None, None, 120, 80,3])
        # self.im_info = tf.placeholder(tf.float32, shape=[None, 3])
        # self.keep_prob = tf.placeholder(tf.float32)
        # self.layers = dict({'data': self.data, 'im_info': self.im_info})
        self.layers=dict({'image_input':self.data})
        self._layer_map = {}
        #self.create_layer_map()
        self.set_up()

    def pre_process(self):
        pass

    def post_process(self):
        pass

    def get_output_tensor(self):
        return tf.get_default_graph().get_tensor_by_name('ground_truth_Reshape:0')


    def quantize_input_image(self,image,bit_frac):
        sh = image.shape
        image = image.flatten()
        for i in range(len(image)):
            sign=1
            if image[i]<0:
                sign=-1
            image[i]=sign*round(abs(image[i])*pow(2,bit_frac))
        image = image.reshape(sh)
        return image

    def convert_to_float_py(self,image, bit_frac):
        sh = image.shape
        image = image.flatten()
        for i in range(len(image)):
            image[i] = float(image[i])/float(pow(2,bit_frac))
        image = image.reshape(sh)
        return image

    def run_sw_demo(self,sess,image_,output_node):

        # get output tensor
        quantized_image=self.quantize_input_image(image_,7)
        feed_dict={self.data:quantized_image}
        output=sess.run(output_node,feed_dict=feed_dict)
        return output


    def load_sub_graph(self,model_path):
        graph_def = tf.GraphDef()
        with open(model_path, "rb") as f:
            graph_def.ParseFromString(f.read())
        return graph_def

    def fusion_graph(self,sub_model_path):

        graph=tf.get_default_graph()

        pool5_dq=graph.get_tensor_by_name('pool5_dequantized/Fp:0')

        graph_def1=self.load_sub_graph(sub_model_path)
        output,=tf.import_graph_def(graph_def1,input_map={'images:0':self.data,
                                                   'pool5_out:0':pool5_dq},
                                                   return_elements=["new_output:0"],name='')

        all_graph_output=tf.identity(output,"new_all_output")
        return all_graph_output


    def save_model(self,sess,model_path,output_node_name=None):

        if output_node_name is None:
            output_node_name='new_all_output'
        const_graph=tf.graph_util.convert_variables_to_constants(sess,sess.graph_def,
                                          output_node_names=[output_node_name])

        with tf.gfile.FastGFile(model_path,mode='wb') as f:
            f.write(const_graph.SerializeToString())


    def create_layer_map(self,feat_table,weights_table):
        pass

        # with open(weight_table,'r') as wf:
        #    filelines=wf.readlines()
        # file_names = [x.strip("\n") for x in filelines]
        # laye_map_list={}
        # for x in file_names:
        #     name=x.split(' ')[0]
        #     num=x.split(' ')[1]
        #     if 'kernel' in name:
        #         layer_name=name[:-7]
        #         if layer_name not in laye_map_list.keys():
        #             continue
        #         laye_map_list[lay_name]['fl_w']=num
        #     elif 'bias' in name:
        #         lay_name=name[:-5]
        #         laye_map_list[lay_name]['fl_y']=num

        # file_names =[ (x.split(' ')[0],x.split(' ')[1]) for x in file_names]

        # # image_names = [os.path.splitext(x)[0] for x in image_names]
        # # print(filelines)

        # with open(feat_table,'rb') as ff:
        #    filelines=ff.readlines()

    def set_up(self):

        (self.feed('image_input')
        	.Reshape(name='reshape')
            .conv(3, 3, 16, 1, 1, name='conv1_1',fl_w=9, fl=7, fl_y=6, rs=10, rate=1)
            .conv(3, 3, 16, 1, 1, name='conv1_2',fl_w=9, fl=6, fl_y=6, rs=9, rate=1)
            .max_pool(2, 2, 2, 2, name='pool1', padding="VALID")

            .conv(3, 3, 32, 1, 1, name='conv2_1', fl_w=9, fl=6, fl_y=7, rs=8, rate=1)
            .conv(3, 3, 32, 1, 1, name='conv2_2', fl_w=9, fl=7, fl_y=7, rs=9, rate=1)
            .max_pool(2, 2, 2, 2, name='pool2', padding="VALID")

            .conv(3, 3, 64, 1, 1, name='conv3_1', fl_w=10, fl=7, fl_y=7, rs=10, rate=1)
            .conv(3, 3, 64, 1, 1, name='conv3_2', fl_w=10, fl=7, fl_y=7, rs=10, rate=1)
            .conv(3, 3, 64, 1, 1, name='conv3_3', fl_w=10, fl=7, fl_y=7, rs=10, rate=1)
            .max_pool(2, 2, 2, 2, name='pool3', padding="VALID")  # 分支

            .conv(3, 3, 128, 1, 1, name='conv4_1', fl_w=10,fl=7, fl_y=7, rs=10, rate=1)
            .conv(3, 3, 128, 1, 1, name='conv4_2', fl_w=10, fl=7, fl_y=6, rs=11, rate=1)
            .conv(3, 3, 128, 1, 1, name='conv4_3',fl_w=10, fl=6, fl_y=6, rs=10, rate=1)
            .max_pool(2, 2, 2, 2, name='pool4', padding="VALID")  # 分支

            .conv(3, 3, 128, 1, 1, name='conv5_1', fl_w=10, fl=6, fl_y=5, rs=11, rate=1)
            .conv(3, 3, 128, 1, 1, name='conv5_2',fl_w=10, fl=5, fl_y=5, rs=10, rate=1)
            .conv(3, 3, 128, 1, 1, name='conv5_3', fl_w=10, fl=5, fl_y=4, rs=11, rate=1)
            .max_pool(2, 2, 2, 2, name='pool5', padding="VALID")
            .inverse_quantization(name='pool5_dequantized',fl_y=4))  # 分支
            

if __name__ == '__main__':
    def load_sub_graph(model_path):
        graph_def = tf.GraphDef()
        with open(model_path, "rb") as f:
            graph_def.ParseFromString(f.read())
        return graph_def

    sub_model_path=r'../tensorflow_model/03_actrec/sub_model.pb'
    model_dir=r'../quantized_model/rcnn/act_params_change.npz'
    all_model_path=r'../tensorflow_model/03_actrec/all_quantized_model.pb'
    origin_model=r'../tensorflow_model/03_actrec/conv_lstm_benchmark_tf_graph.pb'
    
    test_data=np.zeros((1,10,120,80,3))
    with tf.Graph().as_default() as graph:


        net=RCNN(is_Hardware=True)
        out_put=net.fusion_graph(sub_model_path)
        
        config = tf.ConfigProto(allow_soft_placement=True)  
        config.gpu_options.per_process_gpu_memory_fraction = 0.6 
        config.gpu_options.allow_growth = True
        
        with tf.Session(config=config) as sess:

            # load quantized weights and biases
            net.load(model_dir,sess)

            pred=net.run_sw_demo(sess,test_data,out_put)
            print(pred)
            print(pred.shape)
            #net.save_model(sess,all_model_path)

    with tf.Graph().as_default() as graph2:

        graph_def2=load_sub_graph(origin_model)
        tf.import_graph_def(graph_def2,name='')
        
        input_node=graph2.get_tensor_by_name('images:0')
        out_put_node=graph2.get_tensor_by_name('action/Reshape_1:0')

        config = tf.ConfigProto(allow_soft_placement=True)  
        config.gpu_options.per_process_gpu_memory_fraction = 0.6 
        config.gpu_options.allow_growth = True
        
        with tf.Session(config=config) as sess:

            # load quantized weights and biases
            new_pred=sess.run(out_put_node,feed_dict={input_node:test_data})
            print(new_pred)
            print(new_pred.shape)
            #net.save_model(sess,all_model_path)