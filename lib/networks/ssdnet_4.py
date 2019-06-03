# -*- coding: utf-8 -*-
# @Author: liusongwei
# @Date:   2019-05-30 14:14:19
# @Last Modified by:   liusongwei
# @Last Modified time: 2019-05-30 14:19:25
# -*- coding: utf-8 -*-
# @Author: liusongwei
# @Date:   2019-05-11 10:01:04
# @Last Modified by:   liusongwei
# @Last Modified time: 2019-05-21 16:45:33

from  network import Network 
import tensorflow as tf
import numpy as np
import os
import sys


class SSD_4(Network):
    """docstring for FCN8"""

    def __init__(self, is_Hardware,is_trainable=False):

        self.inputs = []
        self.trainable = is_trainable
        self.isHardware = is_Hardware
        # self.data = tf.placeholder(tf.float32, shape=[None, None, None, 3])
        self.data = tf.placeholder(tf.float32, shape=[None, 1080, 1920, 3],name="image_input")
        # self.im_info = tf.placeholder(tf.float32, shape=[None, 3])
        # self.keep_prob = tf.placeholder(tf.float32)
        # self.layers = dict({'data': self.data, 'im_info': self.im_info})
        self.layers=dict({'image':self.data})
        self._layer_map = {}
        #self.create_layer_map()
        self.set_up()

    def pre_process(self):
        pass

    def post_process(self):
        pass

    def get_output_tensor(self):
        return tf.get_default_graph().get_tensor_by_name('bbox_coords/concat:0')

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
        quantized_image=self.quantize_input_image(image_,5)
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

        conv3_3_reg=graph.get_tensor_by_name('conv3_3_reg_dequantized/Fp:0')
        conv4_3_reg=graph.get_tensor_by_name('conv4_3_reg_dequantized/Fp:0')
        conv5_3_reg=graph.get_tensor_by_name('conv5_3_reg_dequantized/Fp:0')

        conv3_3_cls=graph.get_tensor_by_name('conv3_3_cls_dequantized/Fp:0')
        conv4_3_cls=graph.get_tensor_by_name('conv4_3_cls_dequantized/Fp:0')
        conv5_3_cls=graph.get_tensor_by_name('conv5_3_cls_dequantized/Fp:0')


        graph_def1=self.load_sub_graph(sub_model_path)
        output,=tf.import_graph_def(graph_def1,input_map={'conv3_3_reg_BiasAdd:0':conv3_3_reg,
                                                   'conv4_3_reg_BiasAdd:0':conv4_3_reg,
                                                   'conv5_3_reg_BiasAdd:0':conv5_3_reg,
                                                   'conv3_3_cls_BiasAdd:0':conv3_3_cls,
                                                   'conv4_3_cls_BiasAdd:0':conv4_3_cls,
                                                   'conv5_3_cls_BiasAdd:0':conv5_3_cls},

                                                   return_elements=["targets/concat:0"],name='')

        all_graph_output=tf.identity(output,"new_all_output")
        return all_graph_output


    def save_model(self,sess,model_path,output_node_name=None):

        if output_node_name is None:
            output_node_name='new_all_output'
        const_graph=tf.graph_util.convert_variables_to_constants(sess,sess.graph_def,
                                          output_node_names=[output_node_name])

        with tf.gfile.FastGFile(model_path,mode='wb') as f:
            f.write(const_graph.SerializeToString())

    # def output_test_node():
    def create_layer_map(self,feat_table,weights_table):
        '''
        TODO: auto create layer_map
        '''
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

        (self.feed('image')
             .conv(3, 3, 16, 1, 1,name='conv1_1', fl_w=8, fl=5, fl_y=4, rs=9, rate=1)
             .conv(3, 3, 16, 1, 1,name='conv1_2', fl_w=7, fl=4, fl_y=3, rs=8, rate=1)
             .max_pool(2, 2, 2, 2,name='pool1',padding='VALID')

             .conv(3, 3, 32, 1, 1,name='conv2_1', fl_w=7, fl=3, fl_y=3, rs=7, rate=1)
             .conv(3, 3, 32, 1, 1,name='conv2_2', fl_w=7, fl=3, fl_y=3, rs=7, rate=1)
             .max_pool(2, 2, 2, 2,name='pool2',padding='VALID')

             .conv(3, 3, 64, 1, 1, name='conv3_1', fl_w=6, fl=3, fl_y=3, rs=6, rate=1)
             .conv(3, 3, 64, 1, 1, name='conv3_2', fl_w=7, fl=3, fl_y=2, rs=8, rate=1)
             .conv(3, 3, 64, 1, 1, name='conv3_3', fl_w=7, fl=2, fl_y=2, rs=7, rate=1)

             .max_pool(2, 2, 2, 2,name='pool3',padding='VALID')
             .conv(3, 3, 128, 1, 1,name='conv4_1', fl_w=7, fl=2, fl_y=2, rs=7, rate=1)
             .conv(3, 3, 128, 1, 1,name='conv4_2', fl_w=7, fl=2, fl_y=0, rs=9, rate=1)
             .conv(3, 3, 128, 1, 1, name='conv4_3', fl_w=7, fl=0, fl_y=0, rs=7, rate=1)

             .max_pool(2, 2, 2, 2,name='pool4',padding='VALID')
             .conv(3, 3, 128, 1, 1,name='conv5_1', fl_w=7, fl=0, fl_y=-1, rs=8, rate=1)
             .conv(3, 3, 128, 1, 1,name='conv5_2', fl_w=7, fl=-1, fl_y=-2, rs=8, rate=1)
             .conv(3, 3, 128, 1, 1, name='conv5_3', fl_w=7, fl=-2, fl_y=-1, rs=6, rate=1))


        (self.feed('conv3_3')
            .conv(1, 1, 4, 1, 1, name='conv3_3_reg', relu=False,padding='VALID',fl_w=9, fl=2, fl_y=4, rs=7, rate=1)
            .inverse_quantization(name='conv3_3_reg_dequantized',fl_y=4))


        (self.feed('conv4_3')
            .conv(1, 1, 4, 1, 1, name='conv4_3_reg',relu=False,padding='VALID', fl_w=10, fl=0, fl_y=4, rs=6, rate=1)
            .inverse_quantization(name='conv4_3_reg_dequantized',fl_y=4))

        (self.feed('conv5_3')
            .conv(1, 1, 4, 1, 1, name='conv5_3_reg',relu=False,padding='VALID', fl_w=10, fl=-1, fl_y=4, rs=5, rate=1)
            .inverse_quantization(name='conv5_3_reg_dequantized',fl_y=4))


        (self.feed('conv3_3')
            .conv(1, 1, 2, 1, 1, name='conv3_3_cls',relu=False, padding='VALID',fl_w=6, fl=2, fl_y=0, rs=8, rate=1)
            .inverse_quantization(name='conv3_3_cls_dequantized',fl_y=0))

        (self.feed('conv4_3')
            .conv(1, 1, 2, 1, 1, name='conv4_3_cls',relu=False, padding='VALID',fl_w=7, fl=0, fl_y=0, rs=7, rate=1)
            .inverse_quantization(name='conv4_3_cls_dequantized',fl_y=0))


        (self.feed('conv5_3')
            .conv(1, 1, 2, 1, 1, name='conv5_3_cls',relu=False, padding='VALID',fl_w=7, fl=-1, fl_y=0, rs=6, rate=1)
            .inverse_quantization(name='conv5_3_cls_dequantized',fl_y=0))



if __name__ == '__main__':

    def load_sub_graph(model_path):
        graph_def = tf.GraphDef()
        with open(model_path, "rb") as f:
            graph_def.ParseFromString(f.read())
        return graph_def

    def convert_to_float_py(image, bit_frac):
        sh = image.shape
        image = image.flatten()
        for i in range(len(image)):
            image[i] = float(image[i])/float(pow(2,bit_frac))
        image = image.reshape(sh)
        return image


    sub_model_path=r'../tensorflow_model/02_objdet/sub_model.pb'
    model_dir=r'../quantized_model/ssd/obj_params_change.npz'
    all_model_path=r'../tensorflow_model/02_objdet/all_quantized_model_prune4.pb'
    origin_model=r'../tensorflow_model/02_objdet/jaad_pdet_ssd_tf_graph.pb'

    test_data=np.zeros((1,1080,1920,3))


    with tf.Graph().as_default() as graph:

        net=SSD_4(is_Hardware=True)
        out_put=net.fusion_graph(sub_model_path)
        
        config = tf.ConfigProto(allow_soft_placement=True)  
        config.gpu_options.per_process_gpu_memory_fraction = 0.6 
        config.gpu_options.allow_growth = True
        
        with tf.Session(config=config) as sess:

            # load quantized weights and biases
            net.load(model_dir,sess)
            
            # get test node 
            # testnode=net.get_output_tensor()

            # run test
            pred=net.run_sw_demo(sess,test_data,out_put)
            #pred=convert_to_float_py(pred,4)
            print(pred)
            print(pred.shape)

            net.save_model(sess,all_model_path)

    with tf.Graph().as_default() as graph2:

        graph_def2=load_sub_graph(origin_model)
        tf.import_graph_def(graph_def2,name='')

        input_node=graph2.get_tensor_by_name('image:0')
        out_put_node=graph2.get_tensor_by_name('bbox_coords/concat:0')

        config = tf.ConfigProto(allow_soft_placement=True)  
        config.gpu_options.per_process_gpu_memory_fraction = 0.6 
        config.gpu_options.allow_growth = True
        
        with tf.Session(config=config) as sess:

            # load quantized weights and biases
            new_pred=sess.run(out_put_node,feed_dict={input_node:test_data})
            print(new_pred)
            print(new_pred.shape)
            #net.save_model(sess,all_model_path)
