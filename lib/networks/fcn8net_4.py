from network import Network
import tensorflow as tf
import numpy as np
import os
import sys


class FCN8(Network):
    """docstring for FCN8"""

    def __init__(self, is_Hardware,is_trainable=False):

        self.inputs = []
        self.trainable = is_trainable
        self.isHardware = is_Hardware
        # self.data = tf.placeholder(tf.float32, shape=[None, None, None, 3])
        self.data = tf.placeholder(tf.float32, shape=[None, 1088, 1920, 3],name='image_input')
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
        quantized_image=self.quantize_input_image(image_,4)
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

        pool3_conv_out=graph.get_tensor_by_name('pool3_Conv_dequan/Fp:0')
        pool4_conv_out=graph.get_tensor_by_name('pool4_Conv_dequan/Fp:0')
        pool5_conv_out=graph.get_tensor_by_name('pool5_Conv_dequan/Fp:0')

        graph_def1=self.load_sub_graph(sub_model_path)
        output,=tf.import_graph_def(graph_def1,input_map={'pool5_Conv_Relu:0':pool5_conv_out,
                                                   'pool4_Conv_Relu:0':pool4_conv_out,
                                                   'pool3_Conv_Relu:0':pool3_conv_out},
                                                   return_elements=["new_ground_truth/Reshape:0"],name='')

        all_graph_output=tf.identity(output,"new_ground_truth/Reshape")
        return all_graph_output


    def save_model(self,sess,model_path,output_node_name=None):

        if output_node_name is None:
            output_node_name='new_ground_truth/Reshape'
        const_graph=tf.graph_util.convert_variables_to_constants(sess,sess.graph_def,
                                          output_node_names=[output_node_name])
        with tf.gfile.FastGFile(model_path,mode='wb') as f:
            f.write(const_graph.SerializeToString())

    # def output_test_node():
    def create_layer_map(self,feat_table,weights_table):
        '''
        TODO: auto create layer_map
        '''
        with open(weight_table,'r') as wf:
           filelines=wf.readlines()

        file_names = [x.strip("\n") for x in filelines]
        laye_map_list={}
        for x in file_names:
            name=x.split(' ')[0]
            num=x.split(' ')[1]
            if 'kernel' in name:
                layer_name=name[:-7]
                if layer_name not in laye_map_list.keys():
                    continue
                laye_map_list[lay_name]['fl_w']=num
            elif 'bias' in name:
                lay_name=name[:-5]
                laye_map_list[lay_name]['fl_y']=num

        file_names =[ (x.split(' ')[0],x.split(' ')[1]) for x in file_names]

        # image_names = [os.path.splitext(x)[0] for x in image_names]
        # print(filelines)

        with open(feat_table,'rb') as ff:
           filelines=ff.readlines()


    def set_up(self):

        (self.feed('image')
            .conv(3, 3, 16, 1, 1, name='conv1_1', fl_w=8, fl=4, fl_y=4, rs=8, rate=1)
            .conv(3, 3, 16, 1, 1, name='conv1_2', fl_w=8, fl=4, fl_y=3, rs=9, rate=1)
            .max_pool(2, 2, 2, 2, name='pool1', padding="VALID")

            .conv(3, 3, 32, 1, 1, name='conv2_1', fl_w=8, fl=3, fl_y=3, rs=8, rate=1)
            .conv(3, 3, 30, 1, 1, name='conv2_2', fl_w=8, fl=3, fl_y=3, rs=8, rate=1)
            .max_pool(2, 2, 2, 2, name='pool2', padding="VALID")

            .conv(3, 3, 48, 1, 1, name='conv3_1', fl_w=8, fl=3, fl_y=2, rs=9, rate=1)
            .conv(3, 3, 49, 1, 1, name='conv3_2', fl_w=9, fl=2, fl_y=2, rs=9, rate=1)
            .conv(3, 3, 50, 1, 1, name='conv3_3', fl_w=8, fl=2, fl_y=3, rs=7, rate=1)
            .max_pool(2, 2, 2, 2, name='pool3', padding="VALID")  #

            .conv(3, 3, 76, 1, 1, name='conv4_1', fl_w=8, fl=3, fl_y=4, rs=7, rate=1)
            .conv(3, 3, 56, 1, 1, name='conv4_2', fl_w=9, fl=4, fl_y=4, rs=9, rate=1)
            .conv(3, 3, 62, 1, 1, name='conv4_3', fl_w=8, fl=4, fl_y=5, rs=7, rate=1)
            .max_pool(2, 2, 2, 2, name='pool4', padding="VALID")  #

            .conv(3, 3, 82, 1, 1, name='conv5_1', fl_w=8, fl=5, fl_y=4, rs=9, rate=1)
            .conv(3, 3, 39, 1, 1, name='conv5_2', fl_w=8, fl=4, fl_y=4, rs=8, rate=1)
            .conv(3, 3, 128, 1, 1, name='conv5_3', fl_w=8, fl=4, fl_y=3, rs=9, rate=1)
            .max_pool(2, 2, 2, 2, name='pool5', padding="VALID")  # 
            .conv(1, 1, 9, 1, 1, name='pool5_Conv', fl_w=8, fl=3, fl_y=2, rs=9, rate=1)
            .inverse_quantization(name='pool5_Conv_dequan',fl_y=2))
            # .ResizeBilinear(name='pool5_Conv_Upsampled'))  # new opreation

        (self.feed('pool4')
            .conv(1, 1, 9, 1, 1, name='pool4_Conv', fl_w=8, fl=5, fl_y=2, rs=11, rate=1)
            .inverse_quantization(name='pool4_Conv_dequan',fl_y=2))

        (self.feed('pool3')
            .conv(1, 1, 9, 1, 1, name='pool3_Conv', fl_w=8, fl=3, fl_y=3, rs=8, rate=1)
            .inverse_quantization(name='pool3_Conv_dequan',fl_y=3))

        # (self.feed('pool4_Conv',
        #           'pool5_Conv_Upsampled')
        #     .add(name='skip_layer_sum_0_0')
        #     .ResizeBilinear(name='skip_layer_sum_0_0_Upsampled'))

        # (self.feed('pool3_Conv',
        #           'skip_layer_sum_0_0_Upsampled')
        #     .add(name='skip_layer_sum_0_1')
        #     .ResizeBilinear(name='skip_layer_sum_0_1Upsampled')
        #     .Max(name='act_Max', reduction_indices=-1, keep_dims=True))

        # (self.feed('act_Max',
        #     'skip_layer_sum_0_1Upsampled')
        #     .Sub(name='act_sub')
        #     .Exp(name='act_Exp')
        #     .Sum(name='act_Sum', reduction_indices=-1, keep_dims=True))


        # (self.feed('act_Exp', 'act_Sum')
        #     .RealDiv(name='act_truediv')
        #     .StridedSlice(name='ground_truth_strided_slice', 
        #                   begin=1,
        #                   end=1, 
        #                   strides=1))

        # (self.feed('act_truediv',
        #           'ground_truth_strided_slice')
        #     .Reshape(name='ground_truth_Reshape'))

if __name__ == '__main__':

    def load_sub_graph(model_path):
        graph_def = tf.GraphDef()
        with open(model_path, "rb") as f:
            graph_def.ParseFromString(f.read())
        return graph_def

    def quantize_input_image(image,bit_frac):
        sh = image.shape
        image = image.flatten()
        for i in range(len(image)):
            sign=1
            if image[i]<0:
                sign=-1
            image[i]=sign*round(abs(image[i])*pow(2,bit_frac))
        image = image.reshape(sh)
        return image

    sub_model_path=r'../tensorflow_model/01_semseg/three_clip_model.pb'
    model_dir=r'../quantized_model/fcn8_prune0.4/seg4_params_change.npz'
    all_model_path=r'../tensorflow_model/01_semseg/all_quantized_model_prune4.pb'
    origin_model_path=r'../tensorflow_model/01_semseg/mscoco_fcn_padded_lscape_prediction_tf_graph.pb'

    test_data=np.zeros((1,1088,1920,3))

    with tf.Graph().as_default() as graph1:

        net=FCN8(is_Hardware=True)
        out_put=net.fusion_graph(sub_model_path)
        print(out_put)
        
        
        config = tf.ConfigProto(allow_soft_placement=True)  
        config.gpu_options.per_process_gpu_memory_fraction = 0.6 
        config.gpu_options.allow_growth = True
        
        with tf.Session(config=config) as sess:

            # load quantized weights and biases
            net.load(model_dir,sess)

            pred=net.run_sw_demo(sess,test_data,out_put)
            print(pred)
            print(pred.shape)
            net.save_model(sess,all_model_path)


    # test the all model          
    with tf.Graph().as_default() as graph2:

        input_=tf.placeholder(dtype=tf.float32,name="new_input")
        graph_def_=load_sub_graph(all_model_path)
        output,=tf.import_graph_def(graph_def_,input_map={'image_input:0':input_},
                                                            return_elements=['new_ground_truth/Reshape:0'])

        new_output=tf.identity(output,name='new_output')
        print(new_output)

        config = tf.ConfigProto(allow_soft_placement=True)  
        config.gpu_options.per_process_gpu_memory_fraction = 0.6 
        config.gpu_options.allow_growth = True
        
        with tf.Session(config=config) as sess:

            quantized_image=quantize_input_image(test_data,4)
            pred=sess.run(new_output,feed_dict={input_:quantized_image})
            print(pred)
            print(pred.shape)

    # test the origin model
    with tf.Graph().as_default() as graph3:
        graph_def_=load_sub_graph(origin_model_path)
        tf.import_graph_def(graph_def_,name='')

        x = graph3.get_tensor_by_name('image:0')
        y = graph3.get_tensor_by_name('ground_truth/Reshape:0')

        config = tf.ConfigProto(allow_soft_placement=True)  
        config.gpu_options.per_process_gpu_memory_fraction = 0.6 
        config.gpu_options.allow_growth = True
        
        with tf.Session(config=config) as sess:

            pred=sess.run(y,feed_dict={x:test_data})
            print(pred)
            print(pred.shape)

