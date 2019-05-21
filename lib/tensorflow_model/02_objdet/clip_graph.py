# -*- coding: utf-8 -*-
# @Author: liusongwei
# @Date:   2019-05-16 13:54:38
# @Last Modified by:   liusongwei
# @Last Modified time: 2019-05-17 11:32:45
import numpy as np 
import tensorflow as tf

# with tf.Graph().as_default() as g1:
#   base64_str = tf.placeholder(tf.string, name='input_string')
#   input_str = tf.decode_base64(base64_str)
#   decoded_image = tf.image.decode_png(input_str, channels=1)
#   # Convert from full range of uint8 to range [0,1] of float32.
#   decoded_image_as_float = tf.image.convert_image_dtype(decoded_image,
#                                                         tf.float32)
#   decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)
#   resize_shape = tf.stack([28, 28])
#   resize_shape_as_int = tf.cast(resize_shape, dtype=tf.int32)
#   resized_image = tf.image.resize_bilinear(decoded_image_4d,
#                                            resize_shape_as_int)
#   # 展开为1维数组
#   resized_image_1d = tf.reshape(resized_image, (-1, 28 * 28))
#   print(resized_image_1d.shape)
#   tf.identity(resized_image_1d, name="DecodeJPGOutput")

# with tf.Graph().as_default() as g_combined:
#   with tf.Session(graph=g_combined) as sess:

#     x = tf.placeholder(tf.string, name="base64_input")

#     y, = tf.import_graph_def(g1def, input_map={"input_string:0": x}, return_elements=["DecodeJPGOutput:0"])

#     z, = tf.import_graph_def(g2def, input_map={"myInput:0": y}, return_elements=["myOutput:0"])
#     tf.identity(z, "myOutput")

#     tf.saved_model.simple_save(sess,
#               "./modelbase64",
#               inputs={"base64_input": x},
#               outputs={"myOutput": z})

def load_graph(model_path):

    graph_def = tf.GraphDef()
    with open(model_path, "rb") as f:
        graph_def.ParseFromString(f.read())
        return graph_def

def clip_graph_(model_path):
    with tf.Graph().as_default() as graph:
        with tf.Session(graph=graph) as sess:

            new_input=tf.placeholder(dtype=tf.float32,name="new_skip_layer_sum_0_1")

            graph1=load_graph(model_path)
            z,=tf.import_graph_def(graph1,input_map={"skip_layer_sum_0_1/add:0":new_input},
                                        return_elements=["ground_truth/Reshape:0"])
            
            output=tf.identity(z,"new_ground_truth/Reshape")

            const_graph=tf.graph_util.convert_variables_to_constants(sess,sess.graph_def,
                                                      output_node_names=['new_ground_truth/Reshape'])

            # 训练完成之后保存模型为.pb文件
            with tf.gfile.FastGFile('./model.pb',mode='wb') as f:
                f.write(const_graph.SerializeToString())

def clip_graph_2(model_path):

    with tf.Graph().as_default() as graph:
        with tf.Session(graph=graph) as sess:

            new_input1=tf.placeholder(dtype=tf.float32,name="conv3_3_reg_BiasAdd")

            new_input2=tf.placeholder(dtype=tf.float32,name="conv4_3_reg_BiasAdd")

            new_input3=tf.placeholder(dtype=tf.float32,name="conv5_3_reg_BiasAdd")

            new_input4=tf.placeholder(dtype=tf.float32,name="conv3_3_cls_BiasAdd")

            new_input5=tf.placeholder(dtype=tf.float32,name="conv4_3_cls_BiasAdd")

            new_input6=tf.placeholder(dtype=tf.float32,name="conv5_3_cls_BiasAdd")


            graph1=load_graph(model_path)
            tf.import_graph_def(graph1,input_map={"conv3_3_reg/BiasAdd:0":new_input1,"conv4_3_reg/BiasAdd:0":new_input2,"conv5_3_reg/BiasAdd:0":new_input3,
                "conv3_3_cls/BiasAdd:0":new_input4,"conv4_3_cls/BiasAdd:0":new_input5,"conv5_3_cls/BiasAdd:0":new_input6},name=''
                                        )
            output=tf.get_default_graph().get_tensor_by_name("targets/concat:0")

            output=tf.identity(output,"new_output")

            const_graph=tf.graph_util.convert_variables_to_constants(sess,sess.graph_def,
                                                      output_node_names=['new_output'])

            # 训练完成之后保存模型为.pb文件
            with tf.gfile.FastGFile('./sub_model.pb',mode='wb') as f:
                f.write(const_graph.SerializeToString())    

def show_all_node():
    # review the graph node
    tensor_node_list = [tensor for tensor in tf.get_default_graph().as_graph_def().node]
    for tensor in tensor_node_list:
        print('{}'.format(tensor.name), '\n')

def split_box(pb_file):
    name=os.path.split(pb_file)
    output_graph_path=os.path.join(name[0],'graph\\')
    return output_graph_path



def load_and_read_graph(model_path,new_model_path):

    test_data=np.zeros((1,1080,1920,3))

    with tf.Graph().as_default() as graph:

        image=tf.placeholder(dtype=tf.float32,shape=[None,1080,1920,3],name='image')
        graph_def1=load_graph(model_path)

        tf.import_graph_def(graph_def1,input_map={'image:0':image},name='')
        conv33=tf.get_default_graph().get_tensor_by_name('conv3_3_reg/BiasAdd:0')
        print(conv33)
        conv43=tf.get_default_graph().get_tensor_by_name('conv4_3_reg/BiasAdd:0')
        print(conv43)
        conv53=tf.get_default_graph().get_tensor_by_name('conv5_3_reg/BiasAdd:0')
        print(conv53)
        conv33_cls=tf.get_default_graph().get_tensor_by_name('conv3_3_cls/BiasAdd:0')
        print(conv33_cls)
        conv43_cls=tf.get_default_graph().get_tensor_by_name('conv4_3_cls/BiasAdd:0')
        print(conv43_cls)
        conv53_cls=tf.get_default_graph().get_tensor_by_name('conv5_3_cls/BiasAdd:0')
        print(conv53_cls)

        output1=tf.get_default_graph().get_tensor_by_name('targets/concat:0')
        print(output1)

        graph_def2=load_graph(new_model_path)
        tf.import_graph_def(graph_def2,input_map={'conv3_3_reg_BiasAdd:0':conv33,
            'conv4_3_reg_BiasAdd:0':conv43,'conv5_3_reg_BiasAdd:0':conv53,
            'conv3_3_cls_BiasAdd:0':conv33_cls,'conv4_3_cls_BiasAdd:0':conv43_cls,
            'conv5_3_cls_BiasAdd:0':conv53_cls},name='')
        
        output_put2=tf.get_default_graph().get_tensor_by_name('new_output:0')
        print(output_put2)
        my_name=tf.identity(output_put2,name="my_name")

        with tf.Session(graph=graph) as sess:

            results2=sess.run(output1,feed_dict={image:test_data})
            print(results2)
            print(results2.shape)

            const_graph=tf.graph_util.convert_variables_to_constants(sess,sess.graph_def,
                                                      output_node_names=['my_name'])
            # 训练完成之后保存模型为.pb文件
            with tf.gfile.FastGFile('./fuse_model_.pb',mode='wb') as f:
                f.write(const_graph.SerializeToString())


def run_clip_graph(model_path):
    test_data=np.zeros((10,136,240,9))
    with tf.Graph().as_default() as graph:
        with tf.Session(graph=graph) as sess:
            image=tf.placeholder(dtype=tf.float32,shape=[None,136,240,9])
            graph_def=load_graph(model_path)
            tf.import_graph_def(graph_def,input_map={'new_skip_layer_sum_0_1:0':image},name='')
            output_put=tf.get_default_graph().get_tensor_by_name('ground_truth/Reshape:0')
            print(output_put)
            result=sess.run(output_put,feed_dict={image:test_data})
            print(result)
            print(result.shape)            



if __name__=="__main__":

    model_path=r"./jaad_pdet_ssd_tf_graph.pb"
    new_model_path=r"./sub_model.pb"
    # clip_graph_2(model_path)
    # load_and_read_graph(model_path)
    # run_clip_graph(new_model_path)
    load_and_read_graph(model_path,new_model_path)






