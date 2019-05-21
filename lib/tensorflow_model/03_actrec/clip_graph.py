# -*- coding: utf-8 -*-
# @Author: liusongwei
# @Date:   2019-05-16 13:54:38
# @Last Modified by:   liusongwei
# @Last Modified time: 2019-05-18 14:24:35
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

            new_input1=tf.placeholder(dtype=tf.float32,name="images")

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

            new_input1=tf.placeholder(dtype=tf.float32,name="images")

            new_input2=tf.placeholder(dtype=tf.float32,name="pool5_out")

            graph1=load_graph(model_path)
            tf.import_graph_def(graph1,input_map={
                "smallVGG16_alpha0.25_conv1_1_pool5/pool5/MaxPool:0":new_input2,
                "images:0":new_input1},name='')
            out_origin=graph.get_tensor_by_name('action/Reshape_1:0')
            print(out_origin)
            output=tf.identity(out_origin,"new_output")
            print(output)

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

    test_data=np.zeros((1,10,120,80,3))

    with tf.Graph().as_default() as graph:

        image=tf.placeholder(dtype=tf.float32,name='image_input')
        graph_def1=load_graph(model_path)

        tf.import_graph_def(graph_def1,input_map={'images:0':image},name='')
        shape=graph.get_tensor_by_name('smallVGG16_alpha0.25_conv1_1_pool5/Shape:0')
        print(shape)
        reshape1=graph.get_tensor_by_name('smallVGG16_alpha0.25_conv1_1_pool5/Reshape:0')
        print(reshape1)
        poolingout=graph.get_tensor_by_name('smallVGG16_alpha0.25_conv1_1_pool5/pool5/MaxPool:0')
        print(poolingout)
        packs=graph.get_tensor_by_name('smallVGG16_alpha0.25_conv1_1_pool5/Reshape_1/shape:0')
        print(packs)
        reshape_node=graph.get_tensor_by_name('smallVGG16_alpha0.25_conv1_1_pool5/Reshape_1:0')
        print(reshape_node)
        output_put1=graph.get_tensor_by_name('action/Reshape_1:0')
        print(output_put1)


        graph_def2=load_graph(new_model_path)
        tf.import_graph_def(graph_def2,input_map={'pool5_out:0':poolingout,
                                                    'images:0':image},name='')
        
        output_put2=tf.get_default_graph().get_tensor_by_name('new_output:0')
        output_put1=tf.identity(output_put2,name="fusion_output")
        print(output_put1)

        with tf.Session(graph=graph) as sess:

            results2=sess.run(output_put1,feed_dict={image:test_data})
            print(results2)
            print(results2.shape)

            const_graph=tf.graph_util.convert_variables_to_constants(sess,sess.graph_def,
                                                      output_node_names=['fusion_output'])

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

def load_whole_graph(pb_file):

    with tf.gfile.GFile(pb_file, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")
    return graph

def read_node_shape(model_path):

    graph=load_whole_graph(model_path)

    imges=graph.get_tensor_by_name('images:0')
    print(imges)

    shape=graph.get_tensor_by_name('smallVGG16_alpha0.25_conv1_1_pool5/Shape:0')
    print(shape)

    reshape1=graph.get_tensor_by_name('smallVGG16_alpha0.25_conv1_1_pool5/Reshape:0')
    print(reshape1)

    poolingout=graph.get_tensor_by_name('smallVGG16_alpha0.25_conv1_1_pool5/pool5/MaxPool:0')
    print(poolingout)

    packs=graph.get_tensor_by_name('smallVGG16_alpha0.25_conv1_1_pool5/Reshape_1/shape:0')
    print(packs)

    rshape2=graph.get_tensor_by_name('smallVGG16_alpha0.25_conv1_1_pool5/Reshape_1:0')
    print(rshape2)

    out=graph.get_tensor_by_name('action/Reshape_1:0')
    print(out)

if __name__=="__main__":

    model_path=r"./conv_lstm_benchmark_tf_graph.pb"
    new_model_path=r"./model.pb"
    # read_node_shape(model_path)
    # clip_graph_2(model_path)
    load_and_read_graph(model_path,r'./sub_model.pb')
    # run_clip_graph(new_model_path)
    # load_and_read_graph(model_path,new_model_path)







