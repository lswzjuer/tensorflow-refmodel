import numpy as py 
import sys
import os
import tensorflow as tf
import  numpy as np

# Add the kaffe module to the import path
sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), '../../')))
import lidar_relu as lidar

def classify(model_data_path, image_paths):
    '''Classify the given images using GoogleNet.'''

    # Create a placeholder for the input image
    input_node = tf.placeholder(tf.float32,
                                shape=(None, 1088, 800, 8))

    # Construct the network
    net = lidar.LIDAR({'data': input_node})

    # make output list
    output1=tf.get_default_graph().get_tensor_by_name('resnetv1pyr1_cls_head_8hybridsequential2_conv0_fwd/BiasAdd:0')
    output2=tf.get_default_graph().get_tensor_by_name('resnetv1pyr1_reg_head_8hybridsequential2_conv0_fwd/BiasAdd:0')
    output3=tf.get_default_graph().get_tensor_by_name('resnetv1pyr1_cls_head_4hybridsequential2_conv0_fwd/BiasAdd:0')
    output4=tf.get_default_graph().get_tensor_by_name('resnetv1pyr1_reg_head_4hybridsequential2_conv0_fwd/BiasAdd:0')
    output5=tf.get_default_graph().get_tensor_by_name('resnetv1pyr1_cls_head_16hybridsequential2_conv0_fwd/BiasAdd:0')
    output6=tf.get_default_graph().get_tensor_by_name('resnetv1pyr1_reg_head_16hybridsequential2_conv0_fwd/BiasAdd:0')
    OUTPUT=[output1,output2,output3,output4,output5,output6]


    # review the graph node
    tensor_node_list=[tensor for tensor in tf.get_default_graph().as_graph_def().node]
    for tensor in tensor_node_list:
        print('{}'.format(tensor.name),'\n')
        # print(sess.run(tensor.name,feed_dict={data_inputs:inputs,is_training:False}))

    # init
    init=tf.global_variables_initializer()

    with tf.Session() as sesh:
        # Load the converted parameters
        print('Loading the model')
        sesh.run(init)
        net.load(model_data_path, sesh)

        # Load the input image
        print('Loading the images')
        if not os.path.exists(image_paths):
            raise ValueError('path not exists ')
        else:
            full_path=[]
            files=os.listdir(image_paths)
            for file in files:
                full_path.append(os.path.join(image_paths,file))
            images=np.transpose(np.load(full_path[0]),(0,2,3,1))

        # Perform a forward pass through the network to get the class probabilities
        print('Classifying')
        probs = sesh.run(OUTPUT,feed_dict={input_node: images})
        for prob in probs:
            print(prob.shape)

if __name__=="__main__":

    image_paths=r'test_data'
    # print('Loading the images')
    # if not os.path.exists(image_paths):
    #     raise ValueError('path not exists ')
    # else:
    #     full_path = []
    #     files = os.listdir(image_paths)
    #     for file in files:
    #         full_path.append(os.path.join(image_paths, file))
    #     images = np.load(full_path[0])
    # print(images.shape)

    model_data=r'model/liadr_relu6.npy'
    # data_dict = np.load(model_data).item()
    # namelist=data_dict.keys()
    # for name in namelist:
    #     if type(data_dict[name]).__name__=='dict':
    #         print(' dict is : {}  shape is : {}'.format(name,data_dict[name]['weights'].shape))
    #     if 'upsampling' in name:
    #         if type(data_dict[name]).__name__ !='list':
    #             raise ValueError('type is not list ')
    #         new_dict={}
    #         new_dict['weights']=np.transpose(data_dict[name][0],(2,3,1,0))
    #         new_dict['biases']=data_dict[name][1]
    #         print(new_dict['weights'].shape)
    #         print(new_dict['biases'].shape)
    #         data_dict[name]=new_dict
    # for op_name in data_dict:
    #     for param_name, data in data_dict[op_name].iteritems():
    #         print('name is :{} and shape is :{}'.format(param_name,data.shape))

    # output results
    classify(model_data,image_paths)




