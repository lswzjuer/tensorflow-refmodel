import tensorflow as tf 



def load_graph(pb_file):
    """

    Parameters
    ----------
    pb_file: string
        Path to to tensorflow model protobuf file.

    Returns
    -------
    graph: tf.Graph
        The model graph that can be loaded in a session.

    """

    with tf.gfile.GFile(pb_file, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")

    tensor_node_list = [tensor for tensor in graph.as_graph_def().node]
    for tensor in tensor_node_list:
        print('{}'.format(tensor.name), '\n')

    iamge=graph.get_tensor_by_name('image:0')
    print(iamge)
    output=graph.get_tensor_by_name('residual_12/add:0')
    print(output)

    init=tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)




add_default_attributes backport_concatv2 backport_tensor_array_v3 flatten_atrous_conv fold_batch_norms fold_constants fold_old_batch_norms freeze_requantization_ranges fuse_pad_and_conv fuse_remote_graph fuse_resize_and_conv fuse_resize_pad_and_conv merge_duplicate_nodes remove_attribute remove_control_dependencies remove_device remove_nodes round_weights set_device sort_by_execution_order sparsify_gather strip_unused_nodes

# def show_all_node():
#     # review the graph node
#     tensor_node_list = [tensor for tensor in tf.get_default_graph().as_graph_def().node]
#     for tensor in tensor_node_list:
#         print('{}'.format(tensor.name), '\n')

if __name__=="__main__":

    graph_path=r'../../01_feature_extractors/03_mobilenet_v2/mobilenet_v2_tf_graph.pb'
    load_graph(graph_path)


