


# test the zero_out.so lib 
import tensorflow as tf
zero_out_module = tf.load_op_library('./zero_out.so')
with tf.Session() as sess:
  zero_out_module.zero_out([[1, 2], [3, 4]]).eval()
  
