"""
Reference Model
Copyright (c) 2019 FABU TECH AMERICA 
Author: Abinash Mohanty
"""

import tensorflow as tf
import numpy as np
import saturate_op

# TESTBENCH TO TEST THE OP
array =  [1.0, 200.0, -340.0, 4.0, 0.0]

data = tf.convert_to_tensor(array, dtype=tf.float32)
yy = saturate_op.sp(data, 8)
init = tf.global_variables_initializer()

## Launch the graph.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
y1 = sess.run(yy)

print( 'output : ' + str(y1))
