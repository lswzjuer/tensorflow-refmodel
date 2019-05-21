"""
Reference Model
Copyright (c) 2019 FABU TECH AMERICA 
Author: Abinash Mohanty
"""

import tensorflow as tf
import numpy as np
import lp_op

def getFixedPoint(num, totalBits, fractionBits, mode=0):
  """
  Returns a fixed point value of num.
  num - input number
  totalBits - total number of bits
  fractionBits - number of fractional bits
  mode - 0: returns str, 1: returns float
  """
  if isinstance(num, str):
    num = float(num)
  sign = 1
  if num < 0:
    sign = -1
  if mode == 1:
    return sign*round(abs(num)*pow(2,fractionBits))/pow(2,fractionBits)
  return str(sign*round(abs(num)*pow(2,fractionBits))/pow(2,fractionBits))

# TESTBENCH TO TEST THE OP
array =-9800
data = tf.convert_to_tensor(array, dtype=tf.float32)
yy = lp_op.lp(data, 7, 3)
init = tf.global_variables_initializer()

## Launch the graph.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
y1 = sess.run(yy)

print ('output : ' + str(y1))
print( 'output2: ' + str(getFixedPoint(array,8,6)))

