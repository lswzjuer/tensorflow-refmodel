"""
Reference Model
Copyright (c) 2019 MobaiTech Inc 
Author: Abinash Mohanty
"""

import tensorflow as tf
import os.path as osp

filename = osp.join(osp.dirname(__file__), 'sp.so')
_sp_module = tf.load_op_library(filename)
sp = _sp_module.sp
