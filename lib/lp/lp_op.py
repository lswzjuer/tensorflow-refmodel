"""
Reference Model
Copyright (c) 2018 MobaiTech Inc 
Author: Abinash Mohanty
"""

import tensorflow as tf
import os.path as osp

filename = osp.join(osp.dirname(__file__), 'lp.so')
_lp_module = tf.load_op_library(filename)
lp = _lp_module.lp
