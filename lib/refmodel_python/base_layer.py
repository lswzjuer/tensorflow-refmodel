# -*- coding: utf-8 -*-
# @Author: liusongwei
# @Date:   2019-05-18 14:27:29
# @Last Modified by:   liusongwei
# @Last Modified time: 2019-06-09 20:14:49

import tensorflow as tf 
import numpy as np 

WORD_WIDTH=8
DEFAULT_PADDING="SAME"
TRAINABLE=False
ENABLE_TENSORBOARD=False


def variable_summaries(var):
        """
        Attach a lot of summaries to a Tensor (for TensorBoard visualization).
        """
        with tf.name_scope('summaries'):
                mean = tf.reduce_mean(var)
                tf.summary.scalar('mean', mean)
                with tf.name_scope('stddev'):
                        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
                tf.summary.scalar('stddev', stddev)
                tf.summary.scalar('max', tf.reduce_max(var))
                tf.summary.scalar('min', tf.reduce_min(var))
                tf.summary.histogram('histogram', var)

def make_var(name,shape,initializer=None,trainable=False,regularizer=None):
    print(shape)
    return tf.get_variable(name, shape)


def l2_regularizer(weight_decay=0.0005, scope=None):
    def regularizer(tensor):
            with tf.name_scope(scope, default_name='l2_regularizer', values=[tensor]):
                    l2_weight = tf.convert_to_tensor(weight_decay,
                                                               dtype=tensor.dtype.base_dtype,
                                                               name='weight_decay')
                    return tf.multiply(l2_weight, tf.nn.l2_loss(tensor), name='value')
    return regularizer

def conv(input, k_h, k_w, c_i, c_o, s_h, s_w, name, bw=WORD_WIDTH, fl_w=0,fl=10,fl_y=10,rs=0, rate=1,
            biased=True,relu=True, padding=DEFAULT_PADDING,trainable=TRAINABLE,isHardware=True):
        """ contribution by miraclebiu, and biased option"""
        
        if (fl_w+fl-fl_y)!=rs:
            raise ValueError("layer: {} rs is wrong ".format(name))
        rs=fl_w+fl-fl_y
        #c_i=input.get_shape().as_list()[]
        #c_i = input.get_shape()[-1].value

        if isHardware:
                convolve = lambda i, k: lp_conv(i, k, s_h, s_w, bw, fl, rs,rate, padding)               # DLA
        else:
                if rate==1:
                        convolve=lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)   # Normal Tensorflow
                else:
                        convolve = lambda i, k: tf.nn.atrous_conv2d(i, k,rate, padding=padding)   # Tensorflow

        with tf.variable_scope(name) as scope:
                init_weights = tf.contrib.layers.variance_scaling_initializer(factor=0.01, mode='FAN_AVG', uniform=False)
                #init_weights=tf.constant_initializer(1.0)
                init_biases = tf.constant_initializer(0.0)
                #print("TF DEBUG, scope: %s\n" %(name))
                kernel = make_var('weights', [k_h, k_w, c_i, c_o], init_weights, trainable, \
                                                           regularizer=l2_regularizer(0.0005))
                if ENABLE_TENSORBOARD:
                        variable_summaries(kernel)
                if biased:
                        biases = make_var('biases', [c_o], init_biases, trainable)
                        if ENABLE_TENSORBOARD:
                                variable_summaries(biases)
                        conv = convolve(input, kernel)
                        if relu:
                                bias = tf.nn.bias_add(conv, biases)
                                if isHardware:
                                        bias_s = saturate(bias, WORD_WIDTH)
                                        relu_bias_s=tf.nn.relu(bias_s)    # New addition for saturation
                                        print(relu_bias_s)
                                        return relu_bias_s
                                relu_bias=tf.nn.relu(bias)
                                print(relu_bias)
                                return relu_bias
                        bias_add = tf.nn.bias_add(conv, biases)
                        if isHardware:
                                s_biasadd=saturate(bias_add, WORD_WIDTH)  # New addition for saturation
                                print(s_biasadd)
                                return s_biasadd
                        print(bias_add)
                        return bias_add
                else:
                        conv = convolve(input, kernel)
                        if relu:
                                conv_relu=tf.nn.relu(conv)
                                print(conv_relu)
                                return conv_relu
                        print(conv)
                        return conv

def max_pool(input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING,fl=3,fl_y=3):
        return tf.nn.max_pool(input,ksize=[1, k_h, k_w, 1],
                                  strides=[1, s_h, s_w, 1],
                                padding=padding,
                            name=name)

def saturate(input,word_width):

    def func_saturate(input):
        sh=input.shape
        input_flatten=input.flatten()
        output_flatten=input_flatten.copy()
        if word_width==8:
            max_val=float(127.0)
            min_val=float(-128.0)
        elif word_width==16:
            max_val=float(32767.0)
            min_val=float(-32768.0)
        for i in range(len(input_flatten)):
            if input_flatten[i]>max_val:
                output_flatten[i]=max_val
            elif input_flatten[i]<min_val:
                output_flatten[i]=min_val
        return output_flatten.reshape(sh)
    return tf.py_func(func_saturate,[input],tf.float32,stateful=False,name='Sp')

def lp_conv(input, k, s_h, s_w, bw, fl, rs, rate,padding):
        """
        Low precision convolution.      
        """
        if rate==1:
                c = tf.nn.conv2d(input, k, [1, s_h, s_w, 1], padding=padding)
        else:
                c=tf.nn.atrous_conv2d(input,k,rate=rate,padding=padding)
        return lp(c, bw, rs)      # Do the right shift here and bit truncation and satturation here !

def lp(input,bw,rs):

    def func_lp(input):
        sh=input.shape
        input_flatten=input.flatten()
        output_flatten=input_flatten.copy()
        if bw==8:
            max_val=float(127)
            min_val=float(-128)
        elif bw==16:
            max_val=float(32767)
            min_val=float(-32768)
        for i in range(len(input_flatten)):
            input_shift=float(input_flatten[i])/float(pow(2,rs))
            if input_shift>0:
                input_round=input_shift+0.5
            else:
                input_round=input_shift-0.5
            if int(input_round)<=int(min_val):
                output_flatten[i]=min_val
            elif int(input_round)>=int(max_val):
                output_flatten[i]=max_val
            else:
                output_flatten[i]=input_round
        output=output_flatten.reshape(sh)
        return output
    return tf.py_func(func_lp,[input],tf.float32,stateful=False,name='Lp')


def fp(input,bit_frac):

    def func_fp(input):
        sh=input.shape
        fla_input=input.flatten()
        for i in range(len(fla_input)):
            fla_input[i]=float(fla_input[i])/float(pow(2,bit_frac))
        output=fla_input.reshape(sh)
        return output
    return tf.py_func(func_fp,[input],tf.float32,stateful=False,name='Fp')

def qp(input,bit_frac):

    def qp_func(input):
        sh = input.shape
        input_flatten = input.flatten()
        for i in range(len(input_flatten)):
            sign=1
            if input_flatten[i]<0:
                sign=-1
            input_flatten[i]=sign*round(abs(input_flatten[i])*pow(2,bit_frac))
        input_flatten = input_flatten.reshape(sh)
        return input_flatten
    return tf.py_func(qp_func,[input],tf.float32,stateful=False,name='Qp')


