#!/usr/bin/env bash
TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
#TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')

cd roi_pooling_layer
g++ -std=c++11 -shared -D_GLIBCXX_USE_CXX11_ABI=0 -o roi_pooling.so roi_pooling_op.cc \
	-I $TF_INC -fPIC # -L $TF_LIB -ltensorflow_framework
cd ..
cd lp
g++ -std=c++11 -shared -D_GLIBCXX_USE_CXX11_ABI=0 -o lp.so lp_op.cc \
	-I $TF_INC -fPIC # -L $TF_LIB -ltensorflow_framework
cd .. 
