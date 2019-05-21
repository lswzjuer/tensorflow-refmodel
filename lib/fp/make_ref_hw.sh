

TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
echo $TF_INC
g++ -std=c++11 -shared -D_GLIBCXX_USE_CXX11_ABI=0 -o fp.so fp_op.cc \
	-I $TF_INC -fPIC -L $TF_LIB -ltensorflow_framework

