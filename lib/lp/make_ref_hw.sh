TF_INC=$(python3.7 -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB=$(python3.7 -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')

echo $TF_INC
echo $TF_LIB

g++ -std=c++11 -shared -D_GLIBCXX_USE_CXX11_ABI=0 -o lp.so lp_op.cc -I $TF_INC -fPIC -L $TF_LIB -ltensorflow_framework

