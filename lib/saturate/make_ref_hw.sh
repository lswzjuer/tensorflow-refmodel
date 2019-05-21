TF_CFLAGS=($(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_CFLAGS=($(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
g++ -std=c++11 -shared saturate_op.cc -o sp.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2

# TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
# #TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
# echo $TF_INC
# g++ -std=c++11 -shared -D_GLIBCXX_USE_CXX11_ABI=0 -o sp.so saturate_op.cc -I $TF_INC -fPIC #-L $TF_LIB -ltensorflow_framework

