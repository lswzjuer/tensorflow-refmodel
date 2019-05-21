TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
#TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
echo $TF_INC
#echo $TF_LIB
# g++ -std=c++11 -shared  zero_out.cc -o zero_out.so -I $TF_INC -fPIC  -D_GLIBCXX_USE_CXX11_ABI=0 #-L$TF_LIB -ltensorflow_framework 
g++ -std=c++11 -shared  zero_out.cc -o zero_out.so -fPIC -I$TF_INC -I$TF_INC/external/nsync/public -O2  -D_GLIBCXX_USE_CXX11_ABI=0 #-L$TF_LIB -ltensorflow_framework -O2  -D_GLIBCXX_USE_CXX11_ABI=0
