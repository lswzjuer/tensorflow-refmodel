
# # -*- coding:utf-8 -*-

# # change the .caffemodel file based on .protxt file to fit it 
# import sys
# sys.path.append("/home/lsw/soft/caffe/python")
# import caffe
# import numpy as np
# import collections
# from collections import OrderedDict
# caffe.set_mode_cpu
# net0 = caffe.Net('lidar_relu6.prototxt',\
#     'lidar_relu6.caffemodel',caffe.TEST) #TEST/TRAIN
#     #我的python脚本，prototxt,caffemodel文件放在同级目录下了，你的文件路径按需修改

# keys0 = net0.params.keys()
# print net0.params.keys()
# print net0.params.values()
# print len(keys0)
# # for key0 in keys0:   # 输出所有层名，参数
# #     print key0
# #     try:
# #        print net0.params[key0][0].data
# #     except IndexError:
# #         continue
# #     try:
# #         print net0.params[key0][1].data
# #     except IndexError:
# #         continue
# #     try:
# #         print net0.params[key0][2].data
# #     except IndexError:
# #         continue
#     # finally:
#     #     print '\n'

# net1 = caffe.Net('test2.prototxt','test2.caffemodel',caffe.TEST) 
# keys1 = net1.params.keys() 
# for key1 in keys1: 
#     net0.params.setdefault(key1, newparams[key1]) 
#     #net0 net1共有的层参数用net0的，net1独有的新层，层名和参数一起加入net0    
#     #net0的参数另存为新caffemodel 
#     net0.save('test3.caffemodel')



# import caffe_pb2
import sys
sys.path.append("/home/lsw/soft/caffe/python")
import caffe
import numpy as np
import collections
from collections import OrderedDict
caffe.set_mode_cpu

BEFORE_MODIFY_DEPLOY_NET = "lidar_relu6.prototxt"
AFTER_MODIFY_DEPLOY_NET = "lidar_relu6_change.prototxt"
BEFORE_MODIFY_CAFFEMODEL = "lidar_relu6.caffemodel"
AFTER_MODIFY_CAFFEMODEL = "lidar_relu6_change.caffemodel"

#根据prototxt修改caffemodel
net = caffe.Net(AFTER_MODIFY_DEPLOY_NET, BEFORE_MODIFY_CAFFEMODEL,caffe.TEST)
net.save(AFTER_MODIFY_CAFFEMODEL )

# print("加载修改前后的caffemodel")
# model = caffe_pb2.NetParameter()
# f=open(BEFORE_MODIFY_CAFFEMODEL, 'rb')
# model.ParseFromString(f.read())
# f.close()
# layers_name = [layer.name for layer in model.layers]
# print("修改前",layers_name)
# model2 = caffe_pb2.NetParameter()
# f=open(AFTER_MODIFY_CAFFEMODEL , 'rb')
# model2.ParseFromString(f.read())
# f.close()
# layers2_name = [layer.name for layer in model2.layer]
# print("修改后",layers2_name)

