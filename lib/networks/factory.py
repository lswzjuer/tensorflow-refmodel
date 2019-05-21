"""
Reference Model
Copyright (c) 2018 MobaiTech Inc 
Author: Abinash Mohanty
Revisions: 
1. Abinash: pvanet_ob and pvanet_tl (05/01/2018)
2. Abinash: resNet support (10/24/2018)
3. Abinash: set to easily list all the supported networks (10/24/2018) 
"""

# from .pvanet_ob import pvanet_ob	# TODO
# from .pvanet_tl import pvanet_tl	# TODO
# from .resNet_50 import resnet50	# TODO
# from .resNet_101 import resnet101	# TODO
# from .resNet_152 import resnet152	# TODO
# from .vgg_16 import vgg16 	# TODO
# #from .vgg_19 import vgg19	# TODO
# from .ssdJacintoNetV2 import ssdJacintoNetV2	# TODO
# from .pvanet_roip8 import pvanet_roip8	# TODO 
# #from .pvanet_tl_v1 import pvanet_tl_v1 # TODO
# from .pvanet_8bit_ob import pvanet_8bit_ob
# from .pvanet_8bit_tl import pvanet_8bit_tl
from .lidarnet import LIDAR as lidar
from .lidarnet_change_add import LIDAR as lidar_change

__sets = {}
# __sets['pvanet_ob'] = 1
# __sets['pvanet_tl'] = 1
# __sets['resnet50'] = 1
# __sets['resnet101'] = 1
# __sets['resnet152'] = 1
# __sets['vgg16'] = 1
# __sets['ssdJacintoNetV2'] = 1
# __sets['pvanet_roip8'] = 1
# __sets['pvanet_8bit_ob'] = 1
# __sets['pvanet_8bit_tl'] = 1
__sets['lidar'] = 1
__sets['lidar_change'] = 1
#__sets['pvanet_tl_v1'] = 1	# TODO

def get_network(name, isHardware=False):
	"""
	Get a network instance by name.
	Arguments: 
		name: name of the networks as a string
		isHardware: bool to choose between low precision hardware simulation or floating point software simulation. 		
	Returns: 
		Instance of the network
	"""
	if name == 'pvanet_ob':
		return pvanet_ob(isHardware=isHardware)
	elif name == 'pvanet_8bit_ob':
		return pvanet_8bit_ob(isHardware=isHardware)
	elif name == 'pvanet_tl':	
		return pvanet_tl(isHardware=isHardware)
	elif name == 'pvanet_8bit_tl':
		return pvanet_8bit_tl(isHardware=isHardware)
	elif name == 'resnet50':
		return resnet50(isHardware=isHardware)
	elif name == 'reset101':
		return resnet101(isHardware=isHardware)
	elif name == 'reset152':
		return resnet152(isHardware=isHardware)
	elif name == 'vgg16':
		return vgg16(isHardware=isHardware)
	#elif name == 'vgg19':	#TODO
	#	return vgg19(isHardware=isHardware) #TODO
	#elif name == 'pvanet_tl_v1':	#TODO
	#	return pvanet_tl_v1(isHardware=isHardware)	#TODO
	elif name == 'ssdJacintoNetV2':
		return ssdJacintoNetV2(isHardware=isHardware)
	elif name == 'pvanet_roip8':
		return pvanet_roip8(isHardware=isHardware)
	elif name=='lidar':
		return lidar(isHardware=isHardware)
	elif name=='lidar_change':
		return lidar_change(isHardware=isHardware)
	else:
		raise KeyError('Unknown Network: {}'.format(name))

def list_networks():
	"""
	Returns a list of names of all supported networks.
	"""
	return __sets.keys()
