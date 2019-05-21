"""
Reference Model
Copyright (c) 2018 FABU America 
Author: Abinash Mohanty
"""

import argparse

def parse_args():
	"""
	Parse input arguments.
	"""
	parser = argparse.ArgumentParser(description='RefModel for FABU America')
	parser.add_argument('--network', dest='network_name', help='Network to use [vgg16]',default='pvanet_8bit_ob')
	parser.add_argument('--hw_sim', dest='isHardware', help='Is hardware simulation ?', default=1, type=int)
	parser.add_argument('--image', dest='image', help='Image to test (give only the file name)', default='000.jpg')
	parser.add_argument('--resolution_mode', dest='image_resolution_mode', help='Choose 0,1 or 2. 0: High resolution, 1: mid resolution, 2: low resolution.', default=0, type=int)

	args = parser.parse_args()
	return args
