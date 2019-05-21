"""
Reference Model
Copyright (c) 2019 FABU America  
Author: Abinash Mohanty
"""

import refModel as refModel	# Import refModel

if __name__ == '__main__':

	bb = refModel.get_bias('pvanet_8bit_ob')
	for i in range(len(bb)):
		print len(bb[i])
	print bb[1][0]
	"""
	data = refModel.get_data_for_validation('pvanet_8bit_ob','000.jpg',0,1,2)
	print len(data)
	print data[0][0][0].shape	
	print data[0][1][0][0,0,0,:]	
	"""	
