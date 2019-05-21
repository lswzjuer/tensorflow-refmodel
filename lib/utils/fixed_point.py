"""
Reference Model
Copyright (c) 2018 MobaiTech Inc 
Author: Abinash Mohanty
"""

# from config import cfg
import numpy as np
import math

def getFixedPoint(num, totalBits, fractionBits, mode=1):
	"""
	Returns a fixed point value of num.
	Arguments:
		num - input number
		totalBits - total number of bits
		fractionBits - number of fractional bits
		mode - 0: returns str, 1: returns float, 2: retuns int
	Returns:
		fixedpoint representation of the number as an int32
	"""
	if isinstance(num, basestring):
		num = float(num)
	sign = 1
	if num < 0:
		sign = -1
	if mode == 1:
		return sign*round(abs(num)*pow(2,fractionBits))/pow(2,fractionBits)
	elif mode==2:
		return sign*round(abs(num)*pow(2,fractionBits))
	return str(sign*round(abs(num)*pow(2,fractionBits))/pow(2,fractionBits))


def convert_to_fp(num):
	"""
	Convert a numpy ndarray to fixed point
	Arguemnts: 
		num - input array
		totalBits - total number of bits
		fractionBits - number of fractional bits
		mode - 0: returns str, 1: returns float, 2: retuns int REMOVE
	Returns:
		fixedpoint representation of the array as an int16/int8 based on configuration parameter WORD_WIDTH	
	"""
	sh = num.shape
	num = num.flatten()
	if WORD_WIDTH == 8:
		num = num.astype(np.int8)	
	else: 
		num = num.astype(np.int16) 
	num = num.reshape(sh)	
	return str(num)

def convert_to_float_py(num, fractionBits):

	sh = num.shape
	num = num.flatten()
	for i in range(len(num)):
		num[i] = float(num[i])/float(pow(2,fractionBits))
	num = num.reshape(sh)
	return num

if __name__=="__main__":
	WORD_WIDTH=8
	x1=[2131.2342]
	x2=[-101.9876]
	w1=[0.1023] 
	w2=[0.0512]
	X1=convert_to_fp(np.asarray(w2))
	x11=getFixedPoint(x1[0],8,2,2)
	x22=getFixedPoint(x2[0],8,2,2)
	w11=getFixedPoint(w1[0],8,9,2)
	w22=getFixedPoint(w2[0],8,9,2)

	midle_results=x11*w11+x22*w22
	right_results=math.floor(midle_results/pow(2,5))
	real_value=math.floor(right_results/pow(2,6))
	real_value1=convert_to_float_py(np.asarray(right_results),6)
	
	print(X1)
	print(x11)
	print(x22)
	print(w11)
	print(w22)
	print(midle_results)
	print(right_results)
	print(real_value)
	print(real_value1)
