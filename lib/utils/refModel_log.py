"""
Reference Model
Copyright (c) 2018 FABU America 
Author: Abinash Mohanty
"""

from config import cfg

def print_msg(msg, level):
	if level >= cfg.DEBUG_LEVEL:
		print ('[Ref Model Log] ' + msg)
