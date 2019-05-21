"""
Reference Model
Copyright (c) 2018 MobaiTech Inc 
Author: Abinash Mohanty
"""

import os
import os.path as osp
import numpy as np
from easydict import EasyDict as edict

__C = edict()
cfg = __C


__C.DEBUG_LEVEL = 3                                             # 0: All log messages, 1: low level messages, 2: Mid level messages, 3: Only critical messages
__C.DEBUG = True                                                # Enable this for DEBUG logs
__C.DEBUG_PROPOSAL = False                              # Enable this for DEBUG logs
__C.isHardware = True                                   # Enable this for hardware simulation
__C.DATA_DIR = '/global/mobai2/cnn_emu_data/data/'                              # Path to data dir
__C.MODELS_DIR = '/global/mobai2/cnn_emu_data/models_new/'                  # Path to model dir
#__C.MODELS_DIR = '/global/mobai/users/Xghoward/dla_sw_ref_model_cp/verif/cnn_emu/'
#__C.MODELS_DIR = '/global/mobai/users/Xghoward/dla_resnet_merge/verif/cnn_emu/'
#__C.MODELS_DIR = '/global/mobai/users/abinash/DLA2.0/models/'                  # Path to model dir      local testing remove
__C.ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..' ))      # Path to refModel root
__C.OUTPUT_DIR = osp.abspath(osp.join(__C.ROOT_DIR, 'output'))          # Path to the output directory to store results
__C.USE_GPU_NMS = False                                 # Should always be false. No GPU on SPNS

#__C.PIXEL_MEANS = [102.9801, 115.9465, 112.7717]
#__C.PIXEL_MEANS = [128, 128, 128]
__C.PIXEL_MEANS = np.array([[[128, 128, 128]]])
__C.PIXEL_STDS = np.array([[[1.0, 1.0, 1.0]]])


__C.ENABLE_TENSORBOARD = False

__C.TEST = edict()
__C.TEST.NMS = 0.3                                              # IoU threshold in NMS
__C.TEST.BBOX_REG = True                                # Should be high to visualise the output 
__C.TEST.HAS_RPN = True                                 # Should be high (we have RPN in hardware) TODO: Remove it 
__C.TEST.RPN_NMS_THRESH = 0.7                    
__C.TEST.RPN_PRE_NMS_TOP_N = 2000               
__C.TEST.RPN_POST_NMS_TOP_N = 300               
#TODO __C.TEST.RPN_MIN_SIZE = 10                               
__C.TEST.RPN_MIN_SIZE = 0                               



# Scales to use during testing (can list multiple scales)
# Each scale is the pixel size of an image's shortest side
__C.TEST.SCALES = (1200,)

# Max pixel size of the longest side of a scaled input image
__C.TEST.MAX_SIZE = 1056

# Resize test images so that its width and height are multiples of ...
__C.TEST.SCALE_MULTIPLE_OF = 32



# HARDWARE PARAMETERS
__C.WORD_WIDTH = 8 # INCLUDES SIGN BIT

# OB Related Params (CSR FOR HARDWARE)
__C.NUM_STDDEV_OB = 2.5
__C.MAX_NMS_TOP_N_OB = 400

# TL Related Params (CSR FOR HARDWARE)
__C.NUM_STDDEV_TL = 2.5 
__C.MAX_NMS_TOP_N_TL = 20  #TODO

#__C.BY_N_APPROX =  0.7651      # OB 1080
#__C.BY_N_APPROX =  1.0132      # OB 1920
#__C.BY_N_APPROX =  0.88658     # TL 1920
