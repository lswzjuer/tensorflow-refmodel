
import numpy as np
from utils.refModel_log import print_msg

def exprootLookUp(N):
    """ Hardware Lookup table for exponential root """

    print_msg('Getting exponential root using lookup table.',1)

    exproot = np.zeros(N.shape, dtype=N.dtype)
    for i in range(0,N.shape[0]):
        if(N[i] < -1946.0/2048.0):
                exproot[i] = 24.0/64.0
        elif((N[i] >= -1946.0/2048.0) and (N[i] < -1843.0/2048.0)):
                exproot[i] = 25.0/64.0
        elif((N[i] >= -1843.0/2048.0) and (N[i] < -1741.0/2048.0)):
                exproot[i] = 27.0/64.0
        elif((N[i] >= -1741.0/2048.0) and (N[i] < -1638.0/2048.0)):
                exproot[i] = 28.0/64.0
        elif((N[i] >= -1638.0/2048.0) and (N[i] < -1536.0/2048.0)):
                exproot[i] = 29.0/64.0
        elif((N[i] >= -1536.0/2048.0) and (N[i] < -1434.0/2048.0)):
                exproot[i] = 31.0/64.0
        elif((N[i] >= -1434.0/2048.0) and (N[i] < -1331.0/2048.0)):
                exproot[i] = 33.0/64.0
        elif((N[i] >= -1331.0/2048.0) and (N[i] < -1229.0/2048.0)):
                exproot[i] = 34.0/64.0
        elif((N[i] >= -1229.0/2048.0) and (N[i] < -1126.0/2048.0)):
                exproot[i] = 36.0/64.0
        elif((N[i] >= -1126.0/2048.0) and (N[i] < -1024.0/2048.0)):
                exproot[i] = 38.0/64.0
        elif((N[i] >= -1024.0/2048.0) and (N[i] < -922.0/2048.0)):
                exproot[i] = 40.0/64.0
        elif((N[i] >= -922.0/2048.0) and (N[i] < -819.0/2048.0)):
                exproot[i] = 42.0/64.0
        elif((N[i] >= -819.0/2048.0) and (N[i] < -717.0/2048.0)):
                exproot[i] = 44.0/64.0
        elif((N[i] >= -717.0/2048.0) and (N[i] < -614.0/2048.0)):
                exproot[i] = 46.0/64.0
        elif((N[i] >= -614.0/2048.0) and (N[i] < -512.0/2048.0)):
                exproot[i] = 49.0/64.0
        elif((N[i] >= -512.0/2048.0) and (N[i] < -410.0/2048.0)):
                exproot[i] = 51.0/64.0
        elif((N[i] >= -410.0/2048.0) and (N[i] < -307.0/2048.0)):
                exproot[i] = 54.0/64.0
        elif((N[i] >= -307.0/2048.0) and (N[i] < -205.0/2048.0)):
                exproot[i] = 56.0/64.0
        elif((N[i] >= -205.0/2048.0) and (N[i] < -102.0/2048.0)):
                exproot[i] = 59.0/64.0
        elif((N[i] >= -102.0/2048.0) and (N[i] < 0.0/2048.0)):
                exproot[i] = 62.0/64.0
        elif(N[i] == 0.0/2048.0):
                exproot[i] = 64.0/64.0
        elif((N[i] > 0.0/2048.0) and (N[i] < 102.0/2048.0)):
                exproot[i] = 66.0/64.0
        elif((N[i] >= 102.0/2048.0) and (N[i] < 205.0/2048.0)):
                exproot[i] = 69.0/64.0
        elif((N[i] >= 205.0/2048.0) and (N[i] < 307.0/2048.0)):
                exproot[i] = 73.0/64.0
        elif((N[i] >= 307.0/2048.0) and (N[i] < 410.0/2048.0)):
                exproot[i] = 76.0/64.0
        elif((N[i] >= 410.0/2048.0) and (N[i] < 512.0/2048.0)):
                exproot[i] = 80.0/64.0
        elif((N[i] >= 512.0/2048.0) and (N[i] < 614.0/2048.0)):
                exproot[i] = 84.0/64.0
        elif((N[i] >= 614.0/2048.0) and (N[i] < 717.0/2048.0)):
                exproot[i] = 89.0/64.0
        elif((N[i] >= 717.0/2048.0) and (N[i] < 819.0/2048.0)):
                exproot[i] = 93.0/64.0
        elif((N[i] >= 819.0/2048.0) and (N[i] < 922.0/2048.0)):
                exproot[i] = 98.0/64.0
        elif((N[i] >= 922.0/2048.0) and (N[i] < 1024.0/2048.0)):
                exproot[i] = 103.0/64.0
        elif((N[i] >= 1024.0/2048.0) and (N[i] < 1126.0/2048.0)):
                exproot[i] = 108.0/64.0
        elif((N[i] >= 1126.0/2048.0) and (N[i] < 1229.0/2048.0)):
                exproot[i] = 114.0/64.0
        elif((N[i] >= 1229.0/2048.0) and (N[i] < 1331.0/2048.0)):
                exproot[i] = 120.0/64.0
        elif((N[i] >= 1331.0/2048.0) and (N[i] < 1434.0/2048.0)):
                exproot[i] = 126.0/64.0
        elif((N[i] >= 1434.0/2048.0) and (N[i] < 1536.0/2048.0)):
                exproot[i] = 132.0/64.0
        elif((N[i] >= 1536.0/2048.0) and (N[i] < 1638.0/2048.0)):
                exproot[i] = 139.0/64.0
        elif((N[i] >= 1638.0/2048.0) and (N[i] < 1741.0/2048.0)):
                exproot[i] = 146.0/64.0
        elif((N[i] >= 1741.0/2048.0) and (N[i] < 1843.0/2048.0)):
                exproot[i] = 154.0/64.0
        elif((N[i] >= 1843.0/2048.0) and (N[i] < 1946.0/2048.0)):
                exproot[i] = 161.0/64.0
        else:
                exproot[i] = 170.0/64.0

#        if (i < 10):
#                print '      N[' + str(i) + '] = ' + str(N[i])
#                print 'exproot[' + str(i) + '] = ' + str(exproot[i])
#                print 'np.exp(N)  = ' + str(np.exp(N[i]))

                
    return exproot
