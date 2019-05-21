import numpy as np

def nearest_neighbor_layer(i,s=2):
    
    print("NN DEBUG, input shape: %s\n" %(str(i.shape)))

    i = np.transpose(i)
    o = i.repeat(s, axis=1).repeat(s, axis=2)
    o = np.transpose(o)
    o = np.expand_dims(o,axis=0)
    blob = o.astype(np.float32, copy=False)

    print("NN DEBUG, output shape: %s\n" %(str(blob.shape)))
   
    return blob

