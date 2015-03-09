import cPickle
import numpy as np
#load the batch_file
import time as time

def my_unpickle(filename):
    fo = open(filename, 'r')
    contents = cPickle.load(fo)
    fo.close()
    return contents
libmodel = __import__('_' + 'ImgWorker')
fname = '/Users/alexpark/Data/training_batch_0.0'
tdata = my_unpickle(fname)
jpeglist = tdata['data']

imgsize = 256
innersize = 256
innerpixels = innersize*innersize*3
startimg = 0
num_imgs = 256
unpacked = np.zeros((innerpixels, num_imgs), dtype=np.uint8)
a = time.time()
libmodel.decodeTransformListMT(jpeglist[startimg:startimg+num_imgs], unpacked, imgsize, innersize, False, 6)
print time.time() - a
import matplotlib.pyplot as plt
fig, ax = plt.subplots()

data = unpacked[:, 0].reshape((3, innersize, innersize)).transpose(1,2,0)[:,:,[0,1,2]]

# data = unpacked.reshape(-1)[imgsize:].reshape((256,256,3))[:,:,[2,1,0]]
im = ax.imshow(data)
plt.show()
