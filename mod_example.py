import cPickle
import numpy as np
import time as time
import os
from PIL import Image
from StringIO import StringIO
import imgworker as iw
import sys
def my_unpickle(filename):
    fo = open(filename, 'r')
    contents = cPickle.load(fo)
    fo.close()
    return contents

# labels = [item for sublist in labels for item in sublist]
# labels = np.asarray(labels, dtype='float32')
# a = np.empty((1,3072), dtype=np.float32)
# a[:] = labels
numthreads = int(sys.argv[1])
imgsize = 256
innersize = 224
innerpixels = innersize*innersize*3
startimg = 0
num_imgs = 2048
unpacked = np.zeros((num_imgs, innerpixels), dtype=np.uint8)
t0 = time.time()
for i in range(24):
	t1 = time.time()
	PATH = '/usr/local/data/I1K/macro_batches_256/training_batch_{:d}/'.format(i)
	fname = os.path.join(PATH, 'training_batch_{:d}.0'.format(i))
	tdata = my_unpickle(fname)
	jpeglist = tdata['data']
	# labels = tdata['labels']
	iw.decode_list(jpglist=jpeglist[startimg:startimg+num_imgs],
	               tgt=unpacked, orig_size=imgsize,
	               crop_size=innersize, nthreads=numthreads)
	print("\tC Module version decode time: ", time.time() - t1)
print("Total Time:", time.time() - t0)

# Now save out an image from the packed array to confirm things are working
data = unpacked[5,:].reshape(
					(3, innersize, innersize)).transpose(1,2,0)[:,:,[0,1,2]]
im = Image.fromarray(data)
im.save('out.png')
