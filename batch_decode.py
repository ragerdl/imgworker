# ----------------------------------------------------------------------------
# Copyright 2015 Nervana Systems Inc.  All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------

import cPickle
import numpy as np
import time as time
import os
from PIL import Image
from StringIO import StringIO

def my_unpickle(filename):
    fo = open(filename, 'r')
    contents = cPickle.load(fo)
    fo.close()
    return contents

libmodel = __import__('_ImgWorker')
PATH = '/usr/local/data/I1K/macro_batches_256/training_batch_0/'
fname = os.path.join(PATH, 'training_batch_0.0')
tdata = my_unpickle(fname)
jpeglist = tdata['data']
print len(jpeglist)
imgsize = 256
innersize = 224
innerpixels = innersize*innersize*3
startimg = 0
num_imgs = 1024
unpacked = np.zeros((num_imgs, innerpixels), dtype=np.uint8)

t1 = time.time()
libmodel.decodeTransformListMT(jpglist=jpeglist[startimg:startimg+num_imgs],
							   tgt=unpacked, orig_size=imgsize,
							   crop_size=innersize, center=False,
							   rgb=True, flip=False, nthreads=5,
							   calcmean=False)
print("C Module version decode time: ", time.time() - t1)

unpackedM = np.zeros((imgsize, imgsize, 3), dtype=np.uint8)

libmodel.decodeTransformListMT(jpglist=jpeglist[startimg:startimg+num_imgs],
							   tgt=unpackedM, orig_size=imgsize,
							   crop_size=innersize,
							   calcmean=True)

# Now save out an image from the packed array to confirm things are working
data = unpacked[5,:].reshape(
					(3, innersize, innersize)).transpose(1,2,0)[:,:,[0,1,2]]
im = Image.fromarray(data)
im.save('out.png')
