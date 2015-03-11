import cPickle
import numpy as np
import time as time
from PIL import Image
from StringIO import StringIO

# def jpeg_decoder(inputs, jpeg_strings, diff_size, crop_size):
#     for i, jpeg_string in enumerate(jpeg_strings):
#         img = Image.open(StringIO(jpeg_string))
#         csx = np.random.randint(0, diff_size)
#         csy = np.random.randint(0, diff_size)
#             # horizontal reflections of the image
#         flip_horizontal = np.random.randint(0, 2)
#         if flip_horizontal == 1:
#             img = img.transpose(Image.FLIP_LEFT_RIGHT)
#         img = img.crop((csx, csy, csx + crop_size, csy + crop_size))
#         if (img.mode != 'RGB'):
#             img = img.convert('RGB')
#         inputs[i, :] = (np.array(img, dtype=np.float32)).reshape((-1))


def my_unpickle(filename):
    fo = open(filename, 'r')
    contents = cPickle.load(fo)
    fo.close()
    return contents


libmodel = __import__('_ImgWorker')
# fname = '/Users/alexpark/Data/training_batch_0.0'
fname = '/usr/local/data/I1K/macro_batches_256/training_batch_0/training_batch_0.0'
tdata = my_unpickle(fname)
jpeglist = tdata['data']
print len(jpeglist)
imgsize = 256
innersize = 224
innerpixels = innersize*innersize*3
startimg = 0
num_imgs = 1024
unpacked = np.zeros((num_imgs, innerpixels), dtype=np.uint8)
# unpacked32 = np.zeros((num_imgs, innerpixels), dtype=np.float32)

# a = time.time()
# jpeg_decoder(unpacked32, jpeglist[startimg:startimg+num_imgs], imgsize-innersize, innersize)
# print "Python version: ", time.time() - a

a = time.time()
# libmodel.decodeTransformListMT(jpeglist[startimg:startimg+num_imgs], unpacked, imgsize, innersize, False, True, 5)
# {"jpglist", "tgt", "orig_size", "crop_size", "center", "rgb", "nthreads", NULL};
libmodel.decodeTransformListMT(jpglist=jpeglist[startimg:startimg+num_imgs], tgt=unpacked, orig_size=imgsize, crop_size=innersize, center=False, rgb=True, nthreads=5)
print "C Module version: ", time.time() - a


data = unpacked[5,:].reshape((3, innersize, innersize)).transpose(1,2,0)[:,:,[0,1,2]]
im = Image.fromarray(data)
im.save('out.png')
