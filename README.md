Multithreaded decoding and transforms of jpg strings


Installation
------------

    pip install .

Dependencies
------------
* boost C++ libraries
* numpy
* PIL (for batch decoding)

Synopsis
--------

The extension module `_ImWorker.so` provides a method:

    decodeTransformListMT(jpglist=jpglist, tgt=tgt, orig_size=orig_size,
                          crop_size=crop_size, center=center, rgb=rgb,
                          flip=flip, nthreads=nthreads, calcmean=docalcmean)

Parameters
----------

* `jpglist` : list
  Contains the list of coded jpg strings
* `tgt` : numpy array
  Must be pre-allocated to hold the decoded array data
  * if `calcmean` is True:
    * `tgt` will hold the mean of the images in `jpglist` upon completion
    * `tgt` must be KxKxC, where K is `orig_size` and C is 3 when RGB is
      True and C is 1 otherwise.
  * if calcmean is False:
    * `tgt` will hold the decoded images in the `jpglist` upon completion
    * `tgt` must be N x (KxKxC), where K and C are as above and N is
      the number of images in jpglist.
  * in both cases, `tgt` must be of type numpy.uint8
* `orig_size` : int
  Input dimension for decoded image.  Assumes that the coded jpg
  images have already been pre-processed to have uniform size.
  If images are 256x256 for example, `orig_size` is 256.
* `crop_size` : int
  Desired output dimension for the decoded/cropped image.
  Only applies when `calcmean` is False
* `center` : bool, optional
  Whether to take just the center crop of every image when doing decode.
  Default is False.
* `flip` : bool, optional
  Whether to randomly flip the images horizontally.  Default is False.
* `rgb` : bool, optional
  Whether to decode the images into RGB color space or grayscale.  Default is
  True.
* `nthreads` : int, optional
  How many threads to use for decoding.  Default is 1.
* `calcmean` : bool, optional
  Whether to calculate the mean across all images and store in `tgt` or just
  do a decode and transform of images and store each individual decoded image
  as rows in `tgt`.  Default is False
