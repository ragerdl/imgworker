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

import numpy as np
# from imgworker._ImgWorker import decodeTransformListMT
iw = __import__('_ImgWorker')
# import imgworker

def check_tgt(tgt, otype, oshape):
    assert tgt.dtype == otype, "tgt dtype is incorrect {} vs {}".format(
        tgt.dtype, otype)
    assert tgt.shape == oshape, "tgt shape is incorrect {} vs {}".format(
        tgt.shape, oshape)


def decode_list(jpglist, tgt, orig_size, crop_size,
                center=False, flip=False, rgb=True, nthreads=5):

    channels = 3 if rgb else 1
    out_shape = (len(jpglist), channels * crop_size * crop_size)

    if tgt is None:
        tgt = np.zeros(out_shape, dtype=np.uint8)
    else:
        check_tgt(tgt, np.uint8, out_shape)

    iw.decodeTransformListMT(jpglist=jpglist, tgt=tgt, orig_size=orig_size,
                             crop_size=crop_size, center=center, flip=flip, 
                             rgb=rgb, nthreads=nthreads, calcmean=False)

    return tgt


def calc_batch_mean(jpglist, tgt, orig_size, rgb=True, nthreads=5):
    channels = 3 if rgb else 1
    out_shape = (orig_size, orig_size, channels)

    if tgt is None:
        tgt = np.zeros(out_shape, dtype=np.uint8)
    else:
        check_tgt(tgt, np.uint8, out_shape)

    iw.decodeTransformListMT(jpglist=jpglist, tgt=tgt, orig_size=orig_size,
                             crop_size=orig_size, center=True, flip=False,
                             rgb=rgb, nthreads=nthreads, calcmean=True)
