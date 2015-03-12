/*
 * ---------------------------------------------------------------------------
 * Copyright 2015 Nervana Systems Inc.  All rights reserved.
 * ---------------------------------------------------------------------------
 */

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <assert.h>
#include "imgworker.h"

extern "C" void init_ImgWorker();
PyObject* decodeTransformListMT(PyObject *self, PyObject *args,
                                PyObject *keywds);

boost::thread_group g;

static PyMethodDef _ImgWorkerMethods[] = {
    { "decodeTransformListMT", (PyCFunction) decodeTransformListMT,
                               METH_VARARGS | METH_KEYWORDS, NULL},
    { NULL, NULL }};


PyObject* decodeTransformListMT(PyObject *self, PyObject *args,
                                PyObject *keywds) {
    PyListObject* pyJpegStrings;
    PyArrayObject* pyTarget;
    int img_size, inner_size;
    int center=0;
    int flip=0;
    int rgb=1;
    int nthreads=1;
    int calcMean=0;

    const char *kwlist[] = {"jpglist", "tgt", "orig_size", "crop_size",
                            "center", "flip", "rgb", "nthreads", "calcmean",
                            NULL};

    if (!PyArg_ParseTupleAndKeywords(args, keywds, "O!O!ii|iiiii",
        const_cast<char **> (kwlist),
        &PyList_Type, &pyJpegStrings, &PyArray_Type, &pyTarget,
        &img_size, &inner_size, &center, &flip, &rgb, &nthreads, &calcMean)) {
        return NULL;
    }
    assert(pyTarget != NULL);
    assert(PyArray_ISONESEGMENT(pyTarget));
    assert(PyArray_CHKFLAGS(pyTarget, NPY_ARRAY_C_CONTIGUOUS));
    int num_imgs = PyList_GET_SIZE(pyJpegStrings);
    int num_imgs_per_thread = (num_imgs + nthreads - 1) / nthreads;

    WorkerParams *wp = new WorkerParams(
        (PyObject*) pyJpegStrings, pyTarget, img_size, inner_size,
        center, rgb, flip, num_imgs);

    ImgWorker *workers[nthreads];

    for (int t = 0; t < nthreads; ++t) {
        int start_img = t * num_imgs_per_thread;
        int end_img = std::min(num_imgs, (t+1) * num_imgs_per_thread);
        workers[t] = new ImgWorker(wp, start_img, end_img);
        if (calcMean) {
            g.create_thread(boost::bind( &ImgWorker::accumVals, workers[t]));
        }
        else {
            g.create_thread(boost::bind( &ImgWorker::decodeList, workers[t]));
        }
    }
    g.join_all();

    if (calcMean) {
        int *tot_sum = new int[wp->_npixels_in]();
        int tot_bsize = 0;
        unsigned char *tgt = (unsigned char *) PyArray_DATA(pyTarget);
        for (int t = 0; t < nthreads; ++t) {
            tot_bsize += workers[t]->_bsize;
            for (int64 i = 0; i < wp->_npixels_in; i++) {
                tot_sum[i] += workers[t]->_jpgbuf[i] * workers[t]->_bsize;
            }
        }
        printf("Total num imgs = %d\n", tot_bsize);
        for (int64 i = 0; i < wp->_npixels_in; i++) {
            tgt[i] = tot_sum[i] / tot_bsize;
            if (i<10) {
                printf("\ttgt val = %c sum val = %d\n", tgt[i], tot_sum[i]);
            }
        }
        delete[] tot_sum;
    }

    for (int t = 0; t < nthreads; ++t)
        delete(workers[t]);

    return Py_BuildValue("i", 0);
}

WorkerParams::WorkerParams(PyObject* pyList, PyArrayObject *pyTarget,
                           int img_size, int inner_size, bool center,
                           bool rgb, bool flip, int num_imgs)
    : _pyList(pyList), _pyTgt(pyTarget), _img_size(img_size),
      _inner_size(inner_size), _center(center), _flip(flip), _rgb(rgb),
      _num_imgs(num_imgs) {

    _channels = _rgb ? 3 : 1;
    _npixels_in = img_size * img_size * _channels;
    _npixels_out = inner_size * inner_size * _channels;
    _inner_pixels = _inner_size * _inner_size;
    _num_rows = PyArray_DIM(_pyTgt, 0);
    _num_cols = PyArray_DIM(_pyTgt, 1);
}
WorkerParams::~WorkerParams() {
}

ImgWorker::ImgWorker(WorkerParams *wp, int start_img, int end_img)
        : _wp(wp), _start_img(start_img), _end_img(end_img), _jpgbuf(0) {

    _jpgbuf = new unsigned char[_wp->_npixels_in];
    _tgt = (unsigned char *) PyArray_DATA(_wp->_pyTgt);
    _rseed = time(0);
    _bsize = (int) (_end_img - _start_img);
    _npixels_in = _wp->_npixels_in;
}

ImgWorker::~ImgWorker(){
    delete[] _jpgbuf;
}

void init_ImgWorker() {
    (void) Py_InitModule("_ImgWorker", _ImgWorkerMethods);
    import_array();
}

void ImgWorker::accumVals() {
    int ww, hh;
    int *accum = new int[_npixels_in]();
    for (int64 idx = _start_img; idx < _end_img; idx++) {
        PyObject* pySrc = PyList_GET_ITEM(_wp->_pyList, idx);
        unsigned char* src = (unsigned char *) PyString_AsString(pySrc);
        size_t src_len = PyString_GET_SIZE(pySrc);
        decodeJpeg(src, src_len, ww, hh);
        for (int64 i = 0; i < _npixels_in; i++) {
            accum[i] += _jpgbuf[i];
        }
    }
    for (int64 i = 0; i < _npixels_in; i++) {
        _jpgbuf[i] = accum[i] / _bsize;
    }
    delete[] accum;
}

void ImgWorker::decodeList() {
    int ww, hh;
    for (int64 idx = _start_img; idx < _end_img; idx++) {
        PyObject* pySrc = PyList_GET_ITEM(_wp->_pyList, idx);
        unsigned char* src = (unsigned char *) PyString_AsString(pySrc);
        size_t src_len = PyString_GET_SIZE(pySrc);
        decodeJpeg(src, src_len, ww, hh);
        crop_and_copy(idx, ww, hh, _wp->_flip, -1, -1);
    }
}

void ImgWorker::decodeJpeg(unsigned char* src, size_t src_len,
                           int& width, int& height) {
    struct jpeg_decompress_struct cinf;
    struct jpeg_error_mgr jerr;
    cinf.err = jpeg_std_error(&jerr);
    jpeg_create_decompress(&cinf);
    jpeg_mem_src(&cinf, src, src_len);
    assert(jpeg_read_header(&cinf, TRUE));
    cinf.out_color_space = _wp->_rgb ? JCS_RGB : JCS_GRAYSCALE;
    assert(jpeg_start_decompress(&cinf));
    assert(cinf.num_components == 3 || cinf.num_components == 1);
    width = cinf.image_width;
    height = cinf.image_height;

    assert(_npixels_in >= width * height * _wp->_channels);

    while (cinf.output_scanline < cinf.output_height) {
        int lw = width * cinf.out_color_components * cinf.output_scanline;
        JSAMPROW tmp = &_jpgbuf[lw];
        assert(jpeg_read_scanlines(&cinf, &tmp, 1) > 0);
    }
    assert(jpeg_finish_decompress(&cinf));
    jpeg_destroy_decompress(&cinf);
}

void ImgWorker::crop_and_copy(int64 i, int64 src_w, int64 src_h, bool flip,
                              int64 crop_start_x, int64 crop_start_y) {
    int64 insz = _wp->_inner_size;
    int64 cols = _wp->_num_cols;
    bool cent = _wp->_center;
    unsigned int chan = _wp->_channels;

    if (crop_start_x < 0) {
        const int64 border = src_w - insz;
        crop_start_x = cent ? (border / 2) : (rand_r(&_rseed) % (border + 1));
    }
    if (crop_start_y < 0) {
        const int64 border = src_h - insz;
        crop_start_y = cent ? (border / 2) : (rand_r(&_rseed) % (border + 1));
    }
    for (int64 c = 0; c < chan; ++c) {
        for (int64 y = crop_start_y; y < crop_start_y + insz; ++y) {
            for (int64 x = crop_start_x; x < crop_start_x + insz; ++x) {
                assert((y >= 0 && y < src_h && x >= 0 && x < src_w));
                int64 tgtrow = c * _wp->_inner_pixels +
                               (y - crop_start_y) * insz +
                               (flip ? (insz - 1 - x + crop_start_x)
                                     : (x - crop_start_x));
                _tgt[i * cols + tgtrow] = _jpgbuf[chan * (y * src_w + x) + c];
            }
        }
    }
}
