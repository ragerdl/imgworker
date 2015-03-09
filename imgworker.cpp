//#include <boost/python.hpp>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <arrayobject.h>
#include <assert.h>
#include "imgworker.h"

extern "C" void init_ImgWorker();
PyObject* decodeTransformListMT(PyObject *self, PyObject *args);
void callWorkersDecode(ImgWorker *iw);

boost::thread_group g;

static PyMethodDef _ImgWorkerMethods[] = {{ "decodeTransformListMT",         decodeTransformListMT,             METH_VARARGS },
                                         { NULL, NULL }
};

ImgWorker::ImgWorker(PyObject* pyList, PyArrayObject *pyTarget, int start_img, int end_img, int img_size, int inner_size, bool center, int num_imgs)
: _pyList(pyList), _pyTgt(pyTarget), _start_img(start_img), _end_img(end_img),
  _img_size(img_size), _inner_size(inner_size), _center(center),  _jpgbuf(0), _num_imgs(num_imgs) {

    _npixels_in = img_size * img_size * 3;
    _npixels_out = inner_size * inner_size * 3;
    _jpgbuf = (unsigned char*)malloc(_npixels_in);
    _inner_pixels = _inner_size * _inner_size;
    _tgt = (unsigned char *) PyArray_DATA(_pyTgt);
    _num_rows = PyArray_DIM(_pyTgt, 0);
    _num_cols = PyArray_DIM(_pyTgt, 1);
    _rseed = time(0);
}

ImgWorker::~ImgWorker(){
    free(_jpgbuf);
}

void init_ImgWorker() {
    (void) Py_InitModule("_ImgWorker", _ImgWorkerMethods);
    import_array();
}
void callWorkersDecode(ImgWorker *iw) {
    iw->decodeList();
}
PyObject* decodeTransformListMT(PyObject *self, PyObject *args) {
    PyListObject* pyJpegStrings;
    PyArrayObject* pyTarget;
    int img_size, inner_size, center, nthreads;
    if (!PyArg_ParseTuple(args, "O!O!iiii",
        &PyList_Type, &pyJpegStrings,
        &PyArray_Type, &pyTarget,
        &img_size,
        &inner_size,
        &center,
        &nthreads)) {
        return NULL;
    }

    // Do the checks here:
    assert(pyTarget != NULL);
    assert(PyArray_ISONESEGMENT(pyTarget));
    assert(PyArray_CHKFLAGS(pyTarget, NPY_ARRAY_C_CONTIGUOUS));
    assert(PyArray_NDIM(pyTarget)==2);
    int num_imgs = PyList_GET_SIZE(pyJpegStrings);
    int num_imgs_per_thread = (num_imgs + nthreads - 1) / nthreads;

    ImgWorker *workers[nthreads];

    for (int t = 0; t < nthreads; ++t) {
        int start_img = t * num_imgs_per_thread;
        int end_img = std::min(num_imgs, (t+1) * num_imgs_per_thread);
        workers[t] = new ImgWorker((PyObject*) pyJpegStrings, pyTarget, start_img, end_img, img_size, inner_size, center, num_imgs);
        g.create_thread(boost::bind( &ImgWorker::decodeList, workers[t]));
    }

    g.join_all();

    for (int t = 0; t < nthreads; ++t)
        delete(workers[t]);

    return Py_BuildValue("i", 0);
}

void ImgWorker::decodeList() {
    int m = PyArray_DIM(_pyTgt, 0);
    int n = PyArray_DIM(_pyTgt, 1);
    int ww, hh;
    for (int64 idx = _start_img; idx < _end_img; idx++) {
        PyObject* pySrc = PyList_GET_ITEM(_pyList, idx);
        unsigned char* src = (unsigned char *) PyString_AsString(pySrc);
        size_t src_len = PyString_GET_SIZE(pySrc);
        decodeJpeg(src, src_len, ww, hh);
        crop(idx, ww, hh, false, -1, -1);
    }
}

void ImgWorker::decodeJpeg(unsigned char* src, size_t src_len, int& width, int& height) {
    struct jpeg_decompress_struct cinf;
    struct jpeg_error_mgr jerr;
    cinf.err = jpeg_std_error(&jerr);
    jpeg_create_decompress(&cinf);
    jpeg_mem_src(&cinf, src, src_len);
    assert(jpeg_read_header(&cinf, TRUE));
    cinf.out_color_space = JCS_RGB;
    assert(jpeg_start_decompress(&cinf));
    assert(cinf.num_components == 3 || cinf.num_components == 1);
    width = cinf.image_width;
    height = cinf.image_height;

    assert(_npixels_in >= width * height * 3);

    while (cinf.output_scanline < cinf.output_height) {
        JSAMPROW tmp = &_jpgbuf[width * cinf.out_color_components * cinf.output_scanline];
        assert(jpeg_read_scanlines(&cinf, &tmp, 1) > 0);
    }
    assert(jpeg_finish_decompress(&cinf));
    jpeg_destroy_decompress(&cinf);
}

void ImgWorker::crop(int64 i, int64 src_width, int64 src_height, bool flip, int64 crop_start_x, int64 crop_start_y) {
    if (crop_start_x < 0) {
        const int64 border = src_width - _inner_size;
        crop_start_x = _center ? (border / 2) : (rand_r(&_rseed) % (border + 1));
    }
    if (crop_start_y < 0) {
        const int64 border = src_height - _inner_size;
        crop_start_y = _center ? (border / 2) : (rand_r(&_rseed) % (border + 1));
    }
    for (int64 c = 0; c < 3; ++c) {
        for (int64 y = crop_start_y; y < crop_start_y + _inner_size; ++y) {
            for (int64 x = crop_start_x; x < crop_start_x + _inner_size; ++x) {
                assert((y >= 0 && y < src_height && x >= 0 && x < src_width));
                int64 tgtrow = c * _inner_pixels + (y - crop_start_y) * _inner_size
                                                    + (flip ? (_inner_size - 1 - x + crop_start_x)
                                                        : (x - crop_start_x));
                _tgt[tgtrow * _num_cols + i] = _jpgbuf[3 * (y * src_width + x) + c];
            }
        }
    }
}
