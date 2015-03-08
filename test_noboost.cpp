//#include <boost/python.hpp>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <arrayobject.h>
#include <assert.h>
#include <cv.h>
#include <highgui.h>
#include <jpeglib.h>

typedef long long int int64;
typedef unsigned short int uint8;

extern "C" void init_ImWorker();
PyObject* decodeTransformListMT(PyObject *self, PyObject *args);
void decodeImgToArray(PyObject *pyJpegStrings, PyArrayObject *pyTarget, int start_img, int end_img, int img_size, int inner_size);
void copy_image_to_column(unsigned char *src, int srclen, unsigned char *tgt, int destcol, int tgtncol);

void decodeJpeg(unsigned char* src, size_t src_len, unsigned char* dest, int destsize, int& width, int& height);

using namespace cv;

static PyMethodDef _ImWorkerMethods[] = {{ "decodeTransformListMT",         decodeTransformListMT,             METH_VARARGS },
                                         { NULL, NULL }
};

void init_ImWorker() {
    (void) Py_InitModule("_ImWorker", _ImWorkerMethods);
    import_array();
}

PyObject* decodeTransformListMT(PyObject *self, PyObject *args) {
    PyListObject* pyJpegStrings;
    PyArrayObject* pyTarget;
    int img_size, inner_size, test;
    if (!PyArg_ParseTuple(args, "O!O!iii",
        &PyList_Type, &pyJpegStrings,
        &PyArray_Type, &pyTarget,
        &img_size,
        &inner_size,
        &test)) {
        return NULL;
    }

    // Do the checks here:
    assert(pyTarget != NULL);
    assert(PyArray_ISONESEGMENT(pyTarget));
    assert(PyArray_CHKFLAGS(pyTarget, NPY_ARRAY_C_CONTIGUOUS));
    assert(PyArray_NDIM(pyTarget)==2);

    // Thread* threads[NUM_JPEG_DECODER_THREADS];
    int num_imgs = PyList_GET_SIZE(pyJpegStrings);
    decodeImgToArray((PyObject*) pyJpegStrings, pyTarget, 0, 2, img_size, inner_size);
    // int num_imgs_per_thread = DIVUP(num_imgs, NUM_JPEG_DECODER_THREADS);
    // Matrix& dstMatrix = *new Matrix(pyTarget);
    // for (int t = 0; t < NUM_JPEG_DECODER_THREADS; ++t) {
    //     int start_img = t * num_imgs_per_thread;
    //     int end_img = min(num_imgs, (t+1) * num_imgs_per_thread);

    //     threads[t] = new DecoderThread((PyObject*)pyJpegStrings, dstMatrix, start_img, end_img, img_size, inner_size, test);
    //     threads[t]->start();
    // }

    // for (int t = 0; t < NUM_JPEG_DECODER_THREADS; ++t) {
    //     threads[t]->join();
    //     delete threads[t];
    // }
    // assert(dstMatrix.isView());
    // delete &dstMatrix;
    return Py_BuildValue("i", 0);
}

// return double(rand_r(&_rseed)) / (int64(RAND_MAX) + 1);

void decodeImgToArray(PyObject *pyJpegStrings, PyArrayObject *pyTarget, int start_img, int end_img, int img_size, int inner_size) {
    int m = PyArray_DIM(pyTarget, 0);
    int n = PyArray_DIM(pyTarget, 1);
    int ww, hh;

    int npixels_in = img_size * img_size * 3;
    int npixels_out = inner_size * inner_size * 3;
    unsigned char *tgt = (unsigned char *) PyArray_DATA(pyTarget);
    unsigned char* jpgbuf = (unsigned char*)malloc(npixels_in);

    for (int idx = start_img; idx < end_img; idx++) {
        PyObject* pySrc = PyList_GET_ITEM(pyJpegStrings, idx);
        unsigned char* src = (unsigned char *) PyString_AsString(pySrc);
        size_t src_len = PyString_GET_SIZE(pySrc);

        decodeJpeg(src, src_len, jpgbuf, npixels_in, ww, hh);
        copy_image_to_column(jpgbuf, npixels_out, tgt, idx, n);

        // cv::Mat imgTmp = cv::imdecode(cv::Mat(1, src_len, CV_8UC1, src), CV_LOAD_IMAGE_COLOR);

        // cv::Rect myROI(img_size-inner_size, img_size - inner_size, inner_size, inner_size);
        // cv::Rect myROI2(0, inner_size*inner_size, idx, 1);

        // // cv::Mat imgCrop;
        // imgTmp(myROI).reshape(inner_size*inner_size,1).copyTo(tgtimg(myROI2));


        // copy_image_to_column(imgCrop.data, inner_size * inner_size * 3, tgt, idx, n);
    }
}

void copy_image_to_column(unsigned char *src, int srclen, unsigned char *tgt, int destcol, int tgtncol) {
    for (int i=0; i<srclen; i++) {
        tgt[i*tgtncol + destcol] = src[i];
    }
}
void decodeJpeg(unsigned char* src, size_t src_len, unsigned char* dest, int destsize, int& width, int& height) {
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

    assert(destsize == width * height * 3);

    while (cinf.output_scanline < cinf.output_height) {
        JSAMPROW tmp = &dest[width * cinf.out_color_components * cinf.output_scanline];
        assert(jpeg_read_scanlines(&cinf, &tmp, 1) > 0);
    }
    assert(jpeg_finish_decompress(&cinf));
    jpeg_destroy_decompress(&cinf);
}

