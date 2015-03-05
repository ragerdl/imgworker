//#include <boost/python.hpp>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <arrayobject.h>
#include <assert.h>
#include <vips/vips8>
typedef long long int int64;
typedef unsigned short int uint8;

extern "C" void init_ImWorker();
PyObject* decodeTransformListMT(PyObject *self, PyObject *args);
using namespace vips;

// using namespace std;
// static ImWorker* model = NULL;

static PyMethodDef _ImWorkerMethods[] = {{ "decodeTransformListMT",         decodeTransformListMT,             METH_VARARGS },
                                         { NULL, NULL }
};

void init_ImWorker() {
    (void) Py_InitModule("_ImWorker", _ImWorkerMethods);
    if ( VIPS_INIT("istring") )
        return( -1 );
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
    assert(PyTarget != NULL);
    assert(PyArray_ISONESEGMENT(pyTarget));
    assert(PyArray_CHKFLAGS(pyTarget, NPY_ARRAY_C_CONTIGUOUS));
    assert(PyArray_NDIM(pyTarget)==2);

    // Thread* threads[NUM_JPEG_DECODER_THREADS];
    int num_imgs = PyList_GET_SIZE(pyJpegStrings);
    decodeImgToArray(pyJpegStrings, pyTarget, 0, 1, img_size);
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

void decodeImgToArray(PyObject *pyJpegStrings, PyArrayObject *pyTarget, start_img, end_img, img_size) {
    int m = PyArray_DIM(pyTarget, 0);
    int n = PyArray_DIM(pyTarget, 1);
    uint8 *data = (uint8 *) PyArray_DATA(pyTarget);

    for (int idx = start_img; idx < end_img; idx++) {
        PyObject* pySrc = PyList_GET_ITEM(pyJpegStrings, idx);
        unsigned char* src = (unsigned char *) PyString_AsString(pySrc);
        size_t src_len = PyString_GET_SIZE(pySrc)
        VImage::option()->
        set( "extend", VIPS_EXTEND_BACKGROUND )->
        set( "background", 128 )
        VImage in = VImage::new_from_buffer( (void *) src, src_len, "jpg");


    }

        // if (PyArray_ISONESEGMENT(pyTarget) && PyArray_NDIM(pyTarget)!=0) { 
        //     this->_data = (MTYPE*) PyArray_DATA(pyTarget);
        //     this->_ownsData = false;
        //     this->_trans = PyArray_CHKFLAGS(pyTarget, NPY_ARRAY_C_CONTIGUOUS) ? CblasNoTrans : CblasTrans;
        // } else {
        //     this->_data = new MTYPE[PyArray_DIM(pyTarget,0) * PyArray_DIM(pyTarget,1)];
        //     for (int64 i = 0; i < PyArray_DIM(pyTarget,0); i++) {
        //         for (int64 j = 0; j < PyArray_DIM(pyTarget,1); j++) {
        //             (*this)(i,j) = *reinterpret_cast<MTYPE*>(PyArray_GETPTR2(pyTarget,i,j));
        //         }
        //     }
        //     this->_ownsData = true;
        // }
}

// void DecoderThread::decodeJpeg(int idx, int& width, int& height) {
//     PyObject* pySrc = PyList_GET_ITEM(_pyList, idx);
//     unsigned char* src = (unsigned char*)PyString_AsString(pySrc);
//     size_t src_len = PyString_GET_SIZE(pySrc);
    
//     struct jpeg_decompress_struct cinf;
//     struct jpeg_error_mgr jerr;
//     cinf.err = jpeg_std_error(&jerr);
//     jpeg_create_decompress(&cinf);
//     jpeg_mem_src(&cinf, src, src_len);
//     assert(jpeg_read_header(&cinf, TRUE));
//     cinf.out_color_space = JCS_RGB;
//     assert(jpeg_start_decompress(&cinf));
//     assert(cinf.num_components == 3 || cinf.num_components == 1);
//     width = cinf.image_width;
//     height = cinf.image_height;

//     if (_decodeTargetSize < width * height * 3) {
//         free(_decodeTarget);
//         _decodeTargetSize = width * height * 3 * 3;
//         _decodeTarget = (unsigned char*)malloc(_decodeTargetSize);
//     }
    
//     while (cinf.output_scanline < cinf.output_height) {
//         JSAMPROW tmp = &_decodeTarget[width * cinf.out_color_components * cinf.output_scanline];
//         assert(jpeg_read_scanlines(&cinf, &tmp, 1) > 0);
//     }
//     assert(jpeg_finish_decompress(&cinf));
//     jpeg_destroy_decompress(&cinf);
// }

