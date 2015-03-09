#include <cstdio>
#include <cstdlib>
#include <Python.h>
#include <iostream>
#include <jpeglib.h>
#include <boost/thread.hpp>
#include <boost/bind.hpp>

typedef long long int int64;
typedef unsigned short int uint8;

class ImgWorker {
 protected:
    PyObject* _pyList;
    PyArrayObject* _pyTgt;
    int64 _start_img, _end_img, _num_imgs, _num_cols, _num_rows;
    int64 _img_size, _inner_size, _npixels_in, _npixels_out, _inner_pixels;
    bool _center;

    unsigned char* _tgt;
    unsigned char* _jpgbuf;
    unsigned int _rseed;

    void decodeJpeg(unsigned char* src, size_t src_len, int& width, int& height);
    virtual void crop(int64 i, int64 src_width, int64 src_height, bool flip, int64 crop_start_x, int64 crop_start_y);
 public:
    ImgWorker(PyObject* pyList, PyArrayObject *pyTarget, int start_img, int end_img, int img_size, int inner_size, bool center, int num_imgs);
    virtual ~ImgWorker();
    void decodeList();

};
