#ifndef JPEG_MAIN_H
#define JPEG_MAIN_H

#include <cstdio>
#include <cstdlib>
#include <Python.h>
#include <vector>
#include <string>
#include <iostream>
#include <jpeglib.h>
#include <arrayobject.h>
#include <boost/thread.hpp>

#ifndef DIVUP
#define DIVUP(x, y) (((x) + (y) - 1) / (y))
#endif

#define NUM_JPEG_DECODER_THREADS        4


class DecoderThread : public boost::thread {
 protected:
    PyObject* _pyList;
    int64 _start_img, _end_img;
    int64 _img_size, _inner_size, _inner_pixels;
    bool _test;

    unsigned char* _decodeTarget;
    int64 _decodeTargetSize;
    unsigned int _rseed;

    void* run();
    void decodeJpeg(int idx, int& width, int& height);
    double randUniform();
    double randUniform(double min, double max);
    void crop(int64 i, int64 width, int64 height, bool flip);
    virtual void crop(int64 i, int64 src_width, int64 src_height, bool flip, int64 crop_start_x, int64 crop_start_y);
 public:
    DecoderThread(PyObject* pyList, Matrix& target, int start_img, int end_img, int img_size, int inner_size, bool test, bool multiview);
    virtual ~DecoderThread();
};

#endif // JPEG_MAIN_H