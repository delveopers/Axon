#ifndef __ARRAY__H__
#define __ARRAY__H__

#include <stdlib.h>

typedef struct Array {
  float* data;
  int* strides;
  int* backstrides;
  int* shape;
  size_t size;
  size_t ndim;
} Array;

extern "C" {
  // array initialization & deletion related function
  Array* create_array(float* data, size_t ndim, int* shape, size_t size);
  void delete_array(Array* self);
  void delete_shape(Array* self);
  void delete_data(Array* self);
  void delete_strides(Array* self);
}

#endif  //!__ARRAY__H__