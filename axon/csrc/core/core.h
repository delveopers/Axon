/**
  @file core.h header file for core.cpp & array
  * contains core components & functions for array creation/deletion
  * entry point to all the array functions
  * includes only basic core functionalities, ops are on different file
  * compile it as:
    *- '.so': g++ -shared -fPIC -o libarray.so core/core.cpp core/dtype.cpp array.cpp cpu/maths_ops.cpp cpu/helpers.cpp cpu/utils.cpp
    *- '.dll': g++ -shared -o libarray.dll core/core.cpp core/dtype.cpp array.cpp cpu/maths_ops.cpp cpu/helpers.cpp cpu/utils.cpp
    *- 'dylib': g++ -dynamiclib -o libarray.dylib core/core.cpp core/dtype.cpp array.cpp cpu/maths_ops.cpp cpu/helpers.cpp cpu/utils.cpp
*/

#ifndef __CORE__H__
#define __CORE__H__

#include <stdlib.h>
#include "dtype.h"

typedef struct Array {
  void* data;           // raw data pointer (can be any dtype)
  int* strides;
  int* backstrides;
  int* shape;
  size_t size;
  size_t ndim;
  dtype_t dtype;        // data type of the array
  int is_view;          // flag to indicate if this is a view of another array
} Array;

extern "C" {
  // array initialization & deletion related function
  Array* create_array(float* data, size_t ndim, int* shape, size_t size, dtype_t dtype);
  void delete_array(Array* self);
  void delete_shape(Array* self);
  void delete_data(Array* self);
  void delete_strides(Array* self);
  void print_array(Array* self);

  // dtype casting management functions
  Array* cast_array(Array* self, dtype_t new_dtype);
  Array* cast_array_simple(Array* self, dtype_t new_dtype);
  
  // array creation functions with dtype support
  Array* zeros_like_array(Array* a);
  Array* zeros_array(int* shape, size_t size, size_t ndim, dtype_t dtype);
  Array* ones_like_array(Array* a);
  Array* ones_array(int* shape, size_t size, size_t ndim, dtype_t dtype);
  Array* randn_array(int* shape, size_t size, size_t ndim, dtype_t dtype);
  Array* randint_array(int low, int high, int* shape, size_t size, size_t ndim, dtype_t dtype);
  Array* uniform_array(int low, int high, int* shape, size_t size, size_t ndim, dtype_t dtype);
  Array* fill_array(float fill_val, int* shape, size_t size, size_t ndim, dtype_t dtype);
  Array* linspace_array(float start, float step, float end, int* shape, size_t size, size_t ndim, dtype_t dtype);
}

#endif  //!__CORE__H__