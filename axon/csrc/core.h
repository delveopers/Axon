/**
  @file core.h header file for core.cpp & array
  * contains core components & functions for array creation/deletion
  * entry point to all the array functions
  * includes only basic core functionalities, ops are on different file
  * compile it as:
    *- '.so': g++ -shared -fPIC -o libarray.so core.cpp array.cpp maths_ops.cpp
    *- '.dll': g++ -shared -o libarray.dll core.cpp array.cpp maths_ops.cpp
    *- 'dylib': g++ -dynamiclib -o libarray.dylib core.cpp array.cpp maths_ops.cpp
*/

#ifndef __CORE__H__
#define __CORE__H__

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
  void print_array(Array* self);

  Array* zeros_like_array(Array* a);
  Array* zeros_array(int* shape, size_t size, size_t ndim);
  Array* ones_like_array(Array* a);
  Array* ones_array(int* shape, size_t size, size_t ndim);
  Array* randn_array(int* shape, size_t size, size_t ndim);
  Array* randint_array(int low, int high, int* shape, size_t size, size_t ndim);
  Array* uinform_array(int low, int high, int* shape, size_t size, size_t ndim);
}

#endif  //!__ARRAY__H__