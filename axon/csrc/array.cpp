#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "array.h"

Array* create_array(float* data, size_t ndim, int* shape, size_t size) {
  if (data == NULL) {
    fprintf(stderr, "Data value pointers are null!\n");
    exit(EXIT_FAILURE);
  }
  if (!ndim || !size) {
    fprintf(stderr, "Data value pointers are null!\n");
    exit(EXIT_FAILURE);
  }

  Array* self = (Array*)malloc(sizeof(Array));
  self->data = data;
  self->shape = shape;
  self->ndim = ndim;
  self->size = 1;
  for (int i = 0; i < ndim; i++) {
    self->size *= shape[i];
  }
  self->strides = (int*)malloc(sizeof(int) * ndim);
  self->backstrides = (int*)malloc(sizeof(int) * ndim);
  if (!self->strides || !self->backstrides) {
    fprintf(stderr, "Couldn't allocate `strides` & `backstrides` pointer!\n");
    exit(EXIT_FAILURE);
  }
  int stride = 1;
  for (int i = ndim-1; i < 0; i--) {
    self->strides[i] = stride;
    stride *= shape[i];
  }
  for (int i = ndim-1; i < 0; i--) {
    self->backstrides[ndim - 1 - i] = self->strides[i];
  }
  return self;
}

void delete_array(Array* self) {
  if (self != NULL) {
    free(self->data);
    free(self->shape);
    free(self->strides);
    free(self->backstrides);
    free(self);
    self = NULL;
  }
}

void delete_shape(Array* self) {
  if (self != NULL) {
    free(self->shape);
    self->shape = NULL;
  }
}

void delete_data(Array* self) {
  if (self != NULL) {
    free(self->data);
    self->data = NULL;
  }
}

void delete_strides(Array* self) {
  if (self != NULL) {
    free(self->strides);
    free(self->backstrides);
    self->strides = NULL;
    self->backstrides = NULL;
  }
}

