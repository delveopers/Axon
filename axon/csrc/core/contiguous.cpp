#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include "contiguous.h"
#include "../cpu/helpers.h"

int is_contiguous(Array* self) {
  if (!self || !self->ndim) return 1;
  int expected_stride = 1;
  for (int i = self->ndim - 1; i >= 0; i--) {
    if (self->strides[i] != expected_stride) return 0;
    expected_stride *= self->shape[i];
  }
  return 1;
}

static inline void copy_strided_1d(char* src, char* dst, int size, int stride, size_t elem_size) {
  if (stride == 1) { memcpy(dst, src, size * elem_size); }
  else {
    char* src_ptr = src;
    for (int i = 0; i < size; i++) {
      memcpy(dst, src_ptr, elem_size);
      dst += elem_size;
      src_ptr += stride * elem_size;
    }
  }
}

static void contiguous_recursive(char* src, char* dst, int* strides, int* shape, size_t ndim, size_t elem_size, size_t* dst_offset) {
  if (ndim == 1) {
    copy_strided_1d(src, dst + *dst_offset, shape[0], strides[0], elem_size);
    *dst_offset += shape[0] * elem_size;
    return;
  }
  for (int i = 0; i < shape[0]; i++) contiguous_recursive(src + i * strides[0] * elem_size, dst, strides + 1, shape + 1, ndim - 1, elem_size, dst_offset);
}

void contiguous_array_ops(void* src_data, void* dst_data, int* src_strides, int* shape, size_t ndim, size_t elem_size) {
  if (!ndim) return;
  size_t dst_offset = 0;
  contiguous_recursive((char*)src_data, (char*)dst_data, src_strides, shape, ndim, elem_size, &dst_offset);
}

void make_contiguous_inplace(Array* self) {
  if (!self || is_contiguous(self)) return;
  size_t elem_size = get_dtype_size(self->dtype);
  void* new_data = malloc(self->size * elem_size);
  contiguous_array_ops(self->data, new_data, self->strides, self->shape, self->ndim, elem_size);
  free(self->data);
  self->data = new_data;

  int stride = 1;
  for (int i = self->ndim - 1; i >= 0; i--) {
    self->strides[i] = stride;
    stride *= self->shape[i];
  }
  for (size_t i = 0; i < self->ndim; i++) self->backstrides[self->ndim - 1 - i] = self->strides[i];
  self->is_view = 0;
}

size_t calculate_flat_index(int* indices, int* strides, size_t ndim) {
  size_t flat_idx = 0;
  for (size_t i = 0; i < ndim; i++) flat_idx += indices[i] * strides[i];
  return flat_idx;
}

void flat_to_multi_index(size_t flat_idx, int* shape, size_t ndim, int* indices) {
  for (int i = ndim - 1; i >= 0; i--) {
    indices[i] = flat_idx % shape[i];
    flat_idx /= shape[i];
  }
}