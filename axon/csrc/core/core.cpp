#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "core.h"
#include "contiguous.h"

// helper functions
static inline void calculate_strides(int* strides, int* backstrides, int* shape, size_t ndim) {
  int stride = 1;
  for (int i = ndim - 1; i >= 0; i--) {
    strides[i] = stride;
    stride *= shape[i];
  }
  for (size_t i = 0; i < ndim; i++) backstrides[ndim - 1 - i] = strides[i];
}

static inline Array* alloc_array_struct() {
  Array* arr = (Array*)malloc(sizeof(Array));
  if (!arr) {
    fprintf(stderr, "Memory allocation failed!\n");
    exit(EXIT_FAILURE);
  }
  return arr;
}

static inline void* alloc_arrays(size_t ndim, int** shape, int** strides, int** backstrides) {
  void* ptr = malloc(3 * ndim * sizeof(int));
  if (!ptr) return NULL;
  *shape = (int*)ptr, *strides = (int*)ptr + ndim, *backstrides = (int*)ptr + 2 * ndim;
  return ptr;
}

Array* create_array(float* data, size_t ndim, int* shape, size_t size, dtype_t dtype) {
  if (!data || !size) {
    fprintf(stderr, "Invalid inputs for Array creation. Data or Size parameters are missing.\n");
    exit(EXIT_FAILURE);
  }
  Array* self = alloc_array_struct();
  self->dtype = dtype;
  self->is_view = 0;
  self->ndim = ndim;
  self->size = size;
  self->data = allocate_dtype_array(dtype, size);
  convert_from_float32(data, self->data, dtype, size);

  if (ndim == 0) {
    self->shape = self->strides = self->backstrides = NULL;
  } else {
    alloc_arrays(ndim, &self->shape, &self->strides, &self->backstrides);
    memcpy(self->shape, shape, ndim * sizeof(int));
    calculate_strides(self->strides, self->backstrides, shape, ndim);
  }
  return self;
}

Array* cast_array(Array* self, dtype_t new_dtype) {
  if (self == NULL) return NULL;
  // converting to float for intermediate processing
  float* temp_float = convert_to_float32(self->data, self->dtype, self->size);
  if (temp_float == NULL) return NULL;
  // creating new array with target dtype - create_array handles conversion
  Array* result = create_array(temp_float, self->ndim, self->shape, self->size, new_dtype);
  free(temp_float);   // Cleanup temporary float data  
  return result;
}
Array* cast_array_simple(Array* self, dtype_t new_dtype) {
  void* new_data = cast_array_dtype(self->data, self->dtype, new_dtype, self->size);
  Array* result = alloc_array_struct();
  result->data = new_data;
  result->dtype = new_dtype;
  result->ndim = self->ndim;
  result->size = self->size;
  result->is_view = 0;

  alloc_arrays(self->ndim, &result->shape, &result->strides, &result->backstrides);
  memcpy(result->shape, self->shape, self->ndim * sizeof(int));
  memcpy(result->strides, self->strides, self->ndim * sizeof(int));
  memcpy(result->backstrides, self->backstrides, self->ndim * sizeof(int)); 
  return result;
}

Array* contiguous_array(Array* self) {
  Array* result = alloc_array_struct();
  result->dtype = self->dtype;
  result->ndim = self->ndim;
  result->size = self->size;
  result->is_view = 0;
  size_t elem_size = get_dtype_size(self->dtype);
  result->data = malloc(self->size * elem_size);

  alloc_arrays(self->ndim, &result->shape, &result->strides, &result->backstrides);
  memcpy(result->shape, self->shape, self->ndim * sizeof(int));
  calculate_strides(result->strides, result->backstrides, self->shape, self->ndim);
  if (is_contiguous(self)) { memcpy(result->data, self->data, self->size * elem_size); }
  else { contiguous_array_ops(self->data, result->data, self->strides, self->shape, self->ndim, elem_size); }
  return result;
}

Array* view_array(Array* self) {
  Array* view = alloc_array_struct();
  view->data = self->data;
  view->dtype = self->dtype;
  view->ndim = self->ndim;
  view->size = self->size;
  view->is_view = 1;

  alloc_arrays(self->ndim, &view->shape, &view->strides, &view->backstrides);
  memcpy(view->shape, self->shape, self->ndim * sizeof(int));
  memcpy(view->strides, self->strides, self->ndim * sizeof(int));
  memcpy(view->backstrides, self->backstrides, self->ndim * sizeof(int)); 
  return view;
}

Array* reshape_view(Array* self, int* new_shape, size_t new_ndim) {
  size_t new_size = 1;
  for (size_t i = 0; i < new_ndim; i++) {
    if (new_shape[i] <= 0) {
      fprintf(stderr, "Invalid shape: %d\n", new_shape[i]);
      exit(EXIT_FAILURE);
    }
    new_size *= new_shape[i];
  }

  if (new_size != self->size || !is_contiguous(self)) {
    fprintf(stderr, "Cannot reshape. Either Array.size don't match or they'ren't contiguous.\n");
    return NULL;
  }
  Array* reshaped = alloc_array_struct();
  reshaped->data = self->data;
  reshaped->dtype = self->dtype;
  reshaped->ndim = new_ndim;
  reshaped->size = new_size;
  reshaped->is_view = 1;

  alloc_arrays(new_ndim, &reshaped->shape, &reshaped->strides, &reshaped->backstrides);
  memcpy(reshaped->shape, new_shape, new_ndim * sizeof(int));
  calculate_strides(reshaped->strides, reshaped->backstrides, new_shape, new_ndim);
  return reshaped;
}

Array* slice_view(Array* self, int* start, int* end, int* step) {
  Array* sliced = alloc_array_struct();
  sliced->dtype = self->dtype;
  sliced->ndim = self->ndim;
  sliced->is_view = 1;
  alloc_arrays(self->ndim, &sliced->shape, &sliced->strides, &sliced->backstrides);
  size_t new_size = 1, data_offset = 0;
  for (size_t i = 0; i < self->ndim; i++) {
    int dim_start = (start && start[i] >= 0) ? start[i] : 0;
    int dim_end = (end && end[i] >= 0) ? end[i] : self->shape[i];
    int dim_step = (step && step[i] > 0) ? step[i] : 1;
    if (dim_start >= self->shape[i]) dim_start = self->shape[i] - 1;
    if (dim_end > self->shape[i]) dim_end = self->shape[i];
    if (dim_start < 0) dim_start = 0; 
    sliced->shape[i] = (dim_end - dim_start + dim_step - 1) / dim_step;
    sliced->strides[i] = self->strides[i] * dim_step;
    new_size *= sliced->shape[i];
    data_offset += dim_start * self->strides[i];
  }  
  sliced->size = new_size;
  sliced->data = (char*)self->data + (data_offset * get_dtype_size(self->dtype));

  for (size_t i = 0; i < self->ndim; i++) sliced->backstrides[self->ndim - 1 - i] = sliced->strides[i];
  return sliced;
}

// utility functions
int is_view_array(Array* self) { return (self != NULL) ? self->is_view : 0; }

Array* copy_array(Array* self) {
  float* temp_float = convert_to_float32(self->data, self->dtype, self->size);
  Array* copy = create_array(temp_float, self->ndim, self->shape, self->size, self->dtype);
  free(temp_float);
  return copy;
}

void delete_array(Array* self) {
  if (self) {
    if (!self->is_view && self->data) free(self->data);
    if (self->shape) free(self->shape);
    free(self);
  }
}

void delete_shape(Array* self) {
  if (self && self->shape) {
    free(self->shape);
    self->shape = NULL;
  }
}

void delete_data(Array* self) {
  if (self && self->data) {
    free(self->data);
    self->data = NULL;
  }
}

void delete_strides(Array* self) {
  if (self) {
    if (self->strides) {
      free(self->strides);
      self->strides = NULL;
    }
    if (self->backstrides) {
      free(self->backstrides);
      self->backstrides = NULL;
    }
  }
}

float* out_data(Array* self) { return self ? convert_to_float32(self->data, self->dtype, self->size) : NULL; }
int* out_shape(Array* self) { return self ? self->shape : NULL; }
int* out_strides(Array* self) { return self ? self->strides : NULL; }
int out_size(Array* self) { return self ? self->size : 0; }
int is_view_array(Array* self) { return self ? self->is_view : 0; }
int is_contiguous_array(Array* self) { return is_contiguous(self); }
void make_contiguous_inplace_array(Array* self) { make_contiguous_inplace(self); }

int get_linear_index(Array* self, int* indices) {
  int linear_idx = 0;
  for (int i = 0; i < self->ndim; i++) {
    if (indices[i] < 0) indices[i] += self->shape[i];
    if (indices[i] < 0 || indices[i] >= self->shape[i]) {
      fprintf(stderr, "Index out of bounds\n");
      exit(EXIT_FAILURE);
    }
    linear_idx += indices[i] * self->strides[i];
  }
  return linear_idx;
}

#define DTYPE_GET_ITEM(type, cast) case DTYPE_##type: return (float)((cast*)self->data)[linear_idx];
#define DTYPE_SET_ITEM(type, cast) case DTYPE_##type: ((cast*)self->data)[linear_idx] = (cast)value; break;

float get_item_array(Array* self, int* indices) {
  int linear_idx = get_linear_index(self, indices);
  switch (self->dtype) {
    DTYPE_GET_ITEM(FLOAT32, float)
    DTYPE_GET_ITEM(FLOAT64, double)
    DTYPE_GET_ITEM(INT8, int8_t)
    DTYPE_GET_ITEM(INT16, int16_t)
    DTYPE_GET_ITEM(INT32, int32_t)
    DTYPE_GET_ITEM(INT64, int64_t)
    DTYPE_GET_ITEM(UINT8, uint8_t)
    DTYPE_GET_ITEM(UINT16, uint16_t)
    DTYPE_GET_ITEM(UINT32, uint32_t)
    DTYPE_GET_ITEM(UINT64, uint64_t)
    DTYPE_GET_ITEM(BOOL, uint8_t)
    default: return 0.0f;
  }
}

void set_item_array(Array* self, int* indices, float value) {
  int linear_idx = get_linear_index(self, indices);
  switch (self->dtype) {
    DTYPE_SET_ITEM(FLOAT32, float)
    DTYPE_SET_ITEM(FLOAT64, double)
    DTYPE_SET_ITEM(INT8, int8_t)
    DTYPE_SET_ITEM(INT16, int16_t)
    DTYPE_SET_ITEM(INT32, int32_t)
    DTYPE_SET_ITEM(INT64, int64_t)
    DTYPE_SET_ITEM(UINT8, uint8_t)
    DTYPE_SET_ITEM(UINT16, uint16_t)
    DTYPE_SET_ITEM(UINT32, uint32_t)
    DTYPE_SET_ITEM(UINT64, uint64_t)
    case DTYPE_BOOL: ((uint8_t*)self->data)[linear_idx] = (uint8_t)(value != 0); break;
  }
}

// helper function to format element based on dtype
static inline void format_element_by_dtype(void* data, dtype_t dtype, size_t index, char* buffer) {
  switch (dtype) {
    case DTYPE_FLOAT32: sprintf(buffer, "%.3f", ((float*)data)[index]); break;
    case DTYPE_FLOAT64: sprintf(buffer, "%.4f", ((double*)data)[index]); break;
    case DTYPE_INT8: sprintf(buffer, "%d.", ((int8_t*)data)[index]); break;
    case DTYPE_INT16: sprintf(buffer, "%d.", ((int16_t*)data)[index]); break;
    case DTYPE_INT32: sprintf(buffer, "%d.", ((int32_t*)data)[index]); break;
    case DTYPE_INT64: sprintf(buffer, "%lld.", (long long)((int64_t*)data)[index]); break;
    case DTYPE_UINT8: sprintf(buffer, "%u.", ((uint8_t*)data)[index]); break;
    case DTYPE_UINT16: sprintf(buffer, "%u.", ((uint16_t*)data)[index]); break;
    case DTYPE_UINT32: sprintf(buffer, "%u.", ((uint32_t*)data)[index]); break;
    case DTYPE_UINT64: sprintf(buffer, "%llu.", (unsigned long long)((uint64_t*)data)[index]); break;
    case DTYPE_BOOL: sprintf(buffer, "%s", ((uint8_t*)data)[index] ? "True" : "False"); break;
    default: sprintf(buffer, "0"); break;
  }
}

// helper function to truncate elements in a single row
static void truncate_row(Array* self, int row_offset, int length, int max_display, char* result) {
  strcat(result, "  [");
  if (length > max_display) {
    for (int i = 0; i < max_display / 2; i++) {
      char buffer[32];
      format_element_by_dtype(self->data, self->dtype, row_offset + i, buffer);
      strcat(result, buffer);
      strcat(result, ", ");
    }
    strcat(result, "...");
    for (int i = length - max_display / 2; i < length; i++) {
      char buffer[32];
      format_element_by_dtype(self->data, self->dtype, row_offset + i, buffer);
      strcat(result, ", ");
      strcat(result, buffer);
    }
    if (result[strlen(result) - 2] == ',') result[strlen(result) - 2] = '\0';
  } else {
    for (int i = 0; i < length; i++) {
      char buffer[32];
      format_element_by_dtype(self->data, self->dtype, row_offset + i, buffer);
      strcat(result, buffer);
      if (i != length - 1) strcat(result, ", ");
    }
  }
  strcat(result, "]");
}

static void format_array(Array* self, const int* shape, int ndim, int level, int offset, char* result) {
  if (ndim == 1) {
    truncate_row(self, offset, shape[0], 8, result);
    return;
  }

  strcat(result, "[\n");
  int rows_to_display = shape[0] > 8 ? 4 : shape[0];
  int stride = 1;
  for (int i = 1; i < ndim; i++) stride *= shape[i];

  for (int i = 0; i < rows_to_display; i++) {
    if (i > 0) strcat(result, ",\n");
    for (int j = 0; j < level + 1; j++) strcat(result, "  ");
    format_array(self, shape + 1, ndim - 1, level + 1, offset + i * stride, result);
  }

  if (shape[0] > 8) {
    strcat(result, ",\n");
    for (int j = 0; j < level + 1; j++) strcat(result, "  ");
    strcat(result, "...");
    for (int i = shape[0] - 4; i < shape[0]; i++) {
      strcat(result, ",\n");
      for (int j = 0; j < level + 1; j++) strcat(result, "  ");
      format_array(self, shape + 1, ndim - 1, level + 1, offset + i * stride, result);
    }
  }
  strcat(result, "\n");
  for (int j = 0; j < level; j++) strcat(result, "  ");
  strcat(result, "]");
}

void print_array(Array* self) {
  if (!self) {
    printf("axon.array(NULL)\n");
    return;
  }
  char result[8192] = "";
  format_array(self, self->shape, self->ndim, 0, 0, result);
  printf("axon.array(%s, dtype=%s)\n", result, get_dtype_name(self->dtype));
}