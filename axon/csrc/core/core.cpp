#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "core.h"
#include "contiguous.h"

Array* create_array(float* data, size_t ndim, int* shape, size_t size, dtype_t dtype) {
  if (data == NULL || !ndim || !size) {
    fprintf(stderr, "Invalid input parameters!\n");
    exit(EXIT_FAILURE);
  }
  
  Array* self = (Array*)malloc(sizeof(Array));
  if (self == NULL) {
    fprintf(stderr, "Memory allocation failed for Array struct!\n");
    exit(EXIT_FAILURE);
  }
  
  self->dtype = dtype;
  self->is_view = 0;
  self->ndim = ndim;
  self->size = size;
  
  // Allocate and convert to target dtype
  self->data = allocate_dtype_array(dtype, size);
  if (self->data == NULL) {
    free(self);
    exit(EXIT_FAILURE);
  }
  convert_from_float32(data, self->data, dtype, size);
  
  // Copy shape and calculate strides
  self->shape = (int*)malloc(ndim * sizeof(int));
  self->strides = (int*)malloc(sizeof(int) * ndim);
  self->backstrides = (int*)malloc(sizeof(int) * ndim);
  
  if (!self->shape || !self->strides || !self->backstrides) {
    // cleanup and exit
    free(self->data);
    if (self->shape) free(self->shape);
    if (self->strides) free(self->strides);
    if (self->backstrides) free(self->backstrides);
    free(self);
    exit(EXIT_FAILURE);
  }
  
  for (size_t i = 0; i < ndim; i++) {
    self->shape[i] = shape[i];
  }
  
  int stride = 1;
  for (int i = ndim-1; i >= 0; i--) {
    self->strides[i] = stride;
    stride *= shape[i];
  }
  for (size_t i = 0; i < ndim; i++) {
    self->backstrides[ndim - 1 - i] = self->strides[i];
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
  if (self == NULL) {
    fprintf(stderr, "Array value pointers are null!\n");
    exit(EXIT_FAILURE);
  }

  // using the existing cast_array_dtype function from dtype.h
  void* new_data = cast_array_dtype(self->data, self->dtype, new_dtype, self->size);
  if (new_data == NULL) {
    fprintf(stderr, "Memory allocation failed during dtype conversion\n");
    exit(EXIT_FAILURE);
  }
  
  // creating array structure with new data
  Array* result = (Array*)malloc(sizeof(Array));
  result->data = new_data;
  result->dtype = new_dtype;
  result->ndim = self->ndim;
  result->size = self->size;
  result->is_view = 0;
  
  // copy shape and strides
  result->shape = (int*)malloc(self->ndim * sizeof(int));
  result->strides = (int*)malloc(self->ndim * sizeof(int));
  result->backstrides = (int*)malloc(self->ndim * sizeof(int));
  
  memcpy(result->shape, self->shape, self->ndim * sizeof(int));
  memcpy(result->strides, self->strides, self->ndim * sizeof(int));
  memcpy(result->backstrides, self->backstrides, self->ndim * sizeof(int));
  
  return result;
}

int is_contiguous_array(Array* self) {
  return is_contiguous(self);
}

Array* contiguous_array(Array* self) {
  if (self == NULL) {
    fprintf(stderr, "Array value pointers are null!\n");
    exit(EXIT_FAILURE);
  }

  // if already contiguous, return a copy
  if (is_contiguous(self)) {
    Array* result = (Array*)malloc(sizeof(Array));
    result->dtype = self->dtype;
    result->ndim = self->ndim;
    result->size = self->size;
    result->is_view = 0;

    // allocating new data
    size_t elem_size = get_dtype_size(self->dtype);
    result->data = malloc(self->size * elem_size);
    memcpy(result->data, self->data, self->size * elem_size);

    // copying shape and calulating contiguous strides
    result->shape = (int*)malloc(self->ndim * sizeof(int));
    result->strides = (int*)malloc(self->ndim * sizeof(int));
    result->backstrides = (int*)malloc(self->ndim * sizeof(int));
    
    memcpy(result->shape, self->shape, self->ndim * sizeof(int));
    
    // calculating contiguous strides
    int stride = 1;
    for (int i = self->ndim - 1; i >= 0; i--) {
      result->strides[i] = stride;
      stride *= self->shape[i];
    }
    for (size_t i = 0; i < self->ndim; i++) {
      result->backstrides[self->ndim - 1 - i] = result->strides[i];
    }
    
    return result;
  }
  
  // creating new contiguous array
  Array* result = (Array*)malloc(sizeof(Array));
  result->dtype = self->dtype;
  result->ndim = self->ndim;
  result->size = self->size;
  result->is_view = 0;

  // allocating contiguous data
  size_t elem_size = get_dtype_size(self->dtype);
  result->data = malloc(self->size * elem_size);
  
  // copying shape and calulating contiguous strides
  result->shape = (int*)malloc(self->ndim * sizeof(int));
  result->strides = (int*)malloc(self->ndim * sizeof(int));
  result->backstrides = (int*)malloc(self->ndim * sizeof(int));  
  memcpy(result->shape, self->shape, self->ndim * sizeof(int));

  // calculating contiguous strides
  int stride = 1;
  for (int i = self->ndim - 1; i >= 0; i--) {
    result->strides[i] = stride;
    stride *= self->shape[i];
  }
  for (size_t i = 0; i < self->ndim; i++) {
    result->backstrides[self->ndim - 1 - i] = result->strides[i];
  }

  // rearranging data to contiguous layout
  contiguous_array_ops(self->data, result->data, self->strides, self->shape, self->ndim, elem_size);
  return result;
}

void make_contiguous_inplace_array(Array* self) {
  make_contiguous_inplace(self);
}

Array* view_array(Array* self) {
  if (self == NULL) {
    fprintf(stderr, "Array value pointers are null!\n");
    exit(EXIT_FAILURE);
  }
  
  Array* view = (Array*)malloc(sizeof(Array));
  if (view == NULL) {
    fprintf(stderr, "Memory allocation failed for Array view!\n");
    exit(EXIT_FAILURE);
  }

  // sharing the same data pointer
  view->data = self->data;
  view->dtype = self->dtype;
  view->ndim = self->ndim;
  view->size = self->size;
  view->is_view = 1;  // Mark as view
  
  // copying shape and strides
  view->shape = (int*)malloc(self->ndim * sizeof(int));
  view->strides = (int*)malloc(self->ndim * sizeof(int));
  view->backstrides = (int*)malloc(self->ndim * sizeof(int));
  
  if (!view->shape || !view->strides || !view->backstrides) {
    if (view->shape) free(view->shape);
    if (view->strides) free(view->strides);
    if (view->backstrides) free(view->backstrides);
    free(view);
    exit(EXIT_FAILURE);
  }
  
  memcpy(view->shape, self->shape, self->ndim * sizeof(int));
  memcpy(view->strides, self->strides, self->ndim * sizeof(int));
  memcpy(view->backstrides, self->backstrides, self->ndim * sizeof(int));
  
  return view;
}

Array* reshape_view(Array* self, int* new_shape, size_t new_ndim) {
  if (self == NULL || new_shape == NULL) {
    fprintf(stderr, "Array value pointers are null!\n");
    exit(EXIT_FAILURE);
  }

  // calculating new size
  size_t new_size = 1;
  for (size_t i = 0; i < new_ndim; i++) {
    if (new_shape[i] <= 0) {
      fprintf(stderr, "Invalid shape dimension: %d\n", new_shape[i]);
      return NULL;
    }
    new_size *= new_shape[i];
  }
  
  // checking if reshape is compatible
  if (new_size != self->size) {
    fprintf(stderr, "Cannot reshape array of size %zu into shape with size %zu\n", self->size, new_size);
    return NULL;
  }
  
  // for views, the original array must be contiguous
  if (!is_contiguous(self)) {
    fprintf(stderr, "Cannot reshape non-contiguous array. Use contiguous() first.\n");
    return NULL;
  }

  Array* reshaped = (Array*)malloc(sizeof(Array));
  if (reshaped == NULL) {
    fprintf(stderr, "Memory allocation failed for reshaped array!\n");
    exit(EXIT_FAILURE);
  }
  
  // sharing the same data
  reshaped->data = self->data;
  reshaped->dtype = self->dtype;
  reshaped->ndim = new_ndim;
  reshaped->size = new_size;
  reshaped->is_view = 1;
  
  // allocating new shape and strides
  reshaped->shape = (int*)malloc(new_ndim * sizeof(int));
  reshaped->strides = (int*)malloc(new_ndim * sizeof(int));
  reshaped->backstrides = (int*)malloc(new_ndim * sizeof(int));
  
  if (!reshaped->shape || !reshaped->strides || !reshaped->backstrides) {
    if (reshaped->shape) free(reshaped->shape);
    if (reshaped->strides) free(reshaped->strides);
    if (reshaped->backstrides) free(reshaped->backstrides);
    free(reshaped);
    exit(EXIT_FAILURE);
  }
  
  // setting new shape
  for (size_t i = 0; i < new_ndim; i++) {
    reshaped->shape[i] = new_shape[i];
  }
  
  // calculating new strides
  int stride = 1;
  for (int i = new_ndim - 1; i >= 0; i--) {
    reshaped->strides[i] = stride;
    stride *= new_shape[i];
  }
  for (size_t i = 0; i < new_ndim; i++) {
    reshaped->backstrides[new_ndim - 1 - i] = reshaped->strides[i];
  }
  
  return reshaped;
}

Array* slice_view(Array* self, int* start, int* end, int* step) {
  if (self == NULL) {
    fprintf(stderr, "Array value pointers are null!\n");
    exit(EXIT_FAILURE);
  }
  
  // calculating new shape and strides for sliced view
  int* new_shape = (int*)malloc(self->ndim * sizeof(int));
  int* new_strides = (int*)malloc(self->ndim * sizeof(int));
  size_t new_size = 1;
  size_t data_offset = 0;
  
  for (size_t i = 0; i < self->ndim; i++) {
    int dim_start = (start && start[i] >= 0) ? start[i] : 0;
    int dim_end = (end && end[i] >= 0) ? end[i] : self->shape[i];
    int dim_step = (step && step[i] > 0) ? step[i] : 1;
    
    // clamping to valid range
    if (dim_start >= self->shape[i]) dim_start = self->shape[i] - 1;
    if (dim_end > self->shape[i]) dim_end = self->shape[i];
    if (dim_start < 0) dim_start = 0;
    
    // calculating new dimension size
    new_shape[i] = (dim_end - dim_start + dim_step - 1) / dim_step;
    new_strides[i] = self->strides[i] * dim_step;
    new_size *= new_shape[i];
    
    // calculating offset in original data
    data_offset += dim_start * self->strides[i];
  }
  
  Array* sliced = (Array*)malloc(sizeof(Array));
  if (sliced == NULL) {
    free(new_shape);
    free(new_strides);
    exit(EXIT_FAILURE);
  }
  
  // point to offset data
  size_t elem_size = get_dtype_size(self->dtype);
  sliced->data = (char*)self->data + (data_offset * elem_size);
  sliced->dtype = self->dtype;
  sliced->ndim = self->ndim;
  sliced->size = new_size;
  sliced->is_view = 1;
  
  sliced->shape = new_shape;
  sliced->strides = new_strides;
  sliced->backstrides = (int*)malloc(self->ndim * sizeof(int));
  
  for (size_t i = 0; i < self->ndim; i++) {
    sliced->backstrides[self->ndim - 1 - i] = sliced->strides[i];
  }
  
  return sliced;
}

// utility functions
int is_view_array(Array* self) {
  return (self != NULL) ? self->is_view : 0;
}

Array* copy_array(Array* self) {
  if (self == NULL) {
    fprintf(stderr, "Array value pointers are null!\n");
    exit(EXIT_FAILURE);
  }

  float* temp_float = convert_to_float32(self->data, self->dtype, self->size);
  if (temp_float == NULL) {
    fprintf(stderr, "Couldn't allocate Array value pointers!\n");
    exit(EXIT_FAILURE);
  };
  // creating new array - this will allocate new data
  Array* copy = create_array(temp_float, self->ndim, self->shape, self->size, self->dtype);
  free(temp_float);
  return copy;
}

void delete_array(Array* self) {
  if (self != NULL) {
    // only free data if it's not a view
    if (!self->is_view && self->data) {
      free(self->data);
    }
    if (self->shape) free(self->shape);
    if (self->strides) free(self->strides);
    if (self->backstrides) free(self->backstrides);
    free(self);
  }
}

void delete_shape(Array* self) {
  if (self != NULL && self->shape != NULL) {
    free(self->shape);
    self->shape = NULL;
  }
}

void delete_data(Array* self) {
  if (self != NULL) {
    if (self->data) {
      free(self->data);
      self->data = NULL;
    }
  }
}

void delete_strides(Array* self) {
  if (self != NULL) {
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

float* out_data(Array* self) {
  if (self == NULL) {
    fprintf(stderr, "Invalid input parameters!\n");
    exit(EXIT_FAILURE);
  }
  float* temp_float = convert_to_float32(self->data, self->dtype, self->size);
  return temp_float;
}

int* out_shape(Array* self) {
  if (self == NULL) {
    fprintf(stderr, "Invalid input parameters!\n");
    exit(EXIT_FAILURE);
  }
  return self->shape;
}

int* out_strides(Array* self) {
  if (self == NULL) {
    fprintf(stderr, "Invalid input parameters!\n");
    exit(EXIT_FAILURE);
  }
  return self->strides;
}

int out_size(Array* self) {
  if (self == NULL) {
    fprintf(stderr, "Invalid input parameters!\n");
    exit(EXIT_FAILURE);
  }
  return self->size;
}

// helper function to format element based on dtype
void format_element_by_dtype(void* data, dtype_t dtype, size_t index, char* buffer) {
  switch (dtype) {
    case DTYPE_FLOAT32:
      sprintf(buffer, "%.3f", ((float*)data)[index]);
      break;
    case DTYPE_FLOAT64:
      sprintf(buffer, "%.4f", ((double*)data)[index]);
      break;
    case DTYPE_INT8:
      sprintf(buffer, "%d.", ((int8_t*)data)[index]);
      break;
    case DTYPE_INT16:
      sprintf(buffer, "%d.", ((int16_t*)data)[index]);
      break;
    case DTYPE_INT32:
      sprintf(buffer, "%d.", ((int32_t*)data)[index]);
      break;
    case DTYPE_INT64:
      sprintf(buffer, "%lld.", (long long)((int64_t*)data)[index]);
      break;
    case DTYPE_UINT8:
      sprintf(buffer, "%u.", ((uint8_t*)data)[index]);
      break;
    case DTYPE_UINT16:
      sprintf(buffer, "%u.", ((uint16_t*)data)[index]);
      break;
    case DTYPE_UINT32:
      sprintf(buffer, "%u.", ((uint32_t*)data)[index]);
      break;
    case DTYPE_UINT64:
      sprintf(buffer, "%llu.", (unsigned long long)((uint64_t*)data)[index]);
      break;
    case DTYPE_BOOL:
      sprintf(buffer, "%s", ((uint8_t*)data)[index] ? "True" : "False");
      break;
    default:
      sprintf(buffer, "0");
      break;
  }
}

// helper function to truncate elements in a single row
void truncate_row(Array* self, const void* row_data, int row_offset, int length, int max_display, char* result) {
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

    // removing trailing comma and space
    if (result[strlen(result) - 2] == ',') {
      result[strlen(result) - 2] = '\0';
    }
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

void format_array(Array* self, const int* shape, int ndim, int level, int offset, char* result) {
  if (ndim == 1) {
    truncate_row(self, self->data, offset, shape[0], 8, result);
    return;
  }

  strcat(result, "[\n");
  int rows_to_display = shape[0] > 8 ? 4 : shape[0]; // truncate rows if needed, show first 4 and last 4
  int stride = 1;
  for (int i = 1; i < ndim; i++) {
    stride *= shape[i];
  }

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
  if (self == NULL) {
    printf("axon.array(NULL)\n");
    return;
  }

  char result[8192] = "";
  format_array(self, self->shape, self->ndim, 0, 0, result);
  printf("axon.array(%s, dtype=%s)\n", result, get_dtype_name(self->dtype));
}