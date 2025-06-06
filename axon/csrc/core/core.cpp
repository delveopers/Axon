#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "core.h"
#include "../cpu/helpers.h"

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
  if (self == NULL) return NULL;

  // using the existing cast_array_dtype function from dtype.h
  void* new_data = cast_array_dtype(self->data, self->dtype, new_dtype, self->size);
  if (new_data == NULL) return NULL;
  
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

void delete_array(Array* self) {
  if (self != NULL) {
    if (self->data) free(self->data);
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

// helper function to truncate elements in a single row
void truncate_row(const float* row, int length, int max_display, char* result) {
  strcat(result, "  [");
  if (length > max_display) {
    for (int i = 0; i < max_display / 2; i++) {
      char buffer[16];
      sprintf(buffer, "%.2f", row[i]);
      strcat(result, buffer);
      strcat(result, ", ");
    }
    strcat(result, "...");
    for (int i = length - max_display / 2; i < length; i++) {
      char buffer[16];
      sprintf(buffer, "%.2f", row[i]);
      strcat(result, ", ");
      strcat(result, buffer);
    }

    // removing trailing comma and space
    if (result[strlen(result) - 2] == ',') {
      result[strlen(result) - 2] = '\0';
    }
  } else {
    for (int i = 0; i < length; i++) {
      char buffer[16];
      sprintf(buffer, "%.2f", row[i]);
      strcat(result, buffer);
      if (i != length - 1) strcat(result, ", ");
    }
  }
  strcat(result, "]");
}

void format_array(const float* data, const int* shape, int ndim, int level, char* result) {
  if (ndim == 1) {
    truncate_row(data, shape[0], 8, result);
    return;
  }

  strcat(result, "[\n");
  int rows_to_display = shape[0] > 4 ? 2 : shape[0]; // truncate rows if needed
  for (int i = 0; i < rows_to_display; i++) {
    if (i > 0) strcat(result, ",\n");
    for (int j = 0; j < level + 1; j++) strcat(result, "  ");
    format_array(data + i * shape[1], shape + 1, ndim - 1, level + 1, result);
  }

  if (shape[0] > 4) {
    strcat(result, ",\n");
    for (int j = 0; j < level + 1; j++) strcat(result, "  ");
    strcat(result, "...");
    strcat(result, ",\n");
    for (int j = 0; j < level + 1; j++) strcat(result, "  ");
    for (int i = shape[0] - 2; i < shape[0]; i++) {
      if (i > shape[0] - 2) strcat(result, ",\n");
      format_array(data + i * shape[1], shape + 1, ndim - 1, level + 1, result);
    }
  }
  strcat(result, "\n]");
}

void print_array(Array* self) {
  if (self == NULL) {
    printf("axon.array(NULL)\n");
    return;
  }

  // converting to float32 for display purposes
  float* temp_float = convert_to_float32(self->data, self->dtype, self->size);
  if (temp_float == NULL) {
    printf("axon.array(ERROR: Failed to convert for display)\n");
    return;
  }
  char result[4096] = "";
  format_array(temp_float, self->shape, self->ndim, 0, result);
  printf("axon.array(%s, dtype=%s)\n", result, get_dtype_name(self->dtype));
  free(temp_float);   // clean up temporary float data
}

Array* zeros_like_array(Array* a) {
  float* out = (float*)malloc(a->size * sizeof(float));
  if (out == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(EXIT_FAILURE);
  }
  zeros_like_array_ops(out, a->size);
  Array* result = create_array(out, a->ndim, a->shape, a->size, a->dtype);
  free(out);
  return result;
}

Array* zeros_array(int* shape, size_t size, size_t ndim, dtype_t dtype) {
  float* out = (float*)malloc(size * sizeof(float));
  if (out == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(EXIT_FAILURE);
  }
  zeros_array_ops(out, size);
  Array* result = create_array(out, ndim, shape, size, dtype);
  free(out);
  return result;
}

Array* ones_like_array(Array* a) {
  float* out = (float*)malloc(a->size * sizeof(float));
  if (out == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(EXIT_FAILURE);
  }
  ones_like_array_ops(out, a->size);
  Array* result = create_array(out, a->ndim, a->shape, a->size, a->dtype);
  free(out);
  return result;
}

Array* ones_array(int* shape, size_t size, size_t ndim, dtype_t dtype) {
  float* out = (float*)malloc(size * sizeof(float));
  if (out == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(EXIT_FAILURE);
  }
  ones_array_ops(out, size);
  Array* result = create_array(out, ndim, shape, size, dtype);
  free(out);
  return result;
}

Array* randn_array(int* shape, size_t size, size_t ndim, dtype_t dtype) {
  float* out = (float*)malloc(size * sizeof(float));
  if (out == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(EXIT_FAILURE);
  }
  fill_randn(out, size);
  Array* result = create_array(out, ndim, shape, size, dtype);
  free(out);
  return result;
}

Array* randint_array(int low, int high, int* shape, size_t size, size_t ndim, dtype_t dtype) {
  float* out = (float*)malloc(size * sizeof(float));
  if (out == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(EXIT_FAILURE);
  }
  fill_randint(out, low, high, size);
  Array* result = create_array(out, ndim, shape, size, dtype);
  free(out);
  return result;
}

Array* uniform_array(int low, int high, int* shape, size_t size, size_t ndim, dtype_t dtype) {
  float* out = (float*)malloc(size * sizeof(float));
  if (out == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(EXIT_FAILURE);
  }
  fill_uniform(out, low, high, size);
  Array* result = create_array(out, ndim, shape, size, dtype);
  free(out);
  return result;
}

Array* fill_array(float fill_val, int* shape, size_t size, size_t ndim, dtype_t dtype) {
  float* out = (float*)malloc(size * sizeof(float));
  if (out == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(EXIT_FAILURE);
  }
  fill_array_ops(out, fill_val, size);
  Array* result = create_array(out, ndim, shape, size, dtype);
  free(out);
  return result;
}

Array* linspace_array(float start, float step, float end, int* shape, size_t size, size_t ndim, dtype_t dtype) {
  float* out = (float*)malloc(size * sizeof(float));
  if (out == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(EXIT_FAILURE);
  }
  float step_size = (step > 1) ? (end - start) / (step - 1) : 0.0f;
  linspace_array_ops(out, start, step_size, size);
  Array* result = create_array(out, ndim, shape, size, dtype);
  free(out);
  return result;
}