#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "core.h"
#include "helpers.h"


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
  self->shape = (int*)malloc(ndim * sizeof(int));
  for (int i = 0; i < ndim; i++) {
    self->shape[i] = shape[i];
  }
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
  for (int i = ndim-1; i >= 0; i--) {
    self->strides[i] = stride;
    stride *= shape[i];
  }
  for (int i = ndim-1; i >= 0; i--) {
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
  char result[4096] = "";
  format_array(self->data, self->shape, self->ndim, 0, result);
  printf("axon.array(%s)\n", result);
}

Array* zeros_like_array(Array* a) {
  float* out = (float*)malloc(a->size * sizeof(float));
  if (out == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(EXIT_FAILURE);
  }
  zeros_like_array_ops(out, a->size);
  return create_array(out, a->ndim, a->shape, a->size);
}

Array* zeros_array(int* shape, size_t size, size_t ndim) {
  float* out = (float*)malloc(size * sizeof(float));
  if (out == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(EXIT_FAILURE);
  }
  zeros_array_ops(out, size);
  return create_array(out, ndim, shape, size);
}

Array* ones_like_array(Array* a) {
  float* out = (float*)malloc(a->size * sizeof(float));
  if (out == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(EXIT_FAILURE);
  }
  ones_like_array_ops(out, a->size);
  return create_array(out, a->ndim, a->shape, a->size);
}

Array* ones_array(int* shape, size_t size, size_t ndim) {
  float* out = (float*)malloc(size * sizeof(float));
  if (out == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(EXIT_FAILURE);
  }
  ones_array_ops(out, size);
  return create_array(out, ndim, shape, size);
}

Array* randn_array(int* shape, size_t size, size_t ndim) {
  float* out = (float*)malloc(size * sizeof(float));
  if (out == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(EXIT_FAILURE);
  }
  fill_randn(out, size);
  return create_array(out, ndim, shape, size);
}

Array* randint_array(int low, int high, int* shape, size_t size, size_t ndim) {
  float* out = (float*)malloc(size * sizeof(float));
  if (out == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(EXIT_FAILURE);
  }
  fill_randint(out, low, high, size);
  return create_array(out, ndim, shape, size);
}

Array* uinform_array(int low, int high, int* shape, size_t size, size_t ndim) {
  float* out = (float*)malloc(size * sizeof(float));
  if (out == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(EXIT_FAILURE);
  }
  fill_uniform(out, low, high, size);
  return create_array(out, ndim, shape, size);
}

Array* fill_array(float fill_val, int* shape, size_t size, size_t ndim) {
  float* out = (float*)malloc(size * sizeof(float));
  if (out == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(EXIT_FAILURE);
  }
  fill_array_ops(out, fill_val, size);
  return create_array(out, ndim, shape, size);
}

Array* linspace_array(float start, float step, float end, int* shape, size_t size, size_t ndim) {
  float* out = (float*)malloc(size * sizeof(float));
  if (out == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(EXIT_FAILURE);
  }
  float step_size = (step > 1) ? (end - start) / (step - 1) : 0.0f;
  linspace_array_ops(out, start, step_size, size);
  return create_array(out, ndim, shape, size);
}