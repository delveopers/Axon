#include <stdlib.h>
#include "inc/random.h"
#include "helpers.h"

static RNG global_rng;
static int rng_initialized = 0;

static inline void ensure_rng_initialized() {
  if (!rng_initialized) {
    rng_state(&global_rng, current_time_seed());
    rng_initialized = 1;
  }
}

void fill_randn(float* out, size_t size) {
  ensure_rng_initialized();
  rng_randn(&global_rng, out, size);
}

void fill_uniform(float* out, float low, float high, size_t size) {
  ensure_rng_initialized();
  rng_rand_uniform(&global_rng, out, size, low, high);
}

void fill_randint(float* out, int low, int high, size_t size) {
  ensure_rng_initialized();

  int* temp = (int*)malloc(sizeof(int) * size);
  rng_randint(&global_rng, temp, size, low, high);    // returns int values
  // but since we handle ops in float32 so, will create a temp int array
  // then recast & update values to the float array
  for (int i = 0; i < size; i++) {
    out[i] = (float)temp[i];
  }
  free(temp);
}

void ones_like_array_ops(float* out, size_t size) {
  for (int i = 0; i < size; i++) {
    out[i] = 1.0f;
  }
}

void zeros_like_array_ops(float* out, size_t size) {
  for (int i = 0; i < size; i++) {
    out[i] = 0.0f;
  }
}

void ones_array_ops(float* out, size_t size) {
  for (int i = 0; i < size; i++) {
    out[i] = 1.0f;
  }
}

void zeros_array_ops(float* out, size_t size) {
  for (int i = 0; i < size; i++) {
    out[i] = 0.0f;
  }
}

void fill_array_ops(float* out, float value, size_t size) {
  for (int i = 0; i < size; i++) {
    out[i] = value;
  }
}

void linspace_array_ops(float* out, float start, float step_size, size_t size) {
  for (int i = 0; i < size; i++) {
    out[i] = start + i * step_size;
  }
}