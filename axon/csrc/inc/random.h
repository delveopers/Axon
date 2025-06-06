/**
  @file random.h
  @brief simple random number generator fully deterministic, similar to python's random lib
        used for building higher order random class.
  @author @shivendrra
  @src: https://github.com/delveopers/Axon/blob/main/axon/csrc/inc/random.h
  @license: MIT License
*/

#ifndef __RANDOM__H__
#define __RANDOM__H__

#include <time.h>
#include <stdint.h>
#include <math.h>

#define  PI_VAL  3.14159265f
#define  UINT64_CAST  0xFFFFFFFFFFFFFFFF
#define  UINT32_CAST  0xFFFFFFFF

#ifdef  __cplusplus
extern "C" {
#endif  //__cplusplus

typedef struct {
  uint64_t state;  // initial seed val
  float spare;     // spare value for Box-Muller
  int has_spare;   // flag for spare value
} RNG;

static inline void rng_state(RNG* rng, uint64_t state) {
  rng->state = state;
  rng->has_spare = 0;
  rng->spare = 0.0f;
}

static inline uint32_t random_u32(RNG* rng) {
  rng->state ^= (rng->state >> 12) & UINT64_CAST;
  rng->state ^= (rng->state >> 25) & UINT64_CAST;
  rng->state ^= (rng->state >> 27) & UINT64_CAST;
  return ((rng->state * 0x2545F4914F6CDD1D) >> 32) & UINT32_CAST;
}

static inline float rng_random(RNG* rng) {
  return (random_u32(rng) >> 8) / 16777216.0f;
}

static inline float rng_uniform(RNG* rng, float a, float b) {
  return a + (b - a) * rng_random(rng);
}

static inline void rng_rand(RNG* rng, float* out, int size) {
  for (int i = 0; i < size; i++) {
    out[i] = rng_random(rng);
  }
}

static inline void rng_rand_uniform(RNG* rng, float* out, int size, float low, float high) {
  for (int i = 0; i < size; i++) {
    out[i] = rng_uniform(rng, low, high);
  }
}

static inline void rng_randint(RNG* rng, int* out, int size, int low, int high) {
  for (int i = 0; i < size; i++) {
    float r = rng_random(rng);
    out[i] = low + (int)(r * (high - low));
  }
}

static inline void rng_randn(RNG* rng, float* out, int size) {
  // rng from standard normal distribution using Box-Muller transform
  for (int i = 0; i < size; i++) {
    if (rng->has_spare) {
      out[i] = rng->spare;
      rng->has_spare = 0;
    } else {
      float u1 = rng_random(rng);
      float u2 = rng_random(rng);
      float z0 = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * PI_VAL * u2);
      float z1 = sqrtf(-2.0f * logf(u1)) * sinf(2.0f * PI_VAL * u2);
      out[i] = z0;
      rng->spare = z1;
      rng->has_spare = 1;
    }
  }
}

static inline void rng_choice(RNG* rng, int* a, int* out, int a_len, int size, int replace) {
  if (!replace && size > a_len) return;

  if (replace) {
    for (int i = 0; i < size; i++) {
      int idx = (int)(rng_random(rng) * a_len);
      out[i] = a[idx];
    }
  } else {
    // Fisher-Yates style partial shuffle
    int temp[256];
    if (a_len > 256) return;
    for (int i = 0; i < a_len; i++) temp[i] = a[i];
    for (int i = 0; i < size; i++) {
      int j = i + (int)(rng_random(rng) * (a_len - i));
      int tmp = temp[i];
      temp[i] = temp[j];
      temp[j] = tmp;
    }
    for (int i = 0; i < size; i++) out[i] = temp[i];
  }
}

static inline uint64_t current_time_seed() {
  return (uint64_t)(time(NULL)) * 1000000;
}

#ifdef  __cplusplus
}
#endif

#endif