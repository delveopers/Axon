#include "dtype.h"
#include <string.h>

// batch conversion using function pointers - eliminates switch overhead
typedef void (*conversion_func_t)(void* src, void* dst, size_t count);

// template-like macros for type-specific batch conversions
#define DEFINE_BATCH_TO_FLOAT32(src_type, src_field) \
static void batch_##src_type##_to_float32(void* src, void* dst, size_t count) { \
  src_type* s = (src_type*)src; \
  float* d = (float*)dst; \
  for(size_t i = 0; i < count; i++) d[i] = (float)s[i]; \
}

#define DEFINE_BATCH_FROM_FLOAT32(dst_type, clamp_func) \
static void batch_float32_to_##dst_type(void* src, void* dst, size_t count) { \
  float* s = (float*)src; \
  dst_type* d = (dst_type*)dst; \
  for(size_t i = 0; i < count; i++) d[i] = (dst_type)clamp_func((double)s[i]); \
}

// generating all batch conversion functions
DEFINE_BATCH_TO_FLOAT32(int8_t, i8)
DEFINE_BATCH_TO_FLOAT32(int16_t, i16) 
DEFINE_BATCH_TO_FLOAT32(int32_t, i32)
DEFINE_BATCH_TO_FLOAT32(int64_t, i64)
DEFINE_BATCH_TO_FLOAT32(uint8_t, u8)
DEFINE_BATCH_TO_FLOAT32(uint16_t, u16)
DEFINE_BATCH_TO_FLOAT32(uint32_t, u32)
DEFINE_BATCH_TO_FLOAT32(uint64_t, u64)
DEFINE_BATCH_TO_FLOAT32(double, f64)

static void batch_float32_to_float32(void* src, void* dst, size_t count) {
  memcpy(dst, src, count * sizeof(float));
}

static void batch_bool_to_float32(void* src, void* dst, size_t count) {
  uint8_t* s = (uint8_t*)src;
  float* d = (float*)dst;
  for(size_t i = 0; i < count; i++) d[i] = s[i] ? 1.0f : 0.0f;
}

// pptimized batch conversion lookup table
static conversion_func_t to_float32_funcs[] = {
  [DTYPE_FLOAT32] = batch_float32_to_float32,
  [DTYPE_FLOAT64] = batch_double_to_float32,
  [DTYPE_INT8] = batch_int8_t_to_float32,
  [DTYPE_INT16] = batch_int16_t_to_float32,
  [DTYPE_INT32] = batch_int32_t_to_float32,
  [DTYPE_INT64] = batch_int64_t_to_float32,
  [DTYPE_UINT8] = batch_uint8_t_to_float32,
  [DTYPE_UINT16] = batch_uint16_t_to_float32,
  [DTYPE_UINT32] = batch_uint32_t_to_float32,
  [DTYPE_UINT64] = batch_uint64_t_to_float32,
  [DTYPE_BOOL] = batch_bool_to_float32
};

// fast batch conversion to float32
float* convert_to_float32_fast(void* data, dtype_t dtype, size_t size) {
  float* float_data = (float*)malloc(size * sizeof(float));
  if (!float_data) return NULL;
  conversion_func_t converter = to_float32_funcs[dtype];
  converter(data, float_data, size);
  return float_data;
}

// direct dtype-to-dtype conversion without float32 intermediate
void convert_dtype(void* src, dtype_t src_dtype, void* dst, dtype_t dst_dtype, size_t size) {
  if (src_dtype == dst_dtype) {
    memcpy(dst, src, size * get_dtype_size(src_dtype));
    return;
  }

  // special cases for same-size types (no precision loss)
  if (get_dtype_size(src_dtype) == get_dtype_size(dst_dtype)) {
    switch(src_dtype) {
      case DTYPE_INT32:
        if (dst_dtype == DTYPE_FLOAT32) {
          int32_t* s = (int32_t*)src; float* d = (float*)dst;
          for(size_t i = 0; i < size; i++) d[i] = (float)s[i];
          return;
        }
        break;
      case DTYPE_UINT32:
        if (dst_dtype == DTYPE_FLOAT32) {
          uint32_t* s = (uint32_t*)src; float* d = (float*)dst;
          for(size_t i = 0; i < size; i++) d[i] = (float)s[i];
          return;
        }
        break;
    }
  }

  // fallback to float32 intermediate for complex conversions
  float* temp = convert_to_float32_fast(src, src_dtype, size);
  if (!temp) return;
  convert_from_float32(temp, dst, dst_dtype, size);
  free(temp);
}

// SIMD-optimized version for common conversions (requires SSE2)
#ifdef __SSE2__
#include <emmintrin.h>

void convert_int32_to_float32_simd(int32_t* src, float* dst, size_t size) {
  size_t simd_size = size & ~3; // Process 4 elements at a time
  
  for(size_t i = 0; i < simd_size; i += 4) {
    __m128i ints = _mm_loadu_si128((__m128i*)(src + i));
    __m128 floats = _mm_cvtepi32_ps(ints);
    _mm_storeu_ps(dst + i, floats);
  }
  
  // Handle remaining elements
  for(size_t i = simd_size; i < size; i++) {
    dst[i] = (float)src[i];
  }
}

void convert_float32_to_int32_simd(float* src, int32_t* dst, size_t size) {
  size_t simd_size = size & ~3;
  
  for(size_t i = 0; i < simd_size; i += 4) {
    __m128 floats = _mm_loadu_ps(src + i);
    __m128i ints = _mm_cvtps_epi32(floats);
    _mm_storeu_si128((__m128i*)(dst + i), ints);
  }
  
  for(size_t i = simd_size; i < size; i++) {
    dst[i] = (int32_t)clamp_to_int_range(src[i], DTYPE_INT32);
  }
}
#endif

// chunked processing for very large arrays
void convert_dtype_chunked(void* src, dtype_t src_dtype, void* dst, dtype_t dst_dtype, size_t size) {
  const size_t chunk_size = 8192; // Optimize for L1 cache

  for(size_t offset = 0; offset < size; offset += chunk_size) {
    size_t current_chunk = (offset + chunk_size > size) ? size - offset : chunk_size;
    char* src_ptr = (char*)src + offset * get_dtype_size(src_dtype);
    char* dst_ptr = (char*)dst + offset * get_dtype_size(dst_dtype);
    convert_dtype(src_ptr, src_dtype, dst_ptr, dst_dtype, current_chunk);
  }
}

// memory pool for temporary conversions to reduce malloc/free overhead
typedef struct {
  float* buffer;
  size_t capacity;
} conversion_pool_t;

static conversion_pool_t pool = {NULL, 0};

void init_conversion_pool(size_t initial_size) {
  pool.buffer = (float*)malloc(initial_size * sizeof(float));
  pool.capacity = pool.buffer ? initial_size : 0;
}

void cleanup_conversion_pool() {
  if (pool.buffer) {
    free(pool.buffer);
    pool.buffer = NULL;
    pool.capacity = 0;
  }
}

float* get_temp_float_buffer(size_t size) {
  if (size <= pool.capacity) return pool.buffer;
  
  float* new_buffer = (float*)realloc(pool.buffer, size * sizeof(float));
  if (new_buffer) {
    pool.buffer = new_buffer;
    pool.capacity = size;
    return pool.buffer;
  }
  return (float*)malloc(size * sizeof(float)); // fallback case
}

// Optimized replacement for existing functions
float* convert_to_float32(void* data, dtype_t dtype, size_t size) {
  if (dtype == DTYPE_FLOAT32) {
    float* result = (float*)malloc(size * sizeof(float));
    if (result) memcpy(result, data, size * sizeof(float));
    return result;
  }
  return convert_to_float32_fast(data, dtype, size);
}

void copy_with_dtype_conversion(void* src, dtype_t src_dtype, void* dst, dtype_t dst_dtype, size_t size) {
  if (src_dtype == dst_dtype) {
    memcpy(dst, src, size * get_dtype_size(src_dtype));
    return;
  }

  if (size > 8192) { convert_dtype_chunked(src, src_dtype, dst, dst_dtype, size); }
  else { convert_dtype(src, src_dtype, dst, dst_dtype, size); }
}