/**
  @file test_array.c
  @brief Comprehensive test suite for the array library

  * Compile with:
  * g++ -o test_array test_array.cpp -L. -larray -lm

  * Or if compiling with source files:
  * g++ -o test_array test_array.cpp ../axon/csrc/core/core.cpp ../axon/csrc/core/dtype.cpp ../axon/csrc/array.cpp ../axon/csrc/cpu/maths_ops.cpp ../axon/csrc/cpu/helpers.cpp ../axon/csrc/cpu/utils.cpp ../axon/csrc/cpu/red_ops.cpp -lm -lstdc++
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include "../axon/csrc/core/core.h"
#include "../axon/csrc/array.h"

#define C_PI 3.14159265359

// Test result tracking
static int tests_passed = 0;
static int tests_failed = 0;

// Utility macros for testing
#define TEST_START(name) \
  printf("\n=== Testing %s ===\n", name);

#define ASSERT_TRUE(condition, message) \
  do { \
    if (condition) { \
      printf("// PASS: %s\n", message); \
      tests_passed++; \
    } else { \
      printf("// FAIL: %s\n", message); \
      tests_failed++; \
    } \
  } while(0)

#define ASSERT_NOT_NULL(ptr, message) \
  ASSERT_TRUE((ptr) != NULL, message)

#define ASSERT_EQUAL_INT(expected, actual, message) \
  ASSERT_TRUE((expected) == (actual), message)

#define ASSERT_EQUAL_FLOAT(expected, actual, tolerance, message) \
  ASSERT_TRUE(fabs((expected) - (actual)) < (tolerance), message)

// Helper function to compare arrays
int arrays_equal_float(float* arr1, float* arr2, size_t size, float tolerance) {
  for (size_t i = 0; i < size; i++) {
    if (fabs(arr1[i] - arr2[i]) > tolerance) {
      return 0;
    }
  }
  return 1;
}

// Helper function to get float value from any dtype array at index
float get_array_value_at_index(Array* arr, size_t index) {
  return dtype_to_float32(arr->data, arr->dtype, index);
}

// Test dtype functionality
void test_dtype_functions() {
  TEST_START("dtype functions");
  
  // Test dtype sizes
  ASSERT_EQUAL_INT(4, get_dtype_size(DTYPE_FLOAT32), "float32 size");
  ASSERT_EQUAL_INT(8, get_dtype_size(DTYPE_FLOAT64), "float64 size");
  ASSERT_EQUAL_INT(1, get_dtype_size(DTYPE_INT8), "int8 size");
  ASSERT_EQUAL_INT(2, get_dtype_size(DTYPE_INT16), "int16 size");
  ASSERT_EQUAL_INT(4, get_dtype_size(DTYPE_INT32), "int32 size");
  ASSERT_EQUAL_INT(8, get_dtype_size(DTYPE_INT64), "int64 size");
  ASSERT_EQUAL_INT(1, get_dtype_size(DTYPE_BOOL), "bool size");
  
  // Test dtype names
  ASSERT_TRUE(strcmp(get_dtype_name(DTYPE_FLOAT32), "float32") == 0, "float32 name");
  ASSERT_TRUE(strcmp(get_dtype_name(DTYPE_INT32), "int32") == 0, "int32 name");
  ASSERT_TRUE(strcmp(get_dtype_name(DTYPE_BOOL), "bool") == 0, "bool name");
  
  // Test type checking functions
  ASSERT_TRUE(is_float_dtype(DTYPE_FLOAT32), "float32 is float");
  ASSERT_TRUE(is_float_dtype(DTYPE_FLOAT64), "float64 is float");
  ASSERT_TRUE(!is_float_dtype(DTYPE_INT32), "int32 is not float");
  
  ASSERT_TRUE(is_integer_dtype(DTYPE_INT32), "int32 is integer");
  ASSERT_TRUE(is_integer_dtype(DTYPE_UINT32), "uint32 is integer");
  ASSERT_TRUE(!is_integer_dtype(DTYPE_FLOAT32), "float32 is not integer");
  
  ASSERT_TRUE(is_signed_dtype(DTYPE_INT32), "int32 is signed");
  ASSERT_TRUE(!is_signed_dtype(DTYPE_UINT32), "uint32 is not signed");
  ASSERT_TRUE(is_unsigned_dtype(DTYPE_UINT32), "uint32 is unsigned");
  
  // Test dtype promotion
  ASSERT_EQUAL_INT(DTYPE_FLOAT64, promote_dtypes(DTYPE_FLOAT32, DTYPE_FLOAT64), "float32 + float64 -> float64");
  ASSERT_EQUAL_INT(DTYPE_FLOAT32, promote_dtypes(DTYPE_INT32, DTYPE_FLOAT32), "int32 + float32 -> float32");
  ASSERT_EQUAL_INT(DTYPE_INT64, promote_dtypes(DTYPE_INT32, DTYPE_INT64), "int32 + int64 -> int64");
}

// Test array creation and basic properties
void test_array_creation() {
  TEST_START("array creation");
  
  // Test 1D array creation
  float data1d[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  int shape1d[] = {5};
  Array* arr1d = create_array(data1d, 1, shape1d, 5, DTYPE_FLOAT32);
  
  ASSERT_NOT_NULL(arr1d, "1D array creation");
  ASSERT_EQUAL_INT(1, arr1d->ndim, "1D array ndim");
  ASSERT_EQUAL_INT(5, arr1d->size, "1D array size");
  ASSERT_EQUAL_INT(5, arr1d->shape[0], "1D array shape[0]");
  ASSERT_EQUAL_INT(DTYPE_FLOAT32, arr1d->dtype, "1D array dtype");
  ASSERT_EQUAL_FLOAT(3.0f, get_array_value_at_index(arr1d, 2), 1e-6, "1D array data access");
  
  // Test 2D array creation
  float data2d[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  int shape2d[] = {2, 3};
  Array* arr2d = create_array(data2d, 2, shape2d, 6, DTYPE_FLOAT32);
  
  ASSERT_NOT_NULL(arr2d, "2D array creation");
  ASSERT_EQUAL_INT(2, arr2d->ndim, "2D array ndim");
  ASSERT_EQUAL_INT(6, arr2d->size, "2D array size");
  ASSERT_EQUAL_INT(2, arr2d->shape[0], "2D array shape[0]");
  ASSERT_EQUAL_INT(3, arr2d->shape[1], "2D array shape[1]");
  
  // Test 3D array creation
  float data3d[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  int shape3d[] = {2, 2, 2};
  Array* arr3d = create_array(data3d, 3, shape3d, 8, DTYPE_INT32);
  
  ASSERT_NOT_NULL(arr3d, "3D array creation with int32 dtype");
  ASSERT_EQUAL_INT(3, arr3d->ndim, "3D array ndim");
  ASSERT_EQUAL_INT(DTYPE_INT32, arr3d->dtype, "3D array dtype");
  
  // Cleanup
  delete_array(arr1d);
  delete_array(arr2d);
  delete_array(arr3d);
}

// Test dtype casting
void test_dtype_casting() {
  TEST_START("dtype casting");
  
  float data[] = {1.7f, 2.3f, -3.8f, 4.1f};
  int shape[] = {4};
  Array* float_arr = create_array(data, 1, shape, 4, DTYPE_FLOAT32);
  
  // Test casting to int32
  Array* int_arr = cast_array(float_arr, DTYPE_INT32);
  ASSERT_NOT_NULL(int_arr, "cast to int32");
  ASSERT_EQUAL_INT(DTYPE_INT32, int_arr->dtype, "cast result dtype");
  ASSERT_EQUAL_FLOAT(1.0f, get_array_value_at_index(int_arr, 0), 1e-6, "cast value 1.7->1");
  ASSERT_EQUAL_FLOAT(2.0f, get_array_value_at_index(int_arr, 1), 1e-6, "cast value 2.3->2");
  
  // Test casting to bool
  Array* bool_arr = cast_array(float_arr, DTYPE_BOOL);
  ASSERT_NOT_NULL(bool_arr, "cast to bool");
  ASSERT_EQUAL_INT(DTYPE_BOOL, bool_arr->dtype, "bool cast dtype");
  
  // Test simple casting
  Array* simple_cast = cast_array_simple(float_arr, DTYPE_FLOAT64);
  ASSERT_NOT_NULL(simple_cast, "simple cast to float64");
  ASSERT_EQUAL_INT(DTYPE_FLOAT64, simple_cast->dtype, "simple cast dtype");
  
  // Cleanup
  delete_array(float_arr);
  delete_array(int_arr);
  delete_array(bool_arr);
  delete_array(simple_cast);
}

// Test array creation functions
void test_array_creation_functions() {
  TEST_START("array creation functions");
  
  int shape[] = {3, 4};
  size_t size = 12;
  
  // Test zeros array
  Array* zeros = zeros_array(shape, size, 2, DTYPE_FLOAT32);
  ASSERT_NOT_NULL(zeros, "zeros array creation");
  ASSERT_EQUAL_FLOAT(0.0f, get_array_value_at_index(zeros, 0), 1e-6, "zeros array value");
  ASSERT_EQUAL_FLOAT(0.0f, get_array_value_at_index(zeros, 5), 1e-6, "zeros array value");
  
  // Test ones array
  Array* ones = ones_array(shape, size, 2, DTYPE_FLOAT32);
  ASSERT_NOT_NULL(ones, "ones array creation");
  ASSERT_EQUAL_FLOAT(1.0f, get_array_value_at_index(ones, 0), 1e-6, "ones array value");
  ASSERT_EQUAL_FLOAT(1.0f, get_array_value_at_index(ones, 7), 1e-6, "ones array value");
  
  // Test zeros_like
  Array* zeros_like = zeros_like_array(ones);
  ASSERT_NOT_NULL(zeros_like, "zeros_like array creation");
  ASSERT_EQUAL_INT(ones->ndim, zeros_like->ndim, "zeros_like same ndim");
  ASSERT_EQUAL_INT(ones->size, zeros_like->size, "zeros_like same size");
  ASSERT_EQUAL_FLOAT(0.0f, get_array_value_at_index(zeros_like, 3), 1e-6, "zeros_like value");
  
  // Test ones_like
  Array* ones_like = ones_like_array(zeros);
  ASSERT_NOT_NULL(ones_like, "ones_like array creation");
  ASSERT_EQUAL_FLOAT(1.0f, get_array_value_at_index(ones_like, 8), 1e-6, "ones_like value");
  
  // Test fill array
  Array* filled = fill_array(5.5f, shape, size, 2, DTYPE_FLOAT32);
  ASSERT_NOT_NULL(filled, "fill array creation");
  ASSERT_EQUAL_FLOAT(5.5f, get_array_value_at_index(filled, 2), 1e-6, "fill array value");
  
  // Test random arrays (just check they're created, can't test exact values)
  Array* randn = randn_array(shape, size, 2, DTYPE_FLOAT32);
  ASSERT_NOT_NULL(randn, "randn array creation");
  
  Array* randint = randint_array(1, 10, shape, size, 2, DTYPE_INT32);
  ASSERT_NOT_NULL(randint, "randint array creation");
  ASSERT_EQUAL_INT(DTYPE_INT32, randint->dtype, "randint dtype");
  
  Array* uniform = uniform_array(0, 1, shape, size, 2, DTYPE_FLOAT32);
  ASSERT_NOT_NULL(uniform, "uniform array creation");
  
  // Test linspace
  int shape1d[] = {5};
  Array* linspace = linspace_array(0.0f, 5.0f, 4.0f, shape1d, 5, 1, DTYPE_FLOAT32);
  ASSERT_NOT_NULL(linspace, "linspace array creation");
  ASSERT_EQUAL_FLOAT(0.0f, get_array_value_at_index(linspace, 0), 1e-6, "linspace start");
  ASSERT_EQUAL_FLOAT(4.0f, get_array_value_at_index(linspace, 4), 1e-6, "linspace end");
  
  // Cleanup
  delete_array(zeros);
  delete_array(ones);
  delete_array(zeros_like);
  delete_array(ones_like);
  delete_array(filled);
  delete_array(randn);
  delete_array(randint);
  delete_array(uniform);
  delete_array(linspace);
}

// Test binary operations
void test_binary_operations() {
  TEST_START("binary operations");
  
  float data1[] = {1.0f, 2.0f, 3.0f, 4.0f};
  float data2[] = {5.0f, 6.0f, 7.0f, 8.0f};
  int shape[] = {4};
  
  Array* arr1 = create_array(data1, 1, shape, 4, DTYPE_FLOAT32);
  Array* arr2 = create_array(data2, 1, shape, 4, DTYPE_FLOAT32);
  
  // Test addition
  Array* add_result = add_array(arr1, arr2);
  ASSERT_NOT_NULL(add_result, "array addition");
  ASSERT_EQUAL_FLOAT(6.0f, get_array_value_at_index(add_result, 0), 1e-6, "add result [0]");
  ASSERT_EQUAL_FLOAT(12.0f, get_array_value_at_index(add_result, 3), 1e-6, "add result [3]");
  
  // Test scalar addition
  Array* add_scalar_result = add_scalar_array(arr1, 10.0f);
  ASSERT_NOT_NULL(add_scalar_result, "scalar addition");
  ASSERT_EQUAL_FLOAT(11.0f, get_array_value_at_index(add_scalar_result, 0), 1e-6, "scalar add result");
  
  // Test subtraction
  Array* sub_result = sub_array(arr2, arr1);
  ASSERT_NOT_NULL(sub_result, "array subtraction");
  ASSERT_EQUAL_FLOAT(4.0f, get_array_value_at_index(sub_result, 0), 1e-6, "sub result [0]");
  
  // Test multiplication
  Array* mul_result = mul_array(arr1, arr2);
  ASSERT_NOT_NULL(mul_result, "array multiplication");
  ASSERT_EQUAL_FLOAT(5.0f, get_array_value_at_index(mul_result, 0), 1e-6, "mul result [0]");
  ASSERT_EQUAL_FLOAT(32.0f, get_array_value_at_index(mul_result, 3), 1e-6, "mul result [3]");
  
  // Test division
  Array* div_result = div_array(arr2, arr1);
  ASSERT_NOT_NULL(div_result, "array division");
  ASSERT_EQUAL_FLOAT(5.0f, get_array_value_at_index(div_result, 0), 1e-6, "div result [0]");
  ASSERT_EQUAL_FLOAT(2.0f, get_array_value_at_index(div_result, 3), 1e-6, "div result [3]");
  
  // Test scalar operations
  Array* mul_scalar_result = mul_scalar_array(arr1, 3.0f);
  ASSERT_NOT_NULL(mul_scalar_result, "scalar multiplication");
  ASSERT_EQUAL_FLOAT(3.0f, get_array_value_at_index(mul_scalar_result, 0), 1e-6, "scalar mul result");
  
  Array* div_scalar_result = div_scalar_array(arr2, 2.0f);
  ASSERT_NOT_NULL(div_scalar_result, "scalar division");
  ASSERT_EQUAL_FLOAT(2.5f, get_array_value_at_index(div_scalar_result, 0), 1e-6, "scalar div result");
  
  // Cleanup
  delete_array(arr1);
  delete_array(arr2);
  delete_array(add_result);
  delete_array(add_scalar_result);
  delete_array(sub_result);
  delete_array(mul_result);
  delete_array(div_result);
  delete_array(mul_scalar_result);
  delete_array(div_scalar_result);
}

// Test unary operations
void test_unary_operations() {
  TEST_START("unary operations");
  
  float data[] = {0.0f, C_PI/6, C_PI/4, C_PI/2};
  int shape[] = {4};
  Array* arr = create_array(data, 1, shape, 4, DTYPE_FLOAT32);
  
  // Test trigonometric functions
  Array* sin_result = sin_array(arr);
  ASSERT_NOT_NULL(sin_result, "sin array");
  ASSERT_EQUAL_FLOAT(0.0f, get_array_value_at_index(sin_result, 0), 1e-6, "sin(0)");
  ASSERT_EQUAL_FLOAT(1.0f, get_array_value_at_index(sin_result, 3), 1e-6, "sin(π/2)");
  
  Array* cos_result = cos_array(arr);
  ASSERT_NOT_NULL(cos_result, "cos array");
  ASSERT_EQUAL_FLOAT(1.0f, get_array_value_at_index(cos_result, 0), 1e-6, "cos(0)");
  
  Array* tan_result = tan_array(arr);
  ASSERT_NOT_NULL(tan_result, "tan array");
  ASSERT_EQUAL_FLOAT(0.0f, get_array_value_at_index(tan_result, 0), 1e-6, "tan(0)");
  
  // Test hyperbolic functions
  Array* sinh_result = sinh_array(arr);
  ASSERT_NOT_NULL(sinh_result, "sinh array");
  
  Array* cosh_result = cosh_array(arr);
  ASSERT_NOT_NULL(cosh_result, "cosh array");
  ASSERT_EQUAL_FLOAT(1.0f, get_array_value_at_index(cosh_result, 0), 1e-6, "cosh(0)");
  
  Array* tanh_result = tanh_array(arr);
  ASSERT_NOT_NULL(tanh_result, "tanh array");
  ASSERT_EQUAL_FLOAT(0.0f, get_array_value_at_index(tanh_result, 0), 1e-6, "tanh(0)");
  
  // Test power operations
  float pow_data[] = {2.0f, 3.0f, 4.0f, 5.0f};
  Array* pow_arr = create_array(pow_data, 1, shape, 4, DTYPE_FLOAT32);
  
  Array* pow_result = pow_array(pow_arr, 2.0f);
  ASSERT_NOT_NULL(pow_result, "power array");
  ASSERT_EQUAL_FLOAT(4.0f, get_array_value_at_index(pow_result, 0), 1e-6, "2^2");
  ASSERT_EQUAL_FLOAT(25.0f, get_array_value_at_index(pow_result, 3), 1e-6, "5^2");
  
  Array* pow_scalar_result = pow_scalar(2.0f, pow_arr);
  ASSERT_NOT_NULL(pow_scalar_result, "scalar power array");
  ASSERT_EQUAL_FLOAT(4.0f, get_array_value_at_index(pow_scalar_result, 0), 1e-6, "2^2");
  ASSERT_EQUAL_FLOAT(32.0f, get_array_value_at_index(pow_scalar_result, 3), 1e-6, "2^5");
  
  // Cleanup
  delete_array(arr);
  delete_array(pow_arr);
  delete_array(sin_result);
  delete_array(cos_result);
  delete_array(tan_result);
  delete_array(sinh_result);
  delete_array(cosh_result);
  delete_array(tanh_result);
  delete_array(pow_result);
  delete_array(pow_scalar_result);
}

// Test shaping operations
void test_shaping_operations() {
  TEST_START("shaping operations");
  
  float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  int shape[] = {2, 3};
  Array* arr = create_array(data, 2, shape, 6, DTYPE_FLOAT32);
  
  // Test transpose
  Array* transposed = transpose_array(arr);
  ASSERT_NOT_NULL(transposed, "transpose array");
  ASSERT_EQUAL_INT(2, transposed->ndim, "transpose ndim");
  ASSERT_EQUAL_INT(3, transposed->shape[0], "transpose shape[0]");
  ASSERT_EQUAL_INT(2, transposed->shape[1], "transpose shape[1]");
  
  // Test reshape
  int new_shape[] = {3, 2};
  Array* reshaped = reshape_array(arr, new_shape, 2);
  ASSERT_NOT_NULL(reshaped, "reshape array");
  ASSERT_EQUAL_INT(3, reshaped->shape[0], "reshape shape[0]");
  ASSERT_EQUAL_INT(2, reshaped->shape[1], "reshape shape[1]");
  
  // Test flatten
  Array* flattened = flatten_array(arr);
  ASSERT_NOT_NULL(flattened, "flatten array");
  ASSERT_EQUAL_INT(1, flattened->ndim, "flatten ndim");
  ASSERT_EQUAL_INT(6, flattened->shape[0], "flatten shape");
  
  // Test expand_dims
  Array* expanded = expand_dims_array(flattened, 0);
  ASSERT_NOT_NULL(expanded, "expand_dims array");
  ASSERT_EQUAL_INT(2, expanded->ndim, "expand_dims ndim");
  ASSERT_EQUAL_INT(1, expanded->shape[0], "expand_dims new axis");
  
  // Test squeeze (assuming it removes dimensions of size 1)
  Array* squeezed = squeeze_array(expanded, 0);
  ASSERT_NOT_NULL(squeezed, "squeeze array");
  ASSERT_EQUAL_INT(1, squeezed->ndim, "squeeze ndim");
  
  // Test equality
  float data2[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  Array* arr2 = create_array(data2, 2, shape, 6, DTYPE_FLOAT32);
  Array* equal_result = equal_array(arr, arr2);
  ASSERT_NOT_NULL(equal_result, "equal arrays");
  
  // Cleanup
  delete_array(arr);
  delete_array(arr2);
  delete_array(transposed);
  delete_array(reshaped);
  delete_array(flattened);
  delete_array(expanded);
  delete_array(squeezed);
  delete_array(equal_result);
}

// Test reduction operations
void test_reduction_operations() {
  TEST_START("reduction operations");
  
  float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  int shape[] = {2, 3};
  Array* arr = create_array(data, 2, shape, 6, DTYPE_FLOAT32);
  
  // Test sum
  Array* sum_all = sum_array(arr, -1, false);  // Sum all elements
  ASSERT_NOT_NULL(sum_all, "sum all elements");
  ASSERT_EQUAL_FLOAT(21.0f, get_array_value_at_index(sum_all, 0), 1e-6, "sum result");
  
  // Test mean
  Array* mean_all = mean_array(arr, -1, false);
  ASSERT_NOT_NULL(mean_all, "mean all elements");
  ASSERT_EQUAL_FLOAT(3.5f, get_array_value_at_index(mean_all, 0), 1e-6, "mean result");
  
  // Test max
  Array* max_all = max_array(arr, -1, false);
  ASSERT_NOT_NULL(max_all, "max all elements");
  ASSERT_EQUAL_FLOAT(6.0f, get_array_value_at_index(max_all, 0), 1e-6, "max result");
  
  // Test min
  Array* min_all = min_array(arr, -1, false);
  ASSERT_NOT_NULL(min_all, "min all elements");
  ASSERT_EQUAL_FLOAT(1.0f, get_array_value_at_index(min_all, 0), 1e-6, "min result");
  
  // Test variance
  Array* var_all = var_array(arr, -1, 0);  // Population variance
  ASSERT_NOT_NULL(var_all, "variance all elements");
  
  // Test standard deviation
  Array* std_all = std_array(arr, -1, 0);
  ASSERT_NOT_NULL(std_all, "std all elements");
  
  // Cleanup
  delete_array(arr);
  delete_array(sum_all);
  delete_array(mean_all);
  delete_array(max_all);
  delete_array(min_all);
  delete_array(var_all);
  delete_array(std_all);
}

// Test memory management and edge cases
void test_memory_management() {
  TEST_START("memory management");
  
  // Test array deletion functions
  float data[] = {1.0f, 2.0f, 3.0f};
  int shape[] = {3};
  Array* arr = create_array(data, 1, shape, 3, DTYPE_FLOAT32);
  
  ASSERT_NOT_NULL(arr, "array creation for memory test");
  ASSERT_NOT_NULL(arr->data, "array data exists");
  ASSERT_NOT_NULL(arr->shape, "array shape exists");
  ASSERT_NOT_NULL(arr->strides, "array strides exist");
  
  // Test individual deletion functions (these should not crash)
  Array* test_arr = create_array(data, 1, shape, 3, DTYPE_FLOAT32);
  delete_data(test_arr);
  ASSERT_TRUE(test_arr->data == NULL, "delete_data nullifies data");
  
  Array* test_arr2 = create_array(data, 1, shape, 3, DTYPE_FLOAT32);
  delete_shape(test_arr2);
  ASSERT_TRUE(test_arr2->shape == NULL, "delete_shape nullifies shape");
  
  Array* test_arr3 = create_array(data, 1, shape, 3, DTYPE_FLOAT32);
  delete_strides(test_arr3);
  ASSERT_TRUE(test_arr3->strides == NULL && test_arr3->backstrides == NULL, "delete_strides nullifies strides");
  
  // Clean up test arrays (some fields already freed)
  free(test_arr);
  free(test_arr2);
  free(test_arr3);
  
  // Test normal cleanup
  delete_array(arr);
  ASSERT_TRUE(1, "array deletion completed without crash");
}

// Test print functionality (visual test)
void test_print_functionality() {
  TEST_START("print functionality");
  
  printf("Testing array printing (visual inspection):\n");
  
  // 1D array
  float data1d[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  int shape1d[] = {5};
  Array* arr1d = create_array(data1d, 1, shape1d, 5, DTYPE_FLOAT32);
  printf("1D array: ");
  print_array(arr1d);
  
  // 2D array
  float data2d[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  int shape2d[] = {2, 3};
  Array* arr2d = create_array(data2d, 2, shape2d, 6, DTYPE_INT32);
  printf("2D array: ");
  print_array(arr2d);
  
  // NULL array
  printf("NULL array: ");
  print_array(NULL);
  
  ASSERT_TRUE(1, "print functions completed without crash");
  
  delete_array(arr1d);
  delete_array(arr2d);
}

// Test different dtype scenarios
void test_mixed_dtype_operations() {
  TEST_START("mixed dtype operations");
  
  // Create arrays with different dtypes
  float float_data[] = {1.5f, 2.7f, 3.9f, 4.1f};
  float int_data[] = {10.0f, 20.0f, 30.0f, 40.0f};
  int shape[] = {4};
  
  Array* float_arr = create_array(float_data, 1, shape, 4, DTYPE_FLOAT32);
  Array* int_arr = create_array(int_data, 1, shape, 4, DTYPE_INT32);
  
  // Test operations between different dtypes
  Array* mixed_add = add_array(float_arr, int_arr);
  ASSERT_NOT_NULL(mixed_add, "mixed dtype addition");
  ASSERT_EQUAL_FLOAT(11.5f, get_array_value_at_index(mixed_add, 0), 1e-6, "mixed add result");
  
  // Test casting chain
  Array* bool_arr = cast_array(float_arr, DTYPE_BOOL);
  Array* back_to_float = cast_array(bool_arr, DTYPE_FLOAT64);
  ASSERT_NOT_NULL(back_to_float, "casting chain");
  ASSERT_EQUAL_INT(DTYPE_FLOAT64, back_to_float->dtype, "final dtype after casting chain");
  
  // Test with uint types
  Array* uint_arr = cast_array(int_arr, DTYPE_UINT32);
  ASSERT_NOT_NULL(uint_arr, "cast to uint32");
  ASSERT_EQUAL_INT(DTYPE_UINT32, uint_arr->dtype, "uint32 dtype");
  
  // Cleanup
  delete_array(float_arr);
  delete_array(int_arr);
  delete_array(mixed_add);
  delete_array(bool_arr);
  delete_array(back_to_float);
  delete_array(uint_arr);
}

// Test edge cases and error conditions
void test_edge_cases() {
  TEST_START("edge cases");
  
  // Test with very small arrays
  float single_data[] = {42.0f};
  int single_shape[] = {1};
  Array* single_arr = create_array(single_data, 1, single_shape, 1, DTYPE_FLOAT32);
  ASSERT_NOT_NULL(single_arr, "single element array");
  ASSERT_EQUAL_FLOAT(42.0f, get_array_value_at_index(single_arr, 0), 1e-6, "single element value");
  
  // Test operations on single element
  Array* single_sin = sin_array(single_arr);
  ASSERT_NOT_NULL(single_sin, "sin on single element");
  
  Array* single_add = add_scalar_array(single_arr, 8.0f);
  ASSERT_NOT_NULL(single_add, "scalar add on single element");
  ASSERT_EQUAL_FLOAT(50.0f, get_array_value_at_index(single_add, 0), 1e-6, "single element scalar add");
  
  // Test with zeros
  float zero_data[] = {0.0f, 0.0f, 0.0f};
  int zero_shape[] = {3};
  Array* zero_arr = create_array(zero_data, 1, zero_shape, 3, DTYPE_FLOAT32);
  
  Array* zero_sin = sin_array(zero_arr);
  ASSERT_NOT_NULL(zero_sin, "sin of zeros");
  ASSERT_EQUAL_FLOAT(0.0f, get_array_value_at_index(zero_sin, 1), 1e-6, "sin(0) = 0");
  
  // Test with negative values
  float neg_data[] = {-1.0f, -2.0f, -3.0f};
  Array* neg_arr = create_array(neg_data, 1, zero_shape, 3, DTYPE_FLOAT32);
  
  Array* neg_abs = pow_array(neg_arr, 2.0f);  // Square to get positive
  ASSERT_NOT_NULL(neg_abs, "operations on negative values");
  ASSERT_EQUAL_FLOAT(4.0f, get_array_value_at_index(neg_abs, 1), 1e-6, "(-2)^2 = 4");
  
  // Test large numbers
  float large_data[] = {1000.0f, 10000.0f, 100000.0f};
  Array* large_arr = create_array(large_data, 1, zero_shape, 3, DTYPE_FLOAT32);
  
  Array* large_div = div_scalar_array(large_arr, 1000.0f);
  ASSERT_NOT_NULL(large_div, "operations on large numbers");
  ASSERT_EQUAL_FLOAT(1.0f, get_array_value_at_index(large_div, 0), 1e-6, "1000/1000 = 1");
  
  // Cleanup
  delete_array(single_arr);
  delete_array(single_sin);
  delete_array(single_add);
  delete_array(zero_arr);
  delete_array(zero_sin);
  delete_array(neg_arr);
  delete_array(neg_abs);
  delete_array(large_arr);
  delete_array(large_div);
}

// Test broadcasting scenarios
void test_broadcasting_operations() {
  TEST_START("broadcasting operations");
  
  // Create arrays of different shapes for broadcasting
  float data1[] = {1.0f, 2.0f, 3.0f, 4.0f};
  float data2[] = {10.0f, 20.0f};
  int shape1[] = {2, 2};
  int shape2[] = {2};
  
  Array* arr1 = create_array(data1, 2, shape1, 4, DTYPE_FLOAT32);
  Array* arr2 = create_array(data2, 1, shape2, 2, DTYPE_FLOAT32);
  
  // Test broadcasted operations
  Array* broadcast_add = add_broadcasted_array(arr1, arr2);
  ASSERT_NOT_NULL(broadcast_add, "broadcasted addition");
  
  Array* broadcast_mul = mul_broadcasted_array(arr1, arr2);
  ASSERT_NOT_NULL(broadcast_mul, "broadcasted multiplication");
  
  Array* broadcast_sub = sub_broadcasted_array(arr1, arr2);
  ASSERT_NOT_NULL(broadcast_sub, "broadcasted subtraction");
  
  Array* broadcast_div = div_broadcasted_array(arr1, arr2);
  ASSERT_NOT_NULL(broadcast_div, "broadcasted division");
  
  // Cleanup
  delete_array(arr1);
  delete_array(arr2);
  delete_array(broadcast_add);
  delete_array(broadcast_mul);
  delete_array(broadcast_sub);
  delete_array(broadcast_div);
}

// Test performance with larger arrays
void test_performance() {
  TEST_START("performance test");
  
  // Create a larger array for performance testing
  const size_t large_size = 10000;
  float* large_data = (float*)malloc(large_size * sizeof(float));
  
  // Fill with test data
  for (size_t i = 0; i < large_size; i++) {
    large_data[i] = (float)(i % 100) / 10.0f;
  }
  
  int large_shape[] = {100, 100};
  Array* large_arr = create_array(large_data, 2, large_shape, large_size, DTYPE_FLOAT32);
  
  ASSERT_NOT_NULL(large_arr, "large array creation");
  ASSERT_EQUAL_INT(large_size, large_arr->size, "large array size");
  
  // Test operations on large array
  Array* large_sin = sin_array(large_arr);
  ASSERT_NOT_NULL(large_sin, "sin on large array");
  
  Array* large_add = add_scalar_array(large_arr, 1.0f);
  ASSERT_NOT_NULL(large_add, "scalar add on large array");
  
  Array* large_sum = sum_array(large_arr, -1, false);
  ASSERT_NOT_NULL(large_sum, "sum of large array");
  
  // Test dtype conversion on large array
  Array* large_int = cast_array(large_arr, DTYPE_INT32);
  ASSERT_NOT_NULL(large_int, "large array dtype conversion");
  
  printf("Large array operations completed successfully\n");
  
  // Cleanup
  free(large_data);
  delete_array(large_arr);
  delete_array(large_sin);
  delete_array(large_add);
  delete_array(large_sum);
  delete_array(large_int);
}

// Test multidimensional operations
void test_multidimensional_ops() {
  TEST_START("multidimensional operations");
  
  // Create 3D array
  float data3d[] = {
    1.0f, 2.0f, 3.0f, 4.0f,
    5.0f, 6.0f, 7.0f, 8.0f,
    9.0f, 10.0f, 11.0f, 12.0f
  };
  int shape3d[] = {3, 2, 2};
  Array* arr3d = create_array(data3d, 3, shape3d, 12, DTYPE_FLOAT32);
  
  ASSERT_NOT_NULL(arr3d, "3D array creation");
  ASSERT_EQUAL_INT(3, arr3d->ndim, "3D array dimensions");
  
  // Test operations on 3D array
  Array* arr3d_sin = sin_array(arr3d);
  ASSERT_NOT_NULL(arr3d_sin, "3D array sin operation");
  
  Array* arr3d_flat = flatten_array(arr3d);
  ASSERT_NOT_NULL(arr3d_flat, "3D array flattening");
  ASSERT_EQUAL_INT(1, arr3d_flat->ndim, "flattened array is 1D");
  ASSERT_EQUAL_INT(12, arr3d_flat->size, "flattened array size");
  
  // Test reshaping 3D array
  int new_shape_2d[] = {4, 3};
  Array* reshaped_3d = reshape_array(arr3d, new_shape_2d, 2);
  ASSERT_NOT_NULL(reshaped_3d, "3D to 2D reshape");
  ASSERT_EQUAL_INT(2, reshaped_3d->ndim, "reshaped array dimensions");
  ASSERT_EQUAL_INT(4, reshaped_3d->shape[0], "reshaped array shape[0]");
  ASSERT_EQUAL_INT(3, reshaped_3d->shape[1], "reshaped array shape[1]");
  
  // Test axis-specific reductions on 3D array
  Array* sum_axis0 = sum_array(arr3d, 0, false);
  ASSERT_NOT_NULL(sum_axis0, "3D array sum along axis 0");
  
  Array* mean_axis1 = mean_array(arr3d, 1, true);  // with keepdims
  ASSERT_NOT_NULL(mean_axis1, "3D array mean along axis 1 with keepdims");
  
  // Cleanup
  delete_array(arr3d);
  delete_array(arr3d_sin);
  delete_array(arr3d_flat);
  delete_array(reshaped_3d);
  delete_array(sum_axis0);
  delete_array(mean_axis1);
}

// Main test runner
int main() {
  printf("=== Array Library Test Suite ===\n");
  printf("Running comprehensive tests for the array library...\n\n");
  
  // Run all test suites
  test_dtype_functions();
  test_array_creation();
  test_dtype_casting();
  test_array_creation_functions();
  test_binary_operations();
  test_unary_operations();
  test_shaping_operations();
  test_reduction_operations();
  test_memory_management();
  test_mixed_dtype_operations();
  test_edge_cases();
  test_broadcasting_operations();
  test_performance();
  test_multidimensional_ops();
  test_print_functionality();
  
  // Print final results
  printf("\n=== Test Results ===\n");
  printf("Tests Passed: %d\n", tests_passed);
  printf("Tests Failed: %d\n", tests_failed);
  printf("Total Tests: %d\n", tests_passed + tests_failed);
  
  if (tests_failed == 0) {
    printf("🎉 All tests passed! The array library is working correctly.\n");
    return 0;
  } else {
    printf("❌ Some tests failed. Please check the implementation.\n");
    printf("Success Rate: %.1f%%\n", (float)tests_passed / (tests_passed + tests_failed) * 100);
    return 1;
  }
}