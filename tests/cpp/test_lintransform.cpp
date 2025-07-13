// g++ -o run test.cpp ../core/core.cpp ../core/dtype.cpp ../core/contiguous.cpp ../cpu/ops_array.cpp ../cpu/ops_shape.cpp transform.cpp

#include <stdio.h>
#include <stdlib.h>
#include "transform.h"
#include "../core/core.h"
#include "../core/dtype.h"
#include "../array_ops.h"

int main() {
  printf("Testing Linear Transform Functions\n");
  printf("==================================\n\n");

  // Test 1: 1D Linear Transform
  printf("Test 1: 1D Linear Transform (y = Wx + b)\n");

  // Weight matrix: 3x2 (3 output features, 2 input features)
  float w1_data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  int w1_shape[] = {3, 2};
  Array* weights1 = create_array(w1_data, 2, w1_shape, 6, DTYPE_FLOAT32);
  
  // Input vector: 2 features
  float x1_data[] = {1.0f, 2.0f};
  int x1_shape[] = {2};
  Array* input1 = create_array(x1_data, 1, x1_shape, 2, DTYPE_FLOAT32);

  // Bias vector: 3 features
  float b1_data[] = {0.1f, 0.2f, 0.3f};
  int b1_shape[] = {3};
  Array* bias1 = create_array(b1_data, 1, b1_shape, 3, DTYPE_FLOAT32);

  Array* result1 = linear_1d_array(weights1, input1, bias1);
  printf("input: \n");
  print_array(input1);
  printf("output: \n");
  print_array(result1);
  printf("Expected: [5.1, 11.2, 17.3] (1*1+2*2+0.1, 1*3+2*4+0.2, 1*5+2*6+0.3)\n\n");
  
  // Test 2: 2D Linear Transform (batch)
  printf("Test 2: 2D Linear Transform (Y = XW^T + b)\n");
  
  // Weight matrix: 2x3 (2 output features, 3 input features)
  float w2_data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  int w2_shape[] = {2, 3};
  Array* weights2 = create_array(w2_data, 2, w2_shape, 6, DTYPE_FLOAT32);

  // Input batch: 2x3 (2 samples, 3 features each)
  float x2_data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  int x2_shape[] = {2, 3};
  Array* input2 = create_array(x2_data, 2, x2_shape, 6, DTYPE_FLOAT32);

  float b2_data[] = {0.1f, 0.2f};
  int b2_shape[] = {2};
  Array* bias2 = create_array(b2_data, 1, b2_shape, 2, DTYPE_FLOAT32);
  Array* result2 = linear_2d_array(weights2, input2, bias2);
  // Array* result2 = matmul_array(weights2, input2);

  printf("Input batch shape: [%d, %d]\n", input2->shape[0], input2->shape[1]);
  printf("Weight shape: [%d, %d]\n", weights2->shape[0], weights2->shape[1]);
  printf("\n Output Batch: ");
  print_array(result2);
  printf("Expected: [14.1, 32.2, 32.1, 77.2]\n");
  printf("(Sample 1: [1*1+2*2+3*3+0.1, 1*4+2*5+3*6+0.2])\n");
  printf("(Sample 2: [4*1+5*2+6*3+0.1, 4*4+5*5+6*6+0.2])\n\n");
  
  // Test 3: General function dispatch
  printf("Test 3: General function dispatch\n");
  Array* result3 = linear_transform_array(weights1, input1, bias1);
  // Array* result3 = matmul_array(weights1, input1);
  printf("\n 1d dispatch: ");
  print_array(result3);

  Array* result4 = linear_transform_array(weights2, input2, bias2);
  printf("\n 2d dispatch: ");
  print_array(result4);

  printf("\nAll tests completed!\n");
  
  // Cleanup (in real code, you'd want proper cleanup functions)
  free(weights1->data); free(weights1->shape); free(weights1);
  free(input1->data); free(input1->shape); free(input1);
  free(bias1->data); free(bias1->shape); free(bias1);
  free(result1->data); free(result1->shape); free(result1);
  
  free(weights2->data); free(weights2->shape); free(weights2);
  free(input2->data); free(input2->shape); free(input2);
  free(bias2->data); free(bias2->shape); free(bias2);
  free(result2->data); free(result2->shape); free(result2);
  
  free(result3->data); free(result3->shape); free(result3);
  free(result4->data); free(result4->shape); free(result4);
  
  return 0;
}