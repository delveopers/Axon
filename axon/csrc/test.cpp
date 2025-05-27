#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include "array.h"

int main() {
  int* shape1 = (int*)malloc(2 * sizeof(int));
  int* shape2 = (int*)malloc(2 * sizeof(int));
  shape1[0] = 2, shape1[1] = 4;
  shape2[0] = 2, shape2[1] = 4;
  float* data1 = (float*)malloc(8 * sizeof(float));
  float* data2 = (float*)malloc(8 * sizeof(float));
  data1[0] = 2.0; data1[1] = 4.0; data1[2] = 5.0; data1[3] = -4.0;
  data1[4] = -3.0; data1[5] = 0.0; data1[6] = 9.0; data1[7] = -1.0;
  data2[0] = 1.0; data2[1] = 0.0; data2[2] = -2.0; data2[3] = 0.0;
  data2[4] = -1.0; data2[5] = 10.0; data2[6] = -2.0; data2[7] = 4.0;

  Array* array1 = create_array(data1, 2, shape1, 8);
  Array* array2 = create_array(data2, 2, shape2, 8);

  Array* c = add_array(array1, array2);
  Array* d = sub_array(array1, array2);
  Array* e = mul_array(array1, d);
  print_tensor(array1);
  print_tensor(array2);
  print_tensor(c);
  print_tensor(d);
  print_tensor(e);

  delete_array(array1);
  delete_array(array2);
  delete_array(c);
  delete_array(d);
  delete_array(e);
  return 0;
}