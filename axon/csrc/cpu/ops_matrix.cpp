#include <algorithm>
#include <string.h>
#include <math.h>
#include "ops_matrix.h"

void inv_ops(float* a, float* out, int* shape) {
  int n = shape[0];
  int size = n * n; 
  memcpy(out, a, size * sizeof(float));
  float* temp = new float[n * n * 2];
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      temp[i * n * 2 + j] = out[i * n + j];
      temp[i * n * 2 + j + n] = (i == j) ? 1.0f : 0.0f;
    }
  }

  for (int i = 0; i < n; i++) {
    int pivot = i;
    for (int k = i + 1; k < n; k++) {
      if (fabs(temp[k * n * 2 + i]) > fabs(temp[pivot * n * 2 + i])) pivot = k;
    }

    if (pivot != i) {
      for (int j = 0; j < n * 2; j++) {
        float t = temp[i * n * 2 + j];
        temp[i * n * 2 + j] = temp[pivot * n * 2 + j];
        temp[pivot * n * 2 + j] = t;
      }
    }

    float diag = temp[i * n * 2 + i];
    for (int j = 0; j < n * 2; j++) temp[i * n * 2 + j] /= diag;
    for (int k = 0; k < n; k++) {
      if (k != i) {
        float factor = temp[k * n * 2 + i];
        for (int j = 0; j < n * 2; j++) temp[k * n * 2 + j] -= factor * temp[i * n * 2 + j];
      }
    }
  }
  
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) out[i * n + j] = temp[i * n * 2 + j + n];
  }

  delete[] temp;
}

void batched_inv_ops(float* a, float* out, int* shape, int ndim) {
  if (ndim < 2) return;  
  int batch_size = 1;
  for (int i = 0; i < ndim - 2; i++) batch_size *= shape[i];
  int matrix_size = shape[ndim - 2] * shape[ndim - 1];
  int matrix_shape[2] = {shape[ndim - 2], shape[ndim - 1]};

  for (int b = 0; b < batch_size; b++) inv_ops(a + b * matrix_size, out + b * matrix_size, matrix_shape);
}

void matrix_rank_ops(float* a, float* out, int* shape) {
  int m = shape[0], n = shape[1];
  int size = m * n;

  float* temp = new float[size];
  memcpy(temp, a, size * sizeof(float));
  int rank = 0;
  float eps = 1e-10f;

  for (int col = 0, row = 0; col < n && row < m; col++) {
    int pivot = row;
    for (int i = row + 1; i < m; i++) {
      if (fabs(temp[i * n + col]) > fabs(temp[pivot * n + col])) pivot = i;
    }
    if (fabs(temp[pivot * n + col]) < eps) continue;
    if (pivot != row) {
      for (int j = 0; j < n; j++) {
        float t = temp[row * n + j];
        temp[row * n + j] = temp[pivot * n + j];
        temp[pivot * n + j] = t;
      }
    }
    rank++;
    for (int i = row + 1; i < m; i++) {
      if (fabs(temp[i * n + col]) > eps) {
        float factor = temp[i * n + col] / temp[row * n + col];
        for (int j = col; j < n; j++) temp[i * n + j] -= factor * temp[row * n + j];
      }
    }
    row++;
  }

  *out = (float)rank;
  delete[] temp;
}

void batched_matrix_rank_ops(float* a, float* out, int* shape, int ndim) {
  if (ndim < 2) return;

  int batch_size = 1;
  for (int i = 0; i < ndim - 2; i++) batch_size *= shape[i];
  int matrix_size = shape[ndim - 2] * shape[ndim - 1];
  int matrix_shape[2] = {shape[ndim - 2], shape[ndim - 1]};
  for (int b = 0; b < batch_size; b++) matrix_rank_ops(a + b * matrix_size, out + b, matrix_shape);
}

void solve_ops(float* a, float* b, float* out, int* shape_a, int* shape_b) {
  int n = shape_a[0];
  int nrhs = (shape_b[1] > 0) ? shape_b[1] : 1;
  float *temp_a = new float[n * n], *temp_b = new float[n * nrhs];
  memcpy(temp_a, a, n * n * sizeof(float));
  memcpy(temp_b, b, n * nrhs * sizeof(float));
  int* piv = new int[n];
  for (int i = 0; i < n; i++) {
    int pivot = i;
    for (int k = i + 1; k < n; k++) {
      if (fabs(temp_a[k * n + i]) > fabs(temp_a[pivot * n + i])) pivot = k;
    }
    piv[i] = pivot;

    if (pivot != i) {
      for (int j = 0; j < n; j++) {
        float t = temp_a[i * n + j];
        temp_a[i * n + j] = temp_a[pivot * n + j];
        temp_a[pivot * n + j] = t;
      }
      for (int j = 0; j < nrhs; j++) {
        float t = temp_b[i * nrhs + j];
        temp_b[i * nrhs + j] = temp_b[pivot * nrhs + j];
        temp_b[pivot * nrhs + j] = t;
      }
    }
    
    for (int k = i + 1; k < n; k++) {
      float factor = temp_a[k * n + i] / temp_a[i * n + i];
      for (int j = i + 1; j < n; j++) temp_a[k * n + j] -= factor * temp_a[i * n + j];
      for (int j = 0; j < nrhs; j++) temp_b[k * nrhs + j] -= factor * temp_b[i * nrhs + j];
    }
  }
  
  for (int j = 0; j < nrhs; j++) {
    for (int i = n - 1; i >= 0; i--) {
      float sum = temp_b[i * nrhs + j];
      for (int k = i + 1; k < n; k++) sum -= temp_a[i * n + k] * out[k * nrhs + j];
      out[i * nrhs + j] = sum / temp_a[i * n + i];
    }
  }
  
  delete[] temp_a;
  delete[] temp_b;
  delete[] piv;
}

void batched_solve_ops(float* a, float* b, float* out, int* shape_a, int* shape_b, int ndim) {
  if (ndim < 2) return;
  int batch_size = 1;
  for (int i = 0; i < ndim - 2; i++) batch_size *= shape_a[i];
  int matrix_size_a = shape_a[ndim - 2] * shape_a[ndim - 1], matrix_size_b = shape_b[ndim - 2] * shape_b[ndim - 1];
  int matrix_shape_a[2] = {shape_a[ndim - 2], shape_a[ndim - 1]}, matrix_shape_b[2] = {shape_b[ndim - 2], shape_b[ndim - 1]};
  for (int batch = 0; batch < batch_size; batch++) solve_ops(a + batch * matrix_size_a, b + batch * matrix_size_b, out + batch * matrix_size_b, matrix_shape_a, matrix_shape_b);
}

void lstsq_ops(float* a, float* b, float* out, int* shape_a, int* shape_b) {
  int m = shape_a[0];
  int n = shape_a[1];
  int nrhs = (shape_b[1] > 0) ? shape_b[1] : 1;
  
  float* at = new float[n * m];
  float* ata = new float[n * n];
  float* atb = new float[n * nrhs];
  
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) at[j * m + i] = a[i * n + j];
  }
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      float sum = 0.0f;
      for (int k = 0; k < m; k++) sum += at[i * m + k] * a[k * n + j];
      ata[i * n + j] = sum;
    }
  }
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < nrhs; j++) {
      float sum = 0.0f;
      for (int k = 0; k < m; k++) sum += at[i * m + k] * b[k * nrhs + j];
      atb[i * nrhs + j] = sum;
    }
  }

  int shape_ata[2] = {n, n}, shape_atb[2] = {n, nrhs};
  solve_ops(ata, atb, out, shape_ata, shape_atb);
  delete[] at;
  delete[] ata;
  delete[] atb;
}

void batched_lstsq_ops(float* a, float* b, float* out, int* shape_a, int* shape_b, int ndim) {
  if (ndim < 2) return;

  int batch_size = 1;
  for (int i = 0; i < ndim - 2; i++) batch_size *= shape_a[i];
  int matrix_size_a = shape_a[ndim - 2] * shape_a[ndim - 1], matrix_size_b = shape_b[ndim - 2] * shape_b[ndim - 1];
  int output_size = shape_a[ndim - 1] * shape_b[ndim - 1];
  int matrix_shape_a[2] = {shape_a[ndim - 2], shape_a[ndim - 1]}, matrix_shape_b[2] = {shape_b[ndim - 2], shape_b[ndim - 1]};
  for (int batch = 0; batch < batch_size; batch++) lstsq_ops(a + batch * matrix_size_a, b + batch * matrix_size_b, out + batch * output_size, matrix_shape_a, matrix_shape_b);
}