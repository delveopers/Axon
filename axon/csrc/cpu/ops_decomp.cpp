#include <stdlib.h>
#include <stddef.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include "ops_decomp.h"
#include "ops_array.h"
#include "ops_shape.h"

void det_ops_array(float* a, float* out, size_t size) {
  float det = 1.0f;
  float* temp = (float*)malloc(size * size * sizeof(float));
  if (!temp) { *out = 0.0f; return; }
  for (size_t i = 0; i < size * size; ++i) temp[i] = a[i];
  for (size_t i = 0; i < size; ++i) {   // gaussian elimination with partial pivoting
    // finding pivot
    size_t pivot_row = i;
    float max_val = fabsf(temp[i * size + i]);
    for (size_t row = i + 1; row < size; ++row) {
      float val = fabsf(temp[row * size + i]);
      if (val > max_val) { max_val = val; pivot_row = row; }
    }
    // swapping rows if needed
    if (pivot_row != i) {
      for (size_t col = 0; col < size; ++col) {
        float tmp = temp[i * size + col];
        temp[i * size + col] = temp[pivot_row * size + col];
        temp[pivot_row * size + col] = tmp;
      }
      det = -det; // row swap changes sign
    }
    float pivot = temp[i * size + i];
    if (fabsf(pivot) < 1e-6f) { det = 0.0f; break; }
    det *= pivot;
    // eliminating column
    for (size_t j = i + 1; j < size; ++j) {
      float factor = temp[j * size + i] / pivot;
      for (size_t k = i; k < size; ++k) temp[j * size + k] -= factor * temp[i * size + k];
    }
  }
  free(temp);
  *out = det;
}

void batched_det_ops(float* a, float* out, size_t size, size_t batch) {
  size_t mat_size = size * size;
  for (size_t b = 0; b < batch; ++b) {
    float* mat = &a[b * mat_size];
    det_ops_array(mat, &out[b], size);
  }
}

void qr_decomp_ops(float* a, float* q, float* r, int* shape) {
  int m = shape[0], n = shape[1];  // rows, cols
  float* work = (float*)malloc(m * n * sizeof(float));
  memcpy(work, a, m * n * sizeof(float));
  memset(q, 0, m * m * sizeof(float));  // initialize q as identity matrix (m x m)
  for (int i = 0; i < m; i++) q[i * m + i] = 1.0f;
  memset(r, 0, m * n * sizeof(float));    // initialize r as zero matrix (m x n)
  for (int k = 0; k < n && k < m; k++) {  // modified gram-schmidt process
    float norm = 0.0f;  // compute column norm for r[k][k]
    for (int i = 0; i < m; i++) {
      float val = work[i * n + k];
      norm += val * val;
    }
    norm = sqrtf(norm);
    r[k * n + k] = norm;

    // normalize column k to get q_k
    if (norm > 1e-6f) { for (int i = 0; i < m; i++) { q[i * m + k] = work[i * n + k] / norm; } }
    for (int j = k + 1; j < n; j++) { // orthogonalize remaining columns
      float dot = 0.0f; // compute r[k][j] = q_k^T * a_j (dot product)
      for (int i = 0; i < m; i++) dot += q[i * m + k] * work[i * n + j];
      r[k * n + j] = dot;
      for (int i = 0; i < m; i++) work[i * n + j] -= dot * q[i * m + k]; // subtract projection: a_j = a_j - r[k][j] * q_k
    }
  }  
  free(work);
}

// batched qr decomposition for n-dimensional arrays, processes matrices along the last two dimensions
void batched_qr_decomp_ops(float* a, float* q, float* r, int* shape, int ndim) {
  if (ndim < 2) {
    fprintf(stderr, "error: qr decomposition requires at least 2 dimensions\n");
    exit(EXIT_FAILURE);
  }
  // matrix dimensions are the last two
  int m = shape[ndim - 2], n = shape[ndim - 1]; // rows, cols
  int batch_size = 1; // compute batch size (product of all leading dimensions)
  for (int i = 0; i < ndim - 2; i++) { batch_size *= shape[i]; }
  int a_matrix_size = m * n, q_matrix_size = m * m, r_matrix_size = m * n;
  // process each matrix in the batch
  for (int batch = 0; batch < batch_size; batch++) {
    float *a_batch = a + batch * a_matrix_size, *q_batch = q + batch * q_matrix_size, *r_batch = r + batch * r_matrix_size;
    int matrix_shape[2] = {m, n};
    qr_decomp_ops(a_batch, q_batch, r_batch, matrix_shape);
  }
}

void lu_decomp_ops(float* a, float* l, float* u, int* p, int* shape) {
  int n = shape[0];  // assuming square matrix n x n
  memcpy(u, a, n * n * sizeof(float));    // copy input to u matrix
  memset(l, 0, n * n * sizeof(float));    // initialize l as identity matrix
  for (int i = 0; i < n; i++) l[i * n + i] = 1.0f;
  for (int i = 0; i < n; i++) p[i] = i; // initialize permutation array
  // gaussian elimination with partial pivoting
  for (int k = 0; k < n - 1; k++) {
    int pivot_row = k;
    float max_val = fabsf(u[k * n + k]);
    for (int i = k + 1; i < n; i++) {
      if (fabsf(u[i * n + k]) > max_val) {
        max_val = fabsf(u[i * n + k]);
        pivot_row = i;
      }
    }
    if (pivot_row != k) { // swap rows in u matrix
      for (int j = 0; j < n; j++) {
        float temp = u[k * n + j];
        u[k * n + j] = u[pivot_row * n + j];
        u[pivot_row * n + j] = temp;
      }
      for (int j = 0; j < k; j++) { // swap rows in l matrix (only lower part)
        float temp = l[k * n + j];
        l[k * n + j] = l[pivot_row * n + j];
        l[pivot_row * n + j] = temp;
      }
      int temp_p = p[k];  // update permutation
      p[k] = p[pivot_row];
      p[pivot_row] = temp_p;
    }
    for (int i = k + 1; i < n; i++) {
      if (fabsf(u[k * n + k]) > 1e-9f) {
        float factor = u[i * n + k] / u[k * n + k];
        l[i * n + k] = factor;
        for (int j = k; j < n; j++) u[i * n + j] -= factor * u[k * n + j];
      }
    }
  }
}

void batched_lu_decomp_ops(float* a, float* l, float* u, int* p, int* shape, int ndim) {
  if (ndim < 2) {
    fprintf(stderr, "error: lu decomposition requires at least 2 dimensions\n");
    exit(EXIT_FAILURE);
  }

  int n = shape[ndim - 1], batch_size = 1;
  for (int i = 0; i < ndim - 2; i++) { batch_size *= shape[i]; }
  int matrix_size = n * n;
  for (int batch = 0; batch < batch_size; batch++) {
    float *a_batch = a + batch * matrix_size, *l_batch = l + batch * matrix_size, *u_batch = u + batch * matrix_size;
    int* p_batch = p + batch * n;
    int matrix_shape[2] = {n, n};
    lu_decomp_ops(a_batch, l_batch, u_batch, p_batch, matrix_shape);
  }
}

void lq_decomp_ops(float* a, float* l, float* q, int* shape) {
  int m = shape[0], n = shape[1];
  float* at = (float*)malloc(n * m * sizeof(float));  // transpose input matrix for qr decomposition
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) at[j * m + i] = a[i * n + j];
  }

  float *q_temp = (float*)malloc(n * n * sizeof(float)), *r_temp = (float*)malloc(n * m * sizeof(float));   // perform qr on transposed matrix
  int at_shape[2] = {n, m};
  qr_decomp_ops(at, q_temp, r_temp, at_shape);
  // transpose results back: a = l * q, where a^T = q^T * l^T
  // so l^T = r and q^T = q_temp, thus l = r^T and q = q_temp^T
  // l = r^T (transpose r_temp)
  memset(l, 0, m * m * sizeof(float));
  for (int i = 0; i < m && i < n; i++) {
    for (int j = 0; j <= i && j < m; j++) l[i * m + j] = r_temp[j * m + i];
  }

  // q = q_temp^T (transpose q_temp)
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      if (i < n) q[i * n + j] = q_temp[j * n + i];
      else q[i * n + j] = 0.0f;
    }
  } 
  free(at);
  free(q_temp);
  free(r_temp);
}

void batched_lq_decomp_ops(float* a, float* l, float* q, int* shape, int ndim) {
  if (ndim < 2) {
    fprintf(stderr, "error: lq decomposition requires at least 2 dimensions\n");
    exit(EXIT_FAILURE);
  }
  int m = shape[ndim - 2], n = shape[ndim - 1];
  int batch_size = 1;
  for (int i = 0; i < ndim - 2; i++) { batch_size *= shape[i]; }
  int a_matrix_size = m * n, l_matrix_size = m * m, q_matrix_size = m * n;
  for (int batch = 0; batch < batch_size; batch++) {
    float* a_batch = a + batch * a_matrix_size;
    float* l_batch = l + batch * l_matrix_size;
    float* q_batch = q + batch * q_matrix_size;
    int matrix_shape[2] = {m, n};
    lq_decomp_ops(a_batch, l_batch, q_batch, matrix_shape);
  }
}

static void compute_eigenvals(float* a, float* eigenvals, size_t size) {
  float *temp, *q, *r;
  size_t i, j, iter, mat_size = size * size;
  temp = (float*)malloc(3 * mat_size * sizeof(float));
  if (!temp) { 
    for (i = 0; i < size; ++i) eigenvals[i] = 0.0f; 
    return; 
  }
  q = temp;
  r = temp + mat_size;
  float* curr_a = temp + 2 * mat_size;
  for (i = 0; i < mat_size; ++i) curr_a[i] = a[i];
  for (iter = 0; iter < 200; ++iter) {
    float shift = 0.0f;
    if (size > 1) {
      float a11 = curr_a[(size-2) * size + (size-2)], a12 = curr_a[(size-2) * size + (size-1)], a21 = curr_a[(size-1) * size + (size-2)], a22 = curr_a[(size-1) * size + (size-1)];
      float trace = a11 + a22;
      float det = a11 * a22 - a12 * a21;
      float disc = trace * trace - 4.0f * det;
      if (disc >= 0.0f) {
        float sqrt_disc = sqrtf(disc);
        float lambda1 = (trace + sqrt_disc) / 2.0f, lambda2 = (trace - sqrt_disc) / 2.0f;
        shift = (fabsf(lambda1 - a22) < fabsf(lambda2 - a22)) ? lambda1 : lambda2;
      } else { shift = trace / 2.0f; }
    }
    for (i = 0; i < size; ++i) curr_a[i * size + i] -= shift;
    int qr_shape[2] = {(int)size, (int)size};
    qr_decomp_ops(curr_a, q, r, qr_shape);
    for (i = 0; i < mat_size; ++i) curr_a[i] = 0.0f;
    for (i = 0; i < size; ++i) {
      for (j = 0; j < size; ++j) {
        for (size_t k = 0; k < size; ++k) { curr_a[i * size + j] += r[i * size + k] * q[k * size + j]; }
      }
    }
    for (i = 0; i < size; ++i) curr_a[i * size + i] += shift;
    float off_diag = 0.0f;
    for (i = 0; i < size; ++i) {
      for (j = 0; j < size; ++j) { if (i != j) off_diag += fabsf(curr_a[i * size + j]); }
    }
    if (off_diag < 1e-8f) break;
  }
  for (i = 0; i < size; ++i) eigenvals[i] = curr_a[i * size + i];
  free(temp);
}

static void compute_eigenvecs(float* a, float* eigenvecs, size_t size) {
  float *temp, *q, *r, *qt, *v_acc;
  size_t i, j, iter, mat_size = size * size;
  temp = (float*)malloc(5 * mat_size * sizeof(float));
  if (!temp) { 
    for (i = 0; i < mat_size; ++i) eigenvecs[i] = 0.0f; 
    return;
  }
  q = temp, r = temp + mat_size, qt = temp + 2 * mat_size, v_acc = temp + 3 * mat_size;
  float* curr_a = temp + 4 * mat_size;
  for (i = 0; i < mat_size; ++i) {
    curr_a[i] = a[i];
    eigenvecs[i] = 0.0f;
    v_acc[i] = 0.0f;
  }
  for (i = 0; i < size; ++i) {
    eigenvecs[i * size + i] = 1.0f;
    v_acc[i * size + i] = 1.0f;
  }
  for (iter = 0; iter < 200; ++iter) {
    float shift = 0.0f;
    if (size > 1) {
      float a11 = curr_a[(size-2) * size + (size-2)], a12 = curr_a[(size-2) * size + (size-1)], a21 = curr_a[(size-1) * size + (size-2)], a22 = curr_a[(size-1) * size + (size-1)];
      float trace = a11 + a22;
      float det = a11 * a22 - a12 * a21;
      float disc = trace * trace - 4.0f * det;
      if (disc >= 0.0f) {
        float sqrt_disc = sqrtf(disc);
        float lambda1 = (trace + sqrt_disc) / 2.0f, lambda2 = (trace - sqrt_disc) / 2.0f;
        shift = (fabsf(lambda1 - a22) < fabsf(lambda2 - a22)) ? lambda1 : lambda2;
      } else { shift = trace / 2.0f; }
    }
    for (i = 0; i < size; ++i) curr_a[i * size + i] -= shift;
    int qr_shape[2] = {(int)size, (int)size};
    qr_decomp_ops(curr_a, q, r, qr_shape);
    for (i = 0; i < mat_size; ++i) qt[i] = 0.0f;
    for (i = 0; i < size; ++i) {
      for (j = 0; j < size; ++j) {
        for (size_t k = 0; k < size; ++k) { qt[i * size + j] += v_acc[i * size + k] * q[k * size + j]; }
      }
    }
    for (i = 0; i < mat_size; ++i) v_acc[i] = qt[i];
    for (i = 0; i < mat_size; ++i) curr_a[i] = 0.0f;
    for (i = 0; i < size; ++i) {
      for (j = 0; j < size; ++j) {
        for (size_t k = 0; k < size; ++k) { curr_a[i * size + j] += r[i * size + k] * q[k * size + j]; }
      }
    }
    for (i = 0; i < size; ++i) curr_a[i * size + i] += shift;
    float off_diag = 0.0f;
    for (i = 0; i < size; ++i) { for (j = 0; j < size; ++j) { if (i != j) off_diag += fabsf(curr_a[i * size + j]); } }
    if (off_diag < 1e-8f) break;
  }
  for (i = 0; i < mat_size; ++i) eigenvecs[i] = v_acc[i];
  free(temp);
}

// compute eigenvalues for hermitian/symmetric matrices using jacobi method
static void compute_eigenvals_h(float* a, float* eigenvals, size_t size) {
  float *temp;
  size_t i, j, k, iter, mat_size = size * size;
  temp = (float*)malloc(mat_size * sizeof(float));
  if (!temp) { 
    for (i = 0; i < size; ++i) eigenvals[i] = 0.0f; 
    return; 
  }
  for (i = 0; i < mat_size; ++i) temp[i] = a[i];    // copy input matrix
  // jacobi iteration
  for (iter = 0; iter < 100; ++iter) {
    // find largest off-diagonal element
    float max_val = 0.0f;
    size_t p = 0, q = 1;
    for (i = 0; i < size; ++i) {
      for (j = i + 1; j < size; ++j) {
        if (fabsf(temp[i * size + j]) > max_val) {
          max_val = fabsf(temp[i * size + j]);
          p = i; q = j;
        }
      }
    }

    if (max_val < 1e-10f) break;
    // computing jacobi rotation
    float theta = (temp[q * size + q] - temp[p * size + p]) / (2.0f * temp[p * size + q]);
    float t = 1.0f / (fabsf(theta) + sqrtf(theta * theta + 1.0f));
    if (theta < 0.0f) t = -t;
    float c = 1.0f / sqrtf(t * t + 1.0f);
    float s = t * c;
    // apply rotation
    for (k = 0; k < size; ++k) {
      if (k != p && k != q) {
        float temp_kp = temp[k * size + p], temp_kq = temp[k * size + q];
        temp[k * size + p] = temp[p * size + k] = c * temp_kp - s * temp_kq;
        temp[k * size + q] = temp[q * size + k] = s * temp_kp + c * temp_kq;
      }
    }
    float temp_pp = temp[p * size + p], temp_qq = temp[q * size + q], temp_pq = temp[p * size + q];
    temp[p * size + p] = c * c * temp_pp + s * s * temp_qq - 2.0f * s * c * temp_pq;
    temp[q * size + q] = s * s * temp_pp + c * c * temp_qq + 2.0f * s * c * temp_pq;
    temp[p * size + q] = temp[q * size + p] = 0.0f;
  }
  for (i = 0; i < size; ++i) eigenvals[i] = temp[i * size + i]; // extract eigenvalues from diagonal
  free(temp);
}

// computing eigenvectors for hermitian/symmetric matrices using jacobi method
static void compute_eigenvecs_h(float* a, float* eigenvecs, size_t size) {
  float *temp;
  size_t i, j, k, iter, mat_size = size * size;
  temp = (float*)malloc(mat_size * sizeof(float));
  if (!temp) { 
    for (i = 0; i < mat_size; ++i) eigenvecs[i] = 0.0f; 
    return; 
  }
  // copying input matrix and initialize eigenvectors as identity
  for (i = 0; i < mat_size; ++i) {
    temp[i] = a[i];
    eigenvecs[i] = 0.0f;
  }
  for (i = 0; i < size; ++i) eigenvecs[i * size + i] = 1.0f;
  // jacobi iteration
  for (iter = 0; iter < 100; ++iter) {
    // finding largest off-diagonal element
    float max_val = 0.0f;
    size_t p = 0, q = 1;
    for (i = 0; i < size; ++i) {
      for (j = i + 1; j < size; ++j) {
        if (fabsf(temp[i * size + j]) > max_val) {
          max_val = fabsf(temp[i * size + j]);
          p = i; q = j;
        }
      }
    }
    if (max_val < 1e-10f) break;
    // compute jacobi rotation
    float theta = (temp[q * size + q] - temp[p * size + p]) / (2.0f * temp[p * size + q]);
    float t = 1.0f / (fabsf(theta) + sqrtf(theta * theta + 1.0f));
    if (theta < 0.0f) t = -t;
    float c = 1.0f / sqrtf(t * t + 1.0f);
    float s = t * c;
    // apply rotation to matrix
    for (k = 0; k < size; ++k) {
      if (k != p && k != q) {
        float temp_kp = temp[k * size + p], temp_kq = temp[k * size + q];
        temp[k * size + p] = temp[p * size + k] = c * temp_kp - s * temp_kq;
        temp[k * size + q] = temp[q * size + k] = s * temp_kp + c * temp_kq;
      }
    }
    float temp_pp = temp[p * size + p], temp_qq = temp[q * size + q], temp_pq = temp[p * size + q];
    temp[p * size + p] = c * c * temp_pp + s * s * temp_qq - 2.0f * s * c * temp_pq;
    temp[q * size + q] = s * s * temp_pp + c * c * temp_qq + 2.0f * s * c * temp_pq;
    temp[p * size + q] = temp[q * size + p] = 0.0f;

    // apply rotation to eigenvectors
    for (k = 0; k < size; ++k) {
      float temp_kp = eigenvecs[k * size + p], temp_kq = eigenvecs[k * size + q];
      eigenvecs[k * size + p] = c * temp_kp - s * temp_kq;
      eigenvecs[k * size + q] = s * temp_kp + c * temp_kq;
    }
  }
  free(temp);
}

void eigenvals_ops_array(float* a, float* eigenvals, size_t size) {
  compute_eigenvals(a, eigenvals, size);
}

void batched_eigenvals_ops(float* a, float* eigenvals, size_t size, size_t batch) {
  size_t mat_size = size * size;
  for (size_t b = 0; b < batch; ++b) {
    float *mat = &a[b * mat_size], *vals = &eigenvals[b * size];
    eigenvals_ops_array(mat, vals, size);
  }
}

void eigenvecs_ops_array(float* a, float* eigenvecs, size_t size) {
  compute_eigenvecs(a, eigenvecs, size);
}

void batched_eigenvecs_ops(float* a, float* eigenvecs, size_t size, size_t batch) {
  size_t mat_size = size * size;
  for (size_t b = 0; b < batch; ++b) {
    float *mat = &a[b * mat_size], *vecs = &eigenvecs[b * mat_size];
    eigenvecs_ops_array(mat, vecs, size);
  }
}

void eigenvals_h_ops_array(float* a, float* eigenvals, size_t size) { compute_eigenvals_h(a, eigenvals, size); }

void batched_eigenvals_h_ops(float* a, float* eigenvals, size_t size, size_t batch) {
  size_t mat_size = size * size;
  for (size_t b = 0; b < batch; ++b) {
    float *mat = &a[b * mat_size], *vals = &eigenvals[b * size];
    eigenvals_h_ops_array(mat, vals, size);
  }
}

void eigenvecs_h_ops_array(float* a, float* eigenvecs, size_t size) { compute_eigenvecs_h(a, eigenvecs, size); }

void batched_eigenvecs_h_ops(float* a, float* eigenvecs, size_t size, size_t batch) {
  size_t mat_size = size * size;
  for (size_t b = 0; b < batch; ++b) {
    float *mat = &a[b * mat_size], *vecs = &eigenvecs[b * mat_size];
    eigenvecs_h_ops_array(mat, vecs, size);
  }
}