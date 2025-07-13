#include <stdlib.h>
#include <stddef.h>
#include <math.h>
#include "ops_decomp.h"
#include "ops_array.h"
#include "ops_shape.h"

void det_ops_array(float* a, float* out, size_t size) {
  float det = 1.0f;
  // creating a copy to avoid modifying original matrix
  float* temp = (float*)malloc(size * size * sizeof(float));
  if (!temp) { *out = 0.0f; return; }
  for (size_t i = 0; i < size * size; ++i) { temp[i] = a[i]; }
  // gaussian elimination with partial pivoting
  for (size_t i = 0; i < size; ++i) {
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
      for (size_t k = i; k < size; ++k) { temp[j * size + k] -= factor * temp[i * size + k]; }
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

// qr decomposition helper for eigenvalue computation
static void qr_decomp(float* a, float* q, float* r, size_t n) {
  size_t i, j, k;
  float norm, sum;
  // initialize q as identity, r as copy of a
  for (i = 0; i < n * n; ++i) { q[i] = 0.0f; r[i] = a[i]; }
  for (i = 0; i < n; ++i) q[i * n + i] = 1.0f;
  // gram-schmidt process
  for (j = 0; j < n; ++j) {
    norm = 0.0f;  // computing norm of column j
    for (i = 0; i < n; ++i) norm += r[i * n + j] * r[i * n + j];
    norm = sqrtf(norm);
    if (norm < 1e-10f) continue;
    for (i = 0; i < n; ++i) {     // normalize column j
      q[i * n + j] = r[i * n + j] / norm;
      r[i * n + j] = (i == j) ? norm : 0.0f;
    }   
    // orthogonalize remaining columns
    for (k = j + 1; k < n; ++k) {
      sum = 0.0f;
      for (i = 0; i < n; ++i) sum += q[i * n + j] * r[i * n + k];
      r[j * n + k] = sum;
      for (i = 0; i < n; ++i) r[i * n + k] -= sum * q[i * n + j];
    }
  }
}

// compute eigenvalues using qr iteration
static void compute_eigenvals(float* a, float* eigenvals, size_t size) {
  float *temp, *q, *r, *rq;
  size_t i, j, iter, mat_size = size * size;
  
  // allocate working memory
  temp = (float*)malloc(4 * mat_size * sizeof(float));
  if (!temp) { for (i = 0; i < size; ++i) eigenvals[i] = 0.0f; return; }

  q = temp, r = temp + mat_size, rq = temp + 2 * mat_size;
  for (i = 0; i < mat_size; ++i) temp[3 * mat_size + i] = a[i]; // copying input matrix

  // qr iteration
  for (iter = 0; iter < 100; ++iter) {
    qr_decomp(temp + 3 * mat_size, q, r, size);
    int shape[2] = {(int)size, (int)size};
    matmul_array_ops(r, q, rq, shape, shape);
    for (i = 0; i < mat_size; ++i) temp[3 * mat_size + i] = rq[i];  // copying back for next iteration
    // check convergence
    float off_diag = 0.0f;
    for (i = 0; i < size; ++i) { for (j = 0; j < size; ++j) { if (i != j) off_diag += fabsf(rq[i * size + j]); } }
    if (off_diag < 1e-6f) break;
  }
  for (i = 0; i < size; ++i) eigenvals[i] = rq[i * size + i];   // extract eigenvalues from diagonal
  free(temp);
}

// compute eigenvectors using qr iteration
static void compute_eigenvecs(float* a, float* eigenvecs, size_t size) {
  float *temp, *q, *r, *rq;
  size_t i, j, iter, mat_size = size * size;
  
  // allocate working memory
  temp = (float*)malloc(4 * mat_size * sizeof(float));
  if (!temp) { for (i = 0; i < mat_size; ++i) eigenvecs[i] = 0.0f; return; }

  q = temp, r = temp + mat_size, rq = temp + 2 * mat_size;
  for (i = 0; i < mat_size; ++i) temp[3 * mat_size + i] = a[i]; // copying input matrix
  // initialize eigenvectors as identity
  for (i = 0; i < mat_size; ++i) eigenvecs[i] = 0.0f;
  for (i = 0; i < size; ++i) eigenvecs[i * size + i] = 1.0f;

  // qr iteration
  for (iter = 0; iter < 100; ++iter) {
    qr_decomp(temp + 3 * mat_size, q, r, size);
    int shape[2] = {(int)size, (int)size};
    matmul_array_ops(r, q, rq, shape, shape);
    // update eigenvectors
    float* new_vecs = (float*)malloc(mat_size * sizeof(float));
    if (new_vecs) {
      matmul_array_ops(eigenvecs, q, new_vecs, shape, shape);
      for (i = 0; i < mat_size; ++i) eigenvecs[i] = new_vecs[i];
      free(new_vecs);
    }
    for (i = 0; i < mat_size; ++i) temp[3 * mat_size + i] = rq[i];  // copying back for next iteration
    // check convergence
    float off_diag = 0.0f;
    for (i = 0; i < size; ++i) { for (j = 0; j < size; ++j) { if (i != j) off_diag += fabsf(rq[i * size + j]); } }
    if (off_diag < 1e-6f) break;
  }
  free(temp);
}

// compute eigenvalues for hermitian/symmetric matrices
static void compute_eigenvals_h(float* a, float* eigenvals, size_t size) {
  float *temp, *d, *e;
  size_t i, j, k, iter, mat_size = size * size;

  // allocate working memory
  temp = (float*)malloc((mat_size + 2 * size) * sizeof(float));
  if (!temp) { for (i = 0; i < size; ++i) eigenvals[i] = 0.0f; return; }
  d = temp + mat_size, e = temp + mat_size + size;
  // copy input
  for (i = 0; i < mat_size; ++i) temp[i] = a[i];

  // tridiagonalization using householder reduction
  for (k = 0; k < size - 2; ++k) {
    float alpha = 0.0f, beta, tau, sum;
    // compute norm of column below diagonal
    for (i = k + 1; i < size; ++i) alpha += temp[i * size + k] * temp[i * size + k];
    alpha = sqrtf(alpha);
    if (alpha < 1e-10f) continue;    
    if (temp[(k + 1) * size + k] < 0) alpha = -alpha;

    // compute householder vector
    beta = temp[(k + 1) * size + k] + alpha;
    tau = 2.0f / (alpha * alpha + beta * beta);

    // apply householder transformation
    for (i = k + 1; i < size; ++i) {
      for (j = k + 1; j < size; ++j) {
        sum = 0.0f;
        for (size_t l = k + 1; l < size; ++l) {
          float vi = (l == i) ? beta : temp[l * size + k];
          float vj = (l == j) ? beta : temp[l * size + k];
          sum += vi * vj;
        }
        temp[i * size + j] -= tau * sum;
      }
    }
  }

  // extract diagonal and subdiagonal
  for (i = 0; i < size; ++i) {
    d[i] = temp[i * size + i];
    e[i] = (i < size - 1) ? temp[(i + 1) * size + i] : 0.0f;
  }

  // ql algorithm for tridiagonal matrix
  for (iter = 0; iter < 100; ++iter) {
    float converged = 1.0f;
    for (i = 0; i < size - 1; ++i) {
      if (fabsf(e[i]) > 1e-10f) {
        converged = 0.0f;
        // wilkinson shift
        float delta = (d[i + 1] - d[i]) / (2.0f * e[i]);
        float t = 1.0f / (delta + ((delta > 0) ? 1.0f : -1.0f) * sqrtf(delta * delta + 1.0f));
        float c = 1.0f / sqrtf(1.0f + t * t), s = t * c;
        // applying givens rotation
        float temp_d = d[i], temp_e = e[i];
        d[i] = c * c * temp_d + s * s * d[i + 1] - 2.0f * s * c * temp_e;
        d[i + 1] = s * s * temp_d + c * c * d[i + 1] + 2.0f * s * c * temp_e;
        e[i] = (c * c - s * s) * temp_e + s * c * (d[i + 1] - temp_d);
      }
    }
    if (converged) break;
  }
  for (i = 0; i < size; ++i) eigenvals[i] = d[i];   // copy eigenvalues
  free(temp);
}

// compute eigenvectors for hermitian/symmetric matrices
static void compute_eigenvecs_h(float* a, float* eigenvecs, size_t size) {
  float *temp, *d, *e;
  size_t i, j, k, iter, mat_size = size * size;

  // allocate working memory
  temp = (float*)malloc((mat_size + 2 * size) * sizeof(float));
  if (!temp) { for (i = 0; i < mat_size; ++i) eigenvecs[i] = 0.0f; return; }
  d = temp + mat_size, e = temp + mat_size + size;
  // copy input and initialize eigenvectors as identity
  for (i = 0; i < mat_size; ++i) { temp[i] = a[i]; eigenvecs[i] = 0.0f; }
  for (i = 0; i < size; ++i) eigenvecs[i * size + i] = 1.0f;

  // tridiagonalization using householder reduction
  for (k = 0; k < size - 2; ++k) {
    float alpha = 0.0f, beta, tau, sum;
    // compute norm of column below diagonal
    for (i = k + 1; i < size; ++i) alpha += temp[i * size + k] * temp[i * size + k];
    alpha = sqrtf(alpha);
    if (alpha < 1e-10f) continue;    
    if (temp[(k + 1) * size + k] < 0) alpha = -alpha;

    // compute householder vector
    beta = temp[(k + 1) * size + k] + alpha;
    tau = 2.0f / (alpha * alpha + beta * beta);

    // apply householder transformation
    for (i = k + 1; i < size; ++i) {
      for (j = k + 1; j < size; ++j) {
        sum = 0.0f;
        for (size_t l = k + 1; l < size; ++l) {
          float vi = (l == i) ? beta : temp[l * size + k];
          float vj = (l == j) ? beta : temp[l * size + k];
          sum += vi * vj;
        }
        temp[i * size + j] -= tau * sum;
      }
    }
  }

  // extract diagonal and subdiagonal
  for (i = 0; i < size; ++i) {
    d[i] = temp[i * size + i];
    e[i] = (i < size - 1) ? temp[(i + 1) * size + i] : 0.0f;
  }

  // ql algorithm for tridiagonal matrix
  for (iter = 0; iter < 100; ++iter) {
    float converged = 1.0f;
    for (i = 0; i < size - 1; ++i) {
      if (fabsf(e[i]) > 1e-10f) {
        converged = 0.0f;
        // wilkinson shift
        float delta = (d[i + 1] - d[i]) / (2.0f * e[i]);
        float t = 1.0f / (delta + ((delta > 0) ? 1.0f : -1.0f) * sqrtf(delta * delta + 1.0f));
        float c = 1.0f / sqrtf(1.0f + t * t), s = t * c;
        // applying givens rotation
        float temp_d = d[i], temp_e = e[i];
        d[i] = c * c * temp_d + s * s * d[i + 1] - 2.0f * s * c * temp_e;
        d[i + 1] = s * s * temp_d + c * c * d[i + 1] + 2.0f * s * c * temp_e;
        e[i] = (c * c - s * s) * temp_e + s * c * (d[i + 1] - temp_d);
        
        // updating eigenvectors
        for (j = 0; j < size; ++j) {
          float temp_v = eigenvecs[j * size + i];
          eigenvecs[j * size + i] = c * temp_v - s * eigenvecs[j * size + i + 1];
          eigenvecs[j * size + i + 1] = s * temp_v + c * eigenvecs[j * size + i + 1];
        }
      }
    }
    if (converged) break;
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

void eigenvals_h_ops_array(float* a, float* eigenvals, size_t size) {
  compute_eigenvals_h(a, eigenvals, size);
}

void batched_eigenvals_h_ops(float* a, float* eigenvals, size_t size, size_t batch) {
  size_t mat_size = size * size;
  for (size_t b = 0; b < batch; ++b) {
    float *mat = &a[b * mat_size], *vals = &eigenvals[b * size];
    eigenvals_h_ops_array(mat, vals, size);
  }
}

void eigenvecs_h_ops_array(float* a, float* eigenvecs, size_t size) {
  compute_eigenvecs_h(a, eigenvecs, size);
}

void batched_eigenvecs_h_ops(float* a, float* eigenvecs, size_t size, size_t batch) {
  size_t mat_size = size * size;
  for (size_t b = 0; b < batch; ++b) {
    float *mat = &a[b * mat_size], *vecs = &eigenvecs[b * mat_size];
    eigenvecs_h_ops_array(mat, vecs, size);
  }
}