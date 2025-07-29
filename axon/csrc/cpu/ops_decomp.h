#ifndef __OPS_DECOMP__H__
#define __OPS_DECOMP__H__

#include <stddef.h>

extern "C" {
  void det_ops_array(float* a, float* out, size_t size);
  void qr_decomp_ops(float* a, float* q, float* r, int* shape);
  void batched_qr_decomp_ops(float* a, float* q, float* r, int* shape, int ndim);
  void lu_decomp_ops(float* a, float* l, float* u, int* p, int* shape);
  void batched_lu_decomp_ops(float* a, float* l, float* u, int* p, int* shape, int ndim);
  void lq_decomp_ops(float* a, float* l, float* q, int* shape);
  void batched_lq_decomp_ops(float* a, float* l, float* q, int* shape, int ndim);
  void batched_det_ops(float* a, float* out, size_t size, size_t batch);
  void eigenvals_ops_array(float* a, float* eigenvals, size_t size);
  void batched_eigenvals_ops(float* a, float* eigenvals, size_t size, size_t batch);
  void eigenvecs_ops_array(float* a, float* eigenvecs, size_t size);
  void batched_eigenvecs_ops(float* a, float* eigenvecs, size_t size, size_t batch);
  void eigenvals_h_ops_array(float* a, float* eigenvals, size_t size);
  void batched_eigenvals_h_ops(float* a, float* eigenvals, size_t size, size_t batch);
  void eigenvecs_h_ops_array(float* a, float* eigenvecs, size_t size);
  void batched_eigenvecs_h_ops(float* a, float* eigenvecs, size_t size, size_t batch);
}

#endif  //!__OPS_DECOMP__H__