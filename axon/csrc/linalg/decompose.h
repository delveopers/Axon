#ifndef __DECOMPOSE__H__
#define __DECOMPOSE__H__

#include "../core/core/core.h"
#include "../core/core/dtype.h"

extern "C" {
  Array* det_array(Array* a);
  Array* batched_det_array(Array* a);
  Array* eig_array(Array* a);   // eigen values
  Array* eigv_array(Array* a);  // eigen vectors
  Array* eigh_array(Array* a);    // eigen hermatian values
  Array* eighv_array(Array* a);   // eigen hermatian vectors
}

#endif  //!__DECOMPOSE__H__