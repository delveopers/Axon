#ifndef __MATRIX__H__
#define __MATRIX__H__

#include "../core/core.h"
#include "../core/dtype.h"

extern "C" {
  Array* inv_array(Array* a);
  Array* matrix_rank_array(Array* a);
  Array* solve_array(Array* a, Array* b);
  Array* lstsq_array(Array* a, Array* b);
}

#endif  //!__MATRIX__H__