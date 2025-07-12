from typing import *
from ctypes import c_int, c_float, c_double
from .._cbase import CArray, lib, DType
from .._core import array
from .._helpers import DtypeHelp, ShapeHelp

def det(a: array, dtype: DType = 'float32') -> array:
  a = a if isinstance(a, array) else array(a, 'float32')
  if a.ndim == 2:
    ptr = lib.det_array(a.data).contents
    out_shape, out_size, out_ndim, out_strides = (), 0, 1, ()
  elif a.ndim == 3:
    ptr = lib.batched_det_array(a.data).contents
    out_shape, out_size, out_ndim, out_strides = a.shape[:-2], a.size // (a.shape[-1] * a.shape[-2]), a.ndim - 2, ShapeHelp.get_strides(a.shape[:-2])
  else: raise ValueError("Can't compute determinant for 3d > ndims")
  out = array(ptr, dtype if dtype is not None else a.dtype)
  return (setattr(out, "shape", out_shape), setattr(out, "ndim", out_ndim), setattr(out, "size", out_size), setattr(out, "strides", out_strides), out)[4]

# def eig(a: array, dtype: DType = 'float32') -> array:
#   a = a if isinstance(a, array) else array(a, 'float32')
#   ptr = lib.eig_array(a.data).contents
#   out = array(ptr, dtype if dtype is not None else a.dtype)
#   out_shape, out_size, out_ndim, out_strides = (a.shape[0],), a.shape[0], 1, (1,)
#   return (setattr(out, "shape", out_shape), setattr(out, "ndim", out_ndim), setattr(out, "size", out_size), setattr(out, "strides", out_strides), out)[4]

# def eigv(a: array, dtype: DType = 'float32') -> array:
#   a = a if isinstance(a, array) else array(a, 'float32')
#   ptr = lib.eigv_array(a.data).contents
#   out = array(ptr, dtype if dtype is not None else a.dtype)
#   out_shape, out_size, out_ndim, out_strides = a.shape, a.size, a.ndim, a.strides
#   return (setattr(out, "shape", out_shape), setattr(out, "ndim", out_ndim), setattr(out, "size", out_size), setattr(out, "strides", out_strides), out)[4]

# def eigh(a: array, dtype: DType = 'float32') -> array:
#   a = a if isinstance(a, array) else array(a, 'float32')
#   ptr = lib.eigh_array(a.data).contents
#   out = array(ptr, dtype if dtype is not None else a.dtype)
#   out_shape, out_size, out_ndim, out_strides = (a.shape[0],), a.shape[0], 1, (1,)
#   return (setattr(out, "shape", out_shape), setattr(out, "ndim", out_ndim), setattr(out, "size", out_size), setattr(out, "strides", out_strides), out)[4]

# def eighv(a: array, dtype: DType = 'float32') -> array:
#   a = a if isinstance(a, array) else array(a, 'float32')
#   ptr = lib.eighv_array(a.data).contents
#   out = array(ptr, dtype if dtype is not None else a.dtype)
#   out_shape, out_size, out_ndim, out_strides = a.shape, a.size, a.ndim, a.strides
#   return (setattr(out, "shape", out_shape), setattr(out, "ndim", out_ndim), setattr(out, "size", out_size), setattr(out, "strides", out_strides), out)[4]