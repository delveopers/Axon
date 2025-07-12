from typing import *
from ctypes import c_int, c_float, c_double
from .._cbase import CArray, lib, DType
from .._core import array
from .._helpers import DtypeHelp, ShapeHelp

def dot(a: array, b: array, dtype: DType = 'float32') -> array:
  a, b = a if isinstance(a, array) else array(a, 'float32'), b if isinstance(b, array) else array(b, 'float32')
  ptr = lib.vector_dot(a.data, b.data).contents
  out = array(ptr, dtype if dtype is not None else a.dtype)
  return (setattr(out, "shape", ()), setattr(out, "ndim", 0), setattr(out, "size", 1), setattr(out, "strides", ()), out)[4]

def dot_mv(mat: array, vec: array, dtype: DType = 'float32') -> array:
  mat, vec = mat if isinstance(mat, array) else array(mat, 'float32'), vec if isinstance(vec, array) else array(vec, 'float32')
  ptr = lib.vector_matrix_dot(vec.data, mat.data).contents
  out = array(ptr, dtype if dtype is not None else mat.dtype)
  out_shape, out_size, out_ndim, out_strides = (mat.shape[0],), mat.shape[0], 1, (1,)
  return (setattr(out, "shape", out_shape), setattr(out, "ndim", out_ndim), setattr(out, "size", out_size), setattr(out, "strides", out_strides), out)[4]

def inner(a: array, b: array, dtype: DType = 'float32') -> array:
  a, b = a if isinstance(a, array) else array(a, 'float32'), b if isinstance(b, array) else array(b, 'float32')
  ptr = lib.vector_inner(a.data, b.data).contents
  out = array(ptr, dtype if dtype is not None else a.dtype)
  return (setattr(out, "shape", ()), setattr(out, "ndim", 0), setattr(out, "size", 1), setattr(out, "strides", ()), out)[4]

def outer(a: array, b: array, dtype: DType = 'float32') -> array:
  a, b = a if isinstance(a, array) else array(a, 'float32'), b if isinstance(b, array) else array(b, 'float32')
  ptr = lib.vector_outer(a.data, b.data).contents
  out = array(ptr, dtype if dtype is not None else a.dtype)
  out_shape, out_size, out_ndim, out_strides = (a.shape[0], b.shape[0]), a.shape[0] * b.shape[0], 2, ShapeHelp.get_strides((a.shape[0], b.shape[0]))
  return (setattr(out, "shape", out_shape), setattr(out, "ndim", out_ndim), setattr(out, "size", out_size), setattr(out, "strides", out_strides), out)[4]

def cross(a: array, b: array, axis: int=None, dtype: DType = 'float32') -> array:
  a, b = a if isinstance(a, array) else array(a, 'float32'), b if isinstance(b, array) else array(b, 'float32')
  if axis is None:    
    ptr = lib.vector_cross(a.data, b.data).contents
  else:
    ptr = lib.vector_cross_axis(a.data, b.data, c_int(axis)).contents
  out = array(ptr, dtype if dtype is not None else a.dtype)
  out_shape, out_size, out_ndim, out_strides = a.shape, a.size, a.ndim, a.strides
  return (setattr(out, "shape", out_shape), setattr(out, "ndim", out_ndim), setattr(out, "size", out_size), setattr(out, "strides", out_strides), out)[4]