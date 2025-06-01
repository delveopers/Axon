from ._cbase import CArray, lib
from ctypes import c_int, c_size_t, c_float
from typing import *
from .helpers.shape import get_strides
from ._core import array

def zeros_like(arr: Union[CArray, array]) -> array:
  if isinstance(arr, array):
    result_ptr = lib.zeros_like_array(arr.data).contents
  elif isinstance(arr, CArray):
    result_ptr = lib.zeros_like_array(arr)
  out = array(result_ptr)
  out.shape, out.ndim, out.size, out.strides = arr.shape, arr.ndim, arr.size, arr.strides
  return out

def ones_like(arr: Union[CArray, array]) -> array:
  if isinstance(arr, array):
    result_ptr = lib.ones_like_array(arr.data).contents
  elif isinstance(arr, CArray):
    result_ptr = lib.ones_like_array(arr)
  out = array(result_ptr)
  out.shape, out.ndim, out.size, out.strides = arr.shape, arr.ndim, arr.size, arr.strides
  return out

def zeros(shape: Union[list, tuple]) -> array:
  shape = list(shape)if isinstance(shape, tuple) else shape
  size, ndim = 1, len(shape)
  for dim in shape:
    size *= dim
  shape_arr = (c_int * size)(*shape)
  result_ptr = lib.zeros_array(shape_arr, c_size_t(size), c_size_t(ndim)).contents
  out = array(result_ptr)
  out.shape, out.ndim, out.size, out.strides = tuple(shape), ndim, size, get_strides(tuple(shape))
  return out

def ones(shape: Union[list, tuple]) -> array:
  shape = list(shape)if isinstance(shape, tuple) else shape
  size, ndim = 1, len(shape)
  for dim in shape:
    size *= dim
  shape_arr = (c_int * size)(*shape)
  result_ptr = lib.ones_array(shape_arr, c_size_t(size), c_size_t(ndim)).contents
  out = array(result_ptr)
  out.shape, out.ndim, out.size, out.strides = tuple(shape), ndim, size, get_strides(tuple(shape))
  return out

def randn(shape: Union[list, tuple]) -> array:
  shape = list(shape)if isinstance(shape, tuple) else shape
  size, ndim = 1, len(shape)
  for dim in shape:
    size *= dim
  shape_arr = (c_int * size)(*shape)
  result_ptr = lib.randn_array(shape_arr, c_size_t(size), c_size_t(ndim)).contents
  out = array(result_ptr)
  out.shape, out.ndim, out.size, out.strides = tuple(shape), ndim, size, get_strides(tuple(shape))
  return out

def randint(low: int, high: int, shape: Union[list, tuple]) -> array:
  shape = list(shape)if isinstance(shape, tuple) else shape
  size, ndim = 1, len(shape)
  for dim in shape:
    size *= dim
  shape_arr = (c_int * size)(*shape)
  result_ptr = lib.randint_array(c_int(low), c_int(high), shape_arr, c_size_t(size), c_size_t(ndim)).contents
  out = array(result_ptr)
  out.shape, out.ndim, out.size, out.strides = tuple(shape), ndim, size, get_strides(tuple(shape))
  return out

def uniform(low: int, high: int, shape: Union[list, tuple]) -> array:
  shape = list(shape)if isinstance(shape, tuple) else shape
  size, ndim = 1, len(shape)
  for dim in shape:
    size *= dim
  shape_arr = (c_int * size)(*shape)
  result_ptr = lib.uinform_array(c_int(low), c_int(high), shape_arr, c_size_t(size), c_size_t(ndim)).contents
  out = array(result_ptr)
  out.shape, out.ndim, out.size, out.strides = tuple(shape), ndim, size, get_strides(tuple(shape))
  return out

def fill(fill: float, shape: Union[list, tuple]) -> array:
  shape = list(shape)if isinstance(shape, tuple) else shape
  size, ndim = 1, len(shape)
  for dim in shape:
    size *= dim
  shape_arr = (c_int * size)(*shape)
  result_ptr = lib.fill_array(c_float(fill), shape_arr, c_size_t(size), c_size_t(ndim)).contents
  out = array(result_ptr)
  out.shape, out.ndim, out.size, out.strides = tuple(shape), ndim, size, get_strides(tuple(shape))
  return out

def linspace(start: float, step: float, end: float, shape: Union[list, tuple]) -> array:
  shape = list(shape)if isinstance(shape, tuple) else shape
  size, ndim = 1, len(shape)
  for dim in shape:
    size *= dim
  shape_arr = (c_int * size)(*shape)
  result_ptr = lib.linspace_array(c_float(start), c_float(step), c_float(end), shape_arr, c_size_t(size), c_size_t(ndim)).contents
  out = array(result_ptr)
  out.shape, out.ndim, out.size, out.strides = tuple(shape), ndim, size, get_strides(tuple(shape))
  return out