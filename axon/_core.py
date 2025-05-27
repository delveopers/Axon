from ._cbase import CArray, lib
from ctypes import c_float, c_size_t, c_int
from typing import *
from .helpers.shape import get_shape, flatten

class array:
  def __init__(self, data: Union[List[Any], int, float]):
    if isinstance(data, (CArray, array)):
      self.data, self.shape = data, []
      self.size, self.ndim, self.value = 0, 0, None
    else:
      flat, shape = flatten(data), get_shape(data)
      size, ndim = len(flat), len(shape)

      self._float_arr = (c_float * size)(*flat)
      self._shape_arr = (c_int * ndim)(*shape)

      self.data = lib.create_array(self._float_arr, c_size_t(ndim), self._shape_arr, c_size_t(size))
      self.shape, self.size, self.ndim, self.value = tuple(shape), size, ndim, data

  def __repr__(self):
    return f"array({self.value})"

  def __str__(self):
    lib.print_tensor(self.data)
    return ""

  def __add__(self, other):
    if isinstance(other, (int, float)):
      result_ptr = lib.add_scalar_array(self.data, c_float(other)).contents
    else:
      other = other if isinstance(other, (CArray, array)) else array(other)
      result_ptr = lib.add_array(self.data, other.data).contents
    out = array(result_ptr)
    out.shape, out.size, out.ndim = self.shape, self.size, self.ndim
    return out

  def __radd__(self, other):
    return self + other

  def __sub__(self, other):
    if isinstance(other, (int, float)):
      result_ptr = lib.sub_scalar_array(self.data, c_float(other)).contents
    else:
      other = other if isinstance(other, (CArray, array)) else array(other)
      result_ptr = lib.sub_array(self.data, other.data).contents
    out = array(result_ptr)
    out.shape, out.size, out.ndim = self.shape, self.size, self.ndim
    return out

  def __rsub__(self, other):
    return self + other

  def __mul__(self, other):
    if isinstance(other, (int, float)):
      result_ptr = lib.mul_scalar_array(self.data, c_float(other)).contents
    else:
      other = other if isinstance(other, (CArray, array)) else array(other)
      result_ptr = lib.mul_array(self.data, other.data).contents
    out = array(result_ptr)
    out.shape, out.size, out.ndim = self.shape, self.size, self.ndim
    return out

  def __rmul__(self, other):
    return self + other

  def __truediv__(self, other):
    if isinstance(other, (int, float)):
      result_ptr = lib.div_scalar_array(self.data, c_float(other)).contents
    else:
      other = other if isinstance(other, (CArray, array)) else array(other)
      result_ptr = lib.div_array(self.data, other.data).contents
    out = array(result_ptr)
    out.shape, out.size, out.ndim = self.shape, self.size, self.ndim
    return out

  def __rtruediv__(self, other):
    return self + other

  def sin(self):
    result_ptr = lib.sin_array(self.data).contents
    out = array(result_ptr)
    out.shape, out.size, out.ndim = self.shape, self.size, self.ndim
    return out

  def cos(self):
    result_ptr = lib.cos_array(self.data).contents
    out = array(result_ptr)
    out.shape, out.size, out.ndim = self.shape, self.size, self.ndim
    return out

  def tan(self):
    result_ptr = lib.tan_array(self.data).contents
    out = array(result_ptr)
    out.shape, out.size, out.ndim = self.shape, self.size, self.ndim
    return out

  def sinh(self):
    result_ptr = lib.sinh_array(self.data).contents
    out = array(result_ptr)
    out.shape, out.size, out.ndim = self.shape, self.size, self.ndim
    return out

  def cosh(self):
    result_ptr = lib.cosh_array(self.data).contents
    out = array(result_ptr)
    out.shape, out.size, out.ndim = self.shape, self.size, self.ndim
    return out

  def tanh(self):
    result_ptr = lib.tanh_array(self.data).contents
    out = array(result_ptr)
    out.shape, out.size, out.ndim = self.shape, self.size, self.ndim
    return out