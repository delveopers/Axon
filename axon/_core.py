from ._cbase import CArray, lib
from ctypes import c_float, c_size_t, c_int
from typing import *
from .helpers.shape import get_shape, flatten, get_strides, transposed_shape

class array:
  def __init__(self, data: Union[List[Any], int, float]) -> None:
    if isinstance(data, (CArray, array)):
      self.data, self.shape = data, []
      self.size, self.ndim, self.value, self.strides = 0, 0, None, 0      
    else:
      flat, shape = flatten(data), get_shape(data)
      size, ndim = len(flat), len(shape)

      self._float_arr = (c_float * size)(*flat)
      self._shape_arr = (c_int * ndim)(*shape)

      self.data = lib.create_array(self._float_arr, c_size_t(ndim), self._shape_arr, c_size_t(size))
      self.shape, self.size, self.ndim, self.value, self.strides = tuple(shape), size, ndim, data, get_strides(shape)

  def __repr__(self) -> str:
    return f"array({self.value})"

  def __str__(self) -> str:
    lib.print_array(self.data)
    return ""

  def __add__(self, other) -> "array":
    if isinstance(other, (int, float)):
      result_ptr = lib.add_scalar_array(self.data, c_float(other)).contents
    else:
      other = other if isinstance(other, (CArray, array)) else array(other)
      result_ptr = lib.add_array(self.data, other.data).contents
    out = array(result_ptr)
    out.shape, out.size, out.ndim, out.strides = self.shape, self.size, self.ndim, self.strides
    return out

  def __radd__(self, other) -> "array":
    return self + other

  def __sub__(self, other) -> "array":
    if isinstance(other, (int, float)):
      result_ptr = lib.sub_scalar_array(self.data, c_float(other)).contents
    else:
      other = other if isinstance(other, (CArray, array)) else array(other)
      result_ptr = lib.sub_array(self.data, other.data).contents
    out = array(result_ptr)
    out.shape, out.size, out.ndim, out.strides = self.shape, self.size, self.ndim, self.strides
    return out

  def __rsub__(self, other) -> "array":
    return self + other

  def __mul__(self, other) -> "array":
    if isinstance(other, (int, float)):
      result_ptr = lib.mul_scalar_array(self.data, c_float(other)).contents
    else:
      other = other if isinstance(other, (CArray, array)) else array(other)
      result_ptr = lib.mul_array(self.data, other.data).contents
    out = array(result_ptr)
    out.shape, out.size, out.ndim, out.strides = self.shape, self.size, self.ndim, self.strides
    return out

  def __rmul__(self, other) -> "array":
    return self + other

  def __truediv__(self, other) -> "array":
    if isinstance(other, (int, float)):
      result_ptr = lib.div_scalar_array(self.data, c_float(other)).contents
    else:
      other = other if isinstance(other, (CArray, array)) else array(other)
      result_ptr = lib.div_array(self.data, other.data).contents
    out = array(result_ptr)
    out.shape, out.size, out.ndim, out.strides = self.shape, self.size, self.ndim, self.strides
    return out

  def __rtruediv__(self, other) -> "array":
    return self + other

  def __pow__(self, exp) -> "array":
    if isinstance(exp, (int, float)):
      result_ptr = lib.pow_array(self.data, c_float(exp)).contents
    else:
      pass
    out = array(result_ptr)
    out.shape, out.size, out.ndim, out.strides = self.shape, self.size, self.ndim, self.strides
    return out

  def sin(self) -> "array":
    result_ptr = lib.sin_array(self.data).contents
    out = array(result_ptr)
    out.shape, out.size, out.ndim, out.strides = self.shape, self.size, self.ndim, self.strides
    return out

  def cos(self) -> "array":
    result_ptr = lib.cos_array(self.data).contents
    out = array(result_ptr)
    out.shape, out.size, out.ndim, out.strides = self.shape, self.size, self.ndim, self.strides
    return out

  def tan(self) -> "array":
    result_ptr = lib.tan_array(self.data).contents
    out = array(result_ptr)
    out.shape, out.size, out.ndim, out.strides = self.shape, self.size, self.ndim, self.strides
    return out

  def sinh(self) -> "array":
    result_ptr = lib.sinh_array(self.data).contents
    out = array(result_ptr)
    out.shape, out.size, out.ndim, out.strides = self.shape, self.size, self.ndim, self.strides
    return out

  def cosh(self) -> "array":
    result_ptr = lib.cosh_array(self.data).contents
    out = array(result_ptr)
    out.shape, out.size, out.ndim, out.strides = self.shape, self.size, self.ndim, self.strides
    return out

  def tanh(self) -> "array":
    result_ptr = lib.tanh_array(self.data).contents
    out = array(result_ptr)
    out.shape, out.size, out.ndim, out.strides = self.shape, self.size, self.ndim, self.strides
    return out

  def transpose(self) -> "array":
    assert self.shape <= 3, ".transpose() only supported till 3-d arrays"
    result_ptr = lib.transpose(self.data).contents
    out = array(result_ptr)
    out.shape, out.size, out.ndim = transposed_shape(self.shape), self.size, self.ndim
    out.strides = get_strides(out.shape)
    return out

  def reshape(self, new_shape: Union[List[int], Tuple[int]]) -> "array":
    if isinstance(new_shape, tuple):
      new_shape = list(new_shape)
    new_size = 1
    for dim in new_shape:
      new_size *= dim

    if new_size != self.size:
      raise ValueError(f"Cannot reshape array of size {self.size} into shape {new_shape}")

    ndim = len(new_shape)
    shape_arr = (c_int * ndim)(*new_shape)
    result_ptr = lib.reshape_array(self.data, shape_arr, c_int(ndim)).contents

    out = array(result_ptr)
    out.shape, out.size, out.ndim = tuple(new_shape), self.size, ndim
    out.strides = get_strides(new_shape)
    return out