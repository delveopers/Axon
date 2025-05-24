from ._cbase import CArray, lib
import ctypes
from ctypes import *
from typing import *
from helpers.shape import get_shape, flatten

class array:
  def __init__(self, data: Union[List[int, float], int, float]):
    temp_shape, temp_flatten = get_shape(data), flatten(data)
    temp_ndim = len(temp_shape)
    if isinstance(data, CArray):
      self.data = data
    else:
      self.data = lib.create_array(c_float(data), c_size_t(temp_ndim), POINTER(c_int(temp_shape)), c_size_t(len(temp_flatten)))
    self.shape, self.size, self.ndim = temp_shape, len(temp_flatten), temp_ndim
    del temp_ndim, temp_shape, temp_flatten