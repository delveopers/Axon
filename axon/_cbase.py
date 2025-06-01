import ctypes, os
from ctypes import Structure, c_float, c_int, c_size_t, POINTER
from typing import *

lib_path = os.path.join(os.path.dirname(__file__), '../build/libarray.so')
lib = ctypes.CDLL(lib_path)

class CArray(Structure):
  pass

CArray._fields_ = [
  ("data", POINTER(c_float)),
  ("shape", POINTER(c_int)),
  ("strides", POINTER(c_int)),
  ("backstrides", POINTER(c_int)),
  ("size", c_size_t),
  ("ndim", c_size_t),
]

lib.create_array.argtypes = [POINTER(c_float), c_size_t, POINTER(c_int), c_size_t]
lib.create_array.restype = POINTER(CArray)
lib.delete_array.argtypes = [POINTER(CArray)]
lib.delete_array.restype = None
lib.delete_data.argtypes = [POINTER(CArray)]
lib.delete_data.restype = None
lib.delete_shape.argtypes = [POINTER(CArray)]
lib.delete_shape.restype = None
lib.delete_strides.argtypes = [POINTER(CArray)]
lib.delete_strides.restype = None
lib.print_array.argtypes = [POINTER(CArray)]
lib.print_array.restype = None

# maths ops ----
lib.add_array.argtypes = [POINTER(CArray), POINTER(CArray)]
lib.add_array.restype = POINTER(CArray)
lib.add_scalar_array.argtypes = [POINTER(CArray), c_float]
lib.add_scalar_array.restype = POINTER(CArray)
lib.add_broadcasted_array.argtypes = [POINTER(CArray), POINTER(CArray)]
lib.add_broadcasted_array.restype = POINTER(CArray)
lib.sub_array.argtypes = [POINTER(CArray), POINTER(CArray)]
lib.sub_array.restype = POINTER(CArray)
lib.sub_scalar_array.argtypes = [POINTER(CArray), c_float]
lib.sub_scalar_array.restype = POINTER(CArray)
lib.sub_broadcasted_array.argtypes = [POINTER(CArray), POINTER(CArray)]
lib.sub_broadcasted_array.restype = POINTER(CArray)
lib.mul_array.argtypes = [POINTER(CArray), POINTER(CArray)]
lib.mul_array.restype = POINTER(CArray)
lib.mul_scalar_array.argtypes = [POINTER(CArray), c_float]
lib.mul_scalar_array.restype = POINTER(CArray)
lib.mul_broadcasted_array.argtypes = [POINTER(CArray), POINTER(CArray)]
lib.mul_broadcasted_array.restype = POINTER(CArray)
lib.div_array.argtypes = [POINTER(CArray), POINTER(CArray)]
lib.div_array.restype = POINTER(CArray)
lib.div_scalar_array.argtypes = [POINTER(CArray), c_float]
lib.div_scalar_array.restype = POINTER(CArray)
lib.div_broadcasted_array.argtypes = [POINTER(CArray), POINTER(CArray)]
lib.div_broadcasted_array.restype = POINTER(CArray)
lib.pow_array.argtypes = [POINTER(CArray), c_float]
lib.pow_array.restype = POINTER(CArray)
lib.pow_array.argtypes = [c_float, POINTER(CArray)]
lib.pow_array.restype = POINTER(CArray) 

lib.sin_array.argtypes = [POINTER(CArray)]
lib.sin_array.restype = POINTER(CArray)
lib.sinh_array.argtypes = [POINTER(CArray)]
lib.sinh_array.restype = POINTER(CArray)
lib.cos_array.argtypes = [POINTER(CArray)]
lib.cos_array.restype = POINTER(CArray)
lib.cosh_array.argtypes = [POINTER(CArray)]
lib.cosh_array.restype = POINTER(CArray)
lib.tan_array.argtypes = [POINTER(CArray)]
lib.tan_array.restype = POINTER(CArray)
lib.tanh_array.argtypes = [POINTER(CArray)]
lib.tanh_array.restype = POINTER(CArray)

lib.transpose_array.argtypes = [POINTER(CArray)]
lib.transpose_array.restype = POINTER(CArray)
lib.equal_array.argtypes = [POINTER(CArray), POINTER(CArray)]
lib.equal_array.restype = POINTER(CArray)
lib.reshape_array.argtypes = [POINTER(CArray), POINTER(c_int), c_int]
lib.reshape_array.restype = POINTER(CArray)

# utils functions ---
lib.zeros_like_array.argtypes = [POINTER(CArray)]
lib.zeros_like_array.restype = POINTER(CArray)
lib.ones_like_array.argtypes = [POINTER(CArray)]
lib.ones_like_array.restype = POINTER(CArray)
lib.zeros_array.argtypes = [POINTER(c_int), c_size_t, c_size_t]
lib.zeros_array.restype = POINTER(CArray)
lib.ones_array.argtypes = [POINTER(c_int), c_size_t, c_size_t]
lib.ones_array.restype = POINTER(CArray)
lib.randn_array.argtypes = [POINTER(c_int), c_size_t, c_size_t]
lib.randn_array.restype = POINTER(CArray)
lib.randint_array.argtypes = [c_int, c_int, POINTER(c_int), c_size_t, c_size_t]
lib.randint_array.restype = POINTER(CArray)
lib.uinform_array.argtypes = [c_int, c_int, POINTER(c_int), c_size_t, c_size_t]
lib.uinform_array.restype = POINTER(CArray)
lib.fill_array.argtypes = [c_float, POINTER(c_int), c_size_t, c_size_t]
lib.fill_array.restype = POINTER(CArray)
lib.linspace_array.argtypes = [c_float, c_float, c_float, POINTER(c_int), c_size_t, c_size_t]
lib.linspace_array.restype = POINTER(CArray)