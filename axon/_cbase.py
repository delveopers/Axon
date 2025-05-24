import ctypes, os
from ctypes import Structure, c_float, c_int, c_size_t, POINTER
from typing import *

lib_path = os.path.join(os.path.dirname(__file__), 'build/libscalar.so')
lib = ctypes.CDLL(lib_path)

class CArray(Structure):
  pass

CArray._fields_ = [
  ("data", POINTER(float)),
  ("shape", POINTER(int)),
  ("strides", POINTER(int)),
  ("backstrides", POINTER(int)),
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
lib.print_tensor.argtypes = [POINTER(CArray)]
lib.print_tensor.restype = None

# maths ops ----
lib.add_array.argtypes = [POINTER(CArray), POINTER(CArray)]
lib.add_array.restype = None
lib.sub_array.argtypes = [POINTER(CArray), POINTER(CArray)]
lib.sub_array.restype = None
lib.mul_array.argtypes = [POINTER(CArray), POINTER(CArray)]
lib.mul_array.restype = None
lib.div_array.argtypes = [POINTER(CArray), POINTER(CArray)]
lib.div_array.restype = None

lib.sin_array.argtypes = [POINTER(CArray)]
lib.sin_array.restype = None
lib.sinh_array.argtypes = [POINTER(CArray)]
lib.sinh_array.restype = None
lib.cos_array.argtypes = [POINTER(CArray)]
lib.cos_array.restype = None
lib.cosh_array.argtypes = [POINTER(CArray)]
lib.cosh_array.restype = None
lib.tan_array.argtypes = [POINTER(CArray)]
lib.tan_array.restype = None
lib.tanh_array.argtypes = [POINTER(CArray)]
lib.tanh_array.restype = None