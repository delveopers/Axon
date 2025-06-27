from ctypes import c_float, c_size_t, c_int, c_bool
from typing import *

from ._cbase import CArray, lib, DType
from ._helpers import ShapeHelp, DtypeHelp

int8, int16, int32, int64, long = "int8", "int16", "int32", "int64", "long"
float32, float64, double = "float32", "float64", "double"
uint8, uint16, uint32, uint64 = "uint8", "uint16", "uint32", "uint64"
boolean = "bool"

class array:
  int8, int16, int32, int64, long, float32, float64, double, uint8, uint16, uint32, uint64, boolean = int8, int16, int32, int64, long, float32, float64, double, uint8, uint16, uint32, uint64, boolean
  def __init__(self, data: Union[List[Any], int, float], dtype: str=float32):
    if isinstance(data, CArray): self.data, self.shape, self.size, self.ndim, self.strides, self.dtype = data, (), 0, 0, [], dtype or "float32"
    elif isinstance(data, array): self.data, self.shape, self.dtype, self.size, self.ndim, self.strides = data.data, data.shape, dtype or data.dtype, data.size, data.ndim, data.strides
    else:
      data, shape = ShapeHelp.flatten([data] if isinstance(data, (int, float)) else data), tuple(ShapeHelp.get_shape(data))
      self.size, self.ndim, self.dtype, self.shape, self.strides = len(data), len(shape), dtype or "float32", shape, ShapeHelp.get_strides(shape)
      self._data_ctypes, self._shape_ctypes = (c_float * self.size)(*data.copy()), (c_int * self.ndim)(*shape)
      self.data = lib.create_array(self._data_ctypes, c_size_t(self.ndim), self._shape_ctypes, c_size_t(self.size), c_int(DtypeHelp._parse_dtype(self.dtype)))

  def __repr__(self) -> str: return f"array({self.tolist()}, dtype={self.dtype})"
  def __str__(self) -> str: return (lib.print_array(self.data), "")[1]
  def is_contiguous(self) -> bool: return bool(lib.is_contiguous_array(self.data))
  def is_view(self) -> bool:  return bool(lib.is_view_array(self.data))

  def astype(self, dtype: str) -> "array":
    out = array(lib.cast_array(self.data, c_int(DtypeHelp._parse_dtype(dtype))).contents, dtype)
    out.shape, out.size, out.ndim, out.strides = self.shape, self.size, self.ndim, self.strides
    return out
  def contiguous(self) -> "array":
    out = array(lib.contiguous_array(self.data).contents, self.dtype)
    out.shape, out.size, out.ndim, out.strides = self.shape, self.size, self.ndim, self.strides
    return out
  def make_contiguous(self) -> None:
    lib.make_contiguous_inplace_array(self.data)
    self.strides = ShapeHelp.get_strides(self.shape)  # updating strides since they may have changed
  def view(self) -> "array":
    out = array(lib.view_array(self.data).contents, self.dtype)
    out.shape, out.size, out.ndim, out.strides = self.shape, self.size, self.ndim, self.strides
    return out
  def tolist(self) -> List[Any]:
    data_ptr = lib.out_data(self.data)
    data_array = [data_ptr[i] for i in range(self.size)]
    if self.ndim == 0: return data_array[0]
    elif self.ndim == 1: return data_array
    else: return ShapeHelp.reshape_list(data_array, self.shape)

  def __add__(self, other) -> "array":
    result_ptr = lib.add_scalar_array(self.data, c_float(other)).contents if isinstance(other, (int, float)) else lib.add_array(self.data, (other if isinstance(other, (CArray, array)) else array(other, self.dtype)).data).contents
    out = array(result_ptr, self.dtype)
    return (setattr(out, "shape", self.shape), setattr(out, "size", self.size), setattr(out, "ndim", self.ndim), setattr(out, "strides", self.strides), out)[4]

  def __sub__(self, other) -> "array":
    result_ptr = lib.sub_scalar_array(self.data, c_float(other)).contents if isinstance(other, (int, float)) else lib.sub_array(self.data, (other if isinstance(other, (CArray, array)) else array(other, self.dtype)).data).contents
    out = array(result_ptr, self.dtype)
    return (setattr(out, "shape", self.shape), setattr(out, "size", self.size), setattr(out, "ndim", self.ndim), setattr(out, "strides", self.strides), out)[4]

  def __mul__(self, other) -> "array":
    result_ptr = lib.mul_scalar_array(self.data, c_float(other)).contents if isinstance(other, (int, float)) else lib.mul_array(self.data, (other if isinstance(other, (CArray, array)) else array(other, self.dtype)).data).contents
    out = array(result_ptr, self.dtype)
    return (setattr(out, "shape", self.shape), setattr(out, "size", self.size), setattr(out, "ndim", self.ndim), setattr(out, "strides", self.strides), out)[4]

  def __truediv__(self, other) -> "array":
    result_ptr = lib.div_scalar_array(self.data, c_float(other)).contents if isinstance(other, (int, float)) else lib.div_array(self.data, (other if isinstance(other, (CArray, array)) else array(other, self.dtype)).data).contents
    out = array(result_ptr, self.dtype)
    return (setattr(out, "shape", self.shape), setattr(out, "size", self.size), setattr(out, "ndim", self.ndim), setattr(out, "strides", self.strides), out)[4]

  def __neg__(self) -> "array":
    result_pointer = lib.neg_array(self.data).contents
    out = array(result_pointer, self.dtype)
    return (setattr(out, "shape", self.shape), setattr(out, "size", self.size), setattr(out, "ndim", self.ndim), setattr(out, "strides", self.strides), out)[4]

  def __radd__(self, other): return self + other
  def __rsub__(self, other): return self - other
  def __rmul__(self, other): return self * other
  def __rtruediv__(self, other): return (self / other) ** -1

  def __neg__(self) -> "array":
    out = array(lib.neg_array(self.data).contents, self.dtype)
    return (setattr(out, "shape", self.shape), setattr(out, "size", self.size), setattr(out, "ndim", self.ndim), setattr(out, "strides", self.strides), out)[4]

  def __pow__(self, exp) -> "array":
    if isinstance(exp, (int, float)):
      result_ptr = lib.pow_array(self.data, c_float(exp)).contents
    out = array(result_ptr, self.dtype)
    return (setattr(out, "shape", self.shape), setattr(out, "size", self.size), setattr(out, "ndim", self.ndim), setattr(out, "strides", self.strides), out)[4]

  def __rpow__(self, base) -> "array":
    if isinstance(base, (int, float)):
      result_ptr = lib.pow_scalar(c_float(base), self.data).contents
    else: raise NotImplementedError("__rpow__ with array base not implemented yet")
    out = array(result_ptr, self.dtype)
    return (setattr(out, "shape", self.shape), setattr(out, "size", self.size), setattr(out, "ndim", self.ndim), setattr(out, "strides", self.strides), out)[4]

  def __eq__(self, other) -> "array":
    if isinstance(other, (int, float)):
      # For scalar comparison, create a scalar array first
      other = array([other], dtype=self._get_dtype_name())
    else: other = other if isinstance(other, (CArray, array)) else array(other)
    out = array(lib.equal_array(self.data, other.data).contents, DType.BOOL)
    return (setattr(out, "shape", self.shape), setattr(out, "size", self.size), setattr(out, "ndim", self.ndim), setattr(out, "strides", self.strides), out)[4]


  def log(self) -> "array":
    result_ptr = lib.log_array(self.data).contents
    out = array(result_ptr, self.dtype)
    out.shape, out.size, out.ndim, out.strides = self.shape, self.size, self.ndim, self.strides
    return (setattr(out, "shape", self.shape), setattr(out, "size", self.size), setattr(out, "ndim", self.ndim), setattr(out, "strides", self.strides), out)[4]

  def sqrt(self) -> "array":
    result_ptr = lib.sqrt_array(self.data).contents
    out = array(result_ptr, self.dtype)
    return (setattr(out, "shape", self.shape), setattr(out, "size", self.size), setattr(out, "ndim", self.ndim), setattr(out, "strides", self.strides), out)[4]

  def exp(self) -> "array":
    result_ptr = lib.exp_array(self.data).contents
    out = array(result_ptr, self.dtype)
    return (setattr(out, "shape", self.shape), setattr(out, "size", self.size), setattr(out, "ndim", self.ndim), setattr(out, "strides", self.strides), out)[4]

  def abs(self) -> "array":
    result_ptr = lib.abs_array(self.data).contents
    out = array(result_ptr, self.dtype)
    return (setattr(out, "shape", self.shape), setattr(out, "size", self.size), setattr(out, "ndim", self.ndim), setattr(out, "strides", self.strides), out)[4]

  def __matmul__(self, other):
    other = other if isinstance(other, (CArray, array)) else array(other, self.dtype)
    if self.ndim <= 2 and other.ndim <= 2:
      result_ptr = lib.matmul_array(self.data, other.data).contents
    elif self.ndim == 3 and other.ndim == 3 and self.shape[0] == other.shape[0]:
      result_ptr = lib.batch_matmul_array(self.data, other.data).contents
    else:
      result_ptr = lib.broadcasted_matmul_array(self.data, other.data).contents
    out = array(result_ptr, self.dtype, self.requires_grad)
    shape, ndim, size = lib.out_shape(out.data), self.ndim, lib.out_size(out.data)
    out.shape, out.ndim, out.size = tuple([shape[i] for i in range(ndim)]), ndim, size
    out.strides = ShapeHelp.get_strides(out.shape)
    return out

  def dot(self, other):
    other = other if isinstance(other, (CArray, array)) else array(other, self.dtype)
    if self.ndim <= 2 and other.ndim <= 2:
      result_ptr = lib.dot_array(self.data, other.data).contents
    elif self.ndim == 3 and other.ndim == 3 and self.shape[0] == other.shape[0]:
      result_ptr = lib.batch_dot_array(self.data, other.data).contents
    out = array(result_ptr, self.dtype, self.requires_grad)
    shape, ndim, size = lib.out_shape(result_ptr), out.data.ndim, lib.out_size(result_ptr)
    out.shape, out.ndim, out.size = tuple([shape[i] for i in range(ndim)]), ndim, size
    out.strides = ShapeHelp.get_strides(out.shape)
    return out

  def sin(self) -> "array":
    result_ptr = lib.sin_array(self.data).contents
    out = array(result_ptr, self.dtype)
    return (setattr(out, "shape", self.shape), setattr(out, "size", self.size), setattr(out, "ndim", self.ndim), setattr(out, "strides", self.strides), out)[4]

  def cos(self) -> "array":
    result_ptr = lib.cos_array(self.data).contents
    out = array(result_ptr, self.dtype)
    return (setattr(out, "shape", self.shape), setattr(out, "size", self.size), setattr(out, "ndim", self.ndim), setattr(out, "strides", self.strides), out)[4]

  def tan(self) -> "array":
    result_ptr = lib.tan_array(self.data).contents
    out = array(result_ptr, self.dtype)
    return (setattr(out, "shape", self.shape), setattr(out, "size", self.size), setattr(out, "ndim", self.ndim), setattr(out, "strides", self.strides), out)[4]

  def sinh(self) -> "array":
    result_ptr = lib.sinh_array(self.data).contents
    out = array(result_ptr, self.dtype)
    return (setattr(out, "shape", self.shape), setattr(out, "size", self.size), setattr(out, "ndim", self.ndim), setattr(out, "strides", self.strides), out)[4]

  def cosh(self) -> "array":
    result_ptr = lib.cosh_array(self.data).contents
    out = array(result_ptr, self.dtype)
    return (setattr(out, "shape", self.shape), setattr(out, "size", self.size), setattr(out, "ndim", self.ndim), setattr(out, "strides", self.strides), out)[4]

  def tanh(self) -> "array":
    result_ptr = lib.tanh_array(self.data).contents
    out = array(result_ptr, self.dtype)
    return (setattr(out, "shape", self.shape), setattr(out, "size", self.size), setattr(out, "ndim", self.ndim), setattr(out, "strides", self.strides), out)[4]

  def transpose(self) -> "array":
    assert self.ndim <= 3, ".transpose() only supported till 3-d arrays"
    out = array(lib.transpose_array(self.data).contents, self.dtype)
    out.shape, out.size, out.ndim = ShapeHelp.transposed_shape(self.shape), self.size, self.ndim
    out.strides = ShapeHelp.get_strides(out.shape)
    return out

  def reshape(self, new_shape: Union[List[int], Tuple[int]]) -> "array":
    if isinstance(new_shape, tuple): new_shape = list(new_shape)
    new_size, ndim = 1, len(new_shape)
    for dim in new_shape: new_size *= dim
    if new_size != self.size: raise ValueError(f"Cannot reshape array of size {self.size} into shape {new_shape}")
    result_ptr = lib.reshape_array(self.data, (c_int * ndim)(*new_shape), c_int(ndim)).contents
    out = array(result_ptr, self.dtype)
    out.shape, out.size, out.ndim = tuple(new_shape), self.size, ndim; out.strides = ShapeHelp.get_strides(new_shape)
    return out

  def squeeze(self, axis: int = -1) -> "array":
    result_ptr = lib.squeeze_array(self.data, c_int(axis)).contents
    out = array(result_ptr, self.dtype)
    if axis == -1: new_shape = [dim for dim in self.shape if dim != 1]      # Remove all dimensions of size 1
    else: # Remove specific axis if it has size 1
      if self.shape[axis] != 1: raise ValueError(f"Cannot squeeze axis {axis} with size {self.shape[axis]}")
      new_shape = list(self.shape); new_shape.pop(axis)
    out.shape = tuple(new_shape) if new_shape else (1,)
    out.size, out.ndim, out.strides = self.size, len(out.shape), ShapeHelp.get_strides(out.shape)
    return out

  def expand_dims(self, axis: int) -> "array":
    result_ptr = lib.expand_dims_array(self.data, c_int(axis)).contents
    out = array(result_ptr, self.dtype)
    new_shape = list(self.shape)
    if axis < 0: axis = len(new_shape) + axis + 1
    new_shape.insert(axis, 1)
    out.shape = tuple(new_shape)
    out.size, out.ndim, out.strides = self.size, len(out.shape), ShapeHelp.get_strides(out.shape)
    return out

  def flatten(self) -> "array":
    """Return a copy of the array collapsed into one dimension"""
    result_ptr = lib.flatten_array(self.data).contents
    out = array(result_ptr, self.dtype)
    out.shape = (self.size,)
    out.size, out.ndim, out.strides = self.size, 1, ShapeHelp.get_strides(out.shape)
    return out

  def sum(self, axis: int = -1, keepdims: bool = False) -> "array":
    out = array(lib.sum_array(self.data, c_int(axis), c_bool(keepdims)).contents, self.dtype)
    if axis == -1:out.shape, out.size, out.ndim = (1,) if keepdims else (), 1, 1 if keepdims else 0   # Sum all elements
    else:
      new_shape = list(self.shape)
      if keepdims: new_shape[axis] = 1
      else: new_shape.pop(axis)
      out.shape = tuple(new_shape)
      out.size, out.ndim, out.strides = 1 if not new_shape else eval('*'.join(map(str, new_shape))), len(new_shape), ShapeHelp.get_strides(out.shape) if out.shape else []
    return out

  def mean(self, axis: int = -1, keepdims: bool = False) -> "array":
    out = array(lib.mean_array(self.data, c_int(axis), c_bool(keepdims)).contents, self.dtype)
    if axis == -1: out.shape, out.size, out.ndim = (1,) if keepdims else (), 1, 1 if keepdims else 0
    else:
      new_shape = list(self.shape)
      if keepdims: new_shape[axis] = 1
      else: new_shape.pop(axis)
      out.shape = tuple(new_shape)
      out.size, out.ndim, out.strides = 1 if not new_shape else eval('*'.join(map(str, new_shape))), len(new_shape), ShapeHelp.get_strides(out.shape) if out.shape else []
    return out

  def max(self, axis: int = -1, keepdims: bool = False) -> "array":
    result_ptr = lib.max_array(self.data, c_int(axis), c_bool(keepdims)).contents
    out = array(result_ptr, self.dtype)
    
    if axis == -1: out.shape, out.size, out.ndim = (1,) if keepdims else (), 1, 1 if keepdims else 0
    else:
      new_shape = list(self.shape)
      if keepdims: new_shape[axis] = 1
      else: new_shape.pop(axis)
      out.shape = tuple(new_shape)
      out.size, out.ndim, out.strides = 1 if not new_shape else eval('*'.join(map(str, new_shape))), len(new_shape), ShapeHelp.get_strides(out.shape) if out.shape else []
    return out

  def min(self, axis: int = -1, keepdims: bool = False) -> "array":
    out = array(lib.min_array(self.data, c_int(axis), c_bool(keepdims)).contents, self.dtype)
    if axis == -1: out.shape, out.size, out.ndim = (1,) if keepdims else (), 1, 1 if keepdims else 0
    else:
      new_shape = list(self.shape)
      if keepdims: new_shape[axis] = 1
      else: new_shape.pop(axis)
      out.shape = tuple(new_shape)
      out.size, out.ndim, out.strides = 1 if not new_shape else eval('*'.join(map(str, new_shape))), len(new_shape), ShapeHelp.get_strides(out.shape) if out.shape else []
    return out

  def var(self, axis: int = -1, ddof: int = 0) -> "array":
    out = array(lib.var_array(self.data, c_int(axis), c_int(ddof)).contents, self.dtype)
    if axis == -1:
      out.shape, out.size, out.ndim = (), 1, 0
    else:
      new_shape = list(self.shape)
      new_shape.pop(axis)
      out.shape = tuple(new_shape)
      out.size, out.ndim, out.strides = 1 if not new_shape else eval('*'.join(map(str, new_shape))), len(new_shape), ShapeHelp.get_strides(out.shape) if out.shape else []
    return out

  def std(self, axis: int = -1, ddof: int = 0) -> "array":
    out = array(lib.std_array(self.data, c_int(axis), c_int(ddof)).contents, self.dtype)
    if axis == -1: out.shape, out.size, out.ndim = (), 1, 0
    else:
      new_shape = list(self.shape)
      new_shape.pop(axis)
      out.shape = tuple(new_shape)
      out.size, out.ndim, out.strides = 1 if not new_shape else eval('*'.join(map(str, new_shape))), len(new_shape), ShapeHelp.get_strides(out.shape) if out.shape else []
    return out