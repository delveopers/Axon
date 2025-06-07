# Array Library Documentation

This documentation outlines the API and functionalities of the custom array library, similar in style to NumPy and PyTorch, offering multi-dimensional arrays with support for element-wise operations, reshaping, broadcasting, and statistical computations.

## Core Class: `array`

### Constructor

```python
array(data: Union[List[Any], int, float], dtype: str = 'float32')
```

Initializes an array from nested lists, scalars, or another array. Supports float and integer dtypes.

**Parameters:**

* `data`: Input data (nested lists, float, or int).
* `dtype`: Data type (e.g., 'float32', 'int64').

**Attributes:**

* `shape`: Tuple of array dimensions
* `ndim`: Number of dimensions
* `size`: Total number of elements
* `strides`: Stride values for each dimension
* `dtype`: Internal dtype as enum

## Type Handling

### `_parse_dtype(dtype: str) -> int`

Converts string dtype into internal enum.

### `_get_dtype_name() -> str`

Returns string representation of the internal dtype.

## Dtype Constants

Defined as class-level constants:

```python
int8, int16, int32, int64, long
float32, float64, double
uint8, uint16, uint32, uint64
boolean
```

## Representation

### `__repr__()` / `__str__()`

Returns formatted string or delegates printing to backend.

## Dtype Conversion

### `astype(dtype: str) -> array`

Returns a copy of the array cast to the specified dtype.

## Arithmetic Operations

Supported via operator overloading:

```python
+  -> __add__, __radd__
-  -> __sub__, __rsub__
*  -> __mul__, __rmul__
/  -> __truediv__, __rtruediv__
** -> __pow__, __rpow__ (partially)
== -> __eq__
```

## Trigonometric Operations

* `sin()`
* `cos()`
* `tan()`
* `sinh()`
* `cosh()`
* `tanh()`

Each returns a new array with element-wise trigonometric results.

## Shape Manipulation

* `reshape(new_shape: List[int])`
* `transpose()` *(supports up to 3D)*
* `flatten()`
* `squeeze(axis: int = -1)`
* `expand_dims(axis: int)`

## Reduction Operations

* `sum(axis: int = -1, keepdims: bool = False)`
* `mean(axis: int = -1, keepdims: bool = False)`
* `max(axis: int = -1, keepdims: bool = False)`
* `min(axis: int = -1, keepdims: bool = False)`
* `var(axis: int = -1, ddof: int = 0)`
* `std(axis: int = -1, ddof: int = 0)`

All return an array with the reduced axis removed or kept, based on `keepdims`.

# Utility Functions (from `_utils.py`)

## Array Creation

```python
zeros(shape, dtype=DType.FLOAT32) -> array
ones(shape, dtype=DType.FLOAT32) -> array
randn(shape, dtype=DType.FLOAT32) -> array
randint(low, high, shape, dtype=DType.INT32) -> array
uniform(low, high, shape, dtype=DType.FLOAT32) -> array
fill(fill_val, shape, dtype=DType.FLOAT32) -> array
linspace(start, step, end, shape, dtype=DType.FLOAT32) -> array
```

## Array Constructors from Existing Arrays

```python
zeros_like(arr) -> array
ones_like(arr) -> array
```

## Internal Utilities

* `_process_shape(shape)` → Computes total size, dimension, and returns C-compatible shape array.
* `_parse_dtype(dtype)` → Converts str to internal enum dtype.

## Backend C Bindings

All computations (arithmetic, trigonometric, reduction, shape ops) are executed via `lib` bindings to a C backend (assumed from `lib.X_array()` usage).

## Notes

* The design separates low-level C structures (via `CArray`) from high-level `array` abstraction.
* Memory is managed manually through `ctypes`.
* Mimics NumPy in API, but backend seems optimized for lightweight execution.

### Example

```python
from module_name import array, zeros, ones

x = array([[1, 2], [3, 4]])
y = x + 2
z = x.mean(axis=1)
print(z)
```
