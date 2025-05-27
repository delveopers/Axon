# Axon

Lightweight multi-dimensional array manipulation library powered by GPU, similar to NumPy, but trying to be better.

## Features

- Custom `array` class with Pythonic syntax
- Element-wise arithmetic: `+`, `-`, `-`, `/`
- Scalar operations (e.g., `array + 5`)
- Trigonometric functions: `sin`, `cos`, `tan`, etc.
- Auto handling of `CArray`, scalars, and lists
- Simple `__str__`/`__repr__` for pretty printing

## Requirements

- Python 3.7+
- C compiler (for building the C backend)
- ctypes module (standard in Python)

## Getting Started

### Build

To use Axon, make sure you have compiled the C backend to a shared library (`.dll`, `.so`, or `.dylib`) and exposed the C functions via `ctypes`.

Place the compiled `.dll` (on Windows) or `.so` (Linux/macOS) in your `axon/` folder.

## Example

Here's a quick demo of how Axon works:

```python
from axon import array

a1 = array([[2, -3, 5], [-9, 0, -5]])
a2 = array([[2, -3, 5], [-9, 0, -5]])

c = a1 + a2
d = c + 5

print(a1)
print(a2)
print(c)
print(d.sin())

print(d.shape, d.size, d.ndim)
print(c.shape, c.size, c.ndim)
```

### Output

```text
axon.array([
    [2.00, -3.00, 5.00],
    [-9.00, 0.00, -5.00]
])

axon.array([
    [2.00, -3.00, 5.00],
    [-9.00, 0.00, -5.00]
])

axon.array([
    [4.00, -6.00, 10.00],
    [-18.00, 0.00, -10.00]
])

axon.array([
    [0.41, -0.84, 0.65],
    [-0.42, -0.96, 0.96]
])

(2, 3) 6 2
(2, 3) 6 2
```

## Internals

The `array` class wraps a C structure called `CArray` which stores:

- A flattened data array
- Shape and dimension metadata
- Functions for operations are delegated to native code via `ctypes`.

## API

```python
array(data: Union[list, int, float])
```

### Operators

- `a + b` / `a + 5`
- `a - b` / `a - 2.5`
- `a * b` / `a * 3`
- `a / b` / `a / 1.5`

### Trig Functions

- `a.sin()`
- `a.cos()`
- `a.tan()`
- `a.sinh()`, `cosh()`, `tanh()`

## License

This project is under the [Apache-2.0](./LICENSE) License.
