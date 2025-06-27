import pytest
from axon import array, zeros, ones, zeros_like, ones_like, randn, randint, uniform, fill, linspace

class TestArrayConstruction:
  def test_scalar_creation(self):
    a = array([5])
    assert a.tolist() == [5]
    assert a.dtype == "float32"
    assert a.size == 1
    assert a.ndim == 1

  def test_list_creation(self):
    a = array([1, 2, 3])
    assert a.tolist() == [1, 2, 3]
    assert a.shape == (3,)
    assert a.size == 3
    assert a.ndim == 1

  def test_nested_list_creation(self):
    a = array([[1, 2], [3, 4]])
    assert a.tolist() == [[1, 2], [3, 4]]
    assert a.shape == (2, 2)
    assert a.size == 4
    assert a.ndim == 2

  def test_dtype_specification(self):
    a = array([1, 2, 3], dtype="int32")
    assert a.dtype == "int32"

  def test_copy_constructor(self):
    a = array([1, 2, 3])
    b = array(a)
    assert b.tolist() == [1, 2, 3]
    assert b.shape == a.shape

class TestArrayProperties:
  def test_repr(self):
    a = array([1, 2, 3])
    assert "array([1.0, 2.0, 3.0], dtype=float32)" in repr(a)

  def test_contiguous_check(self):
    a = array([1, 2, 3])
    assert isinstance(a.is_contiguous(), bool)

  def test_view_check(self):
    a = array([1, 2, 3])
    assert isinstance(a.is_view(), bool)

class TestArrayOperations:
  def test_addition_scalar(self):
    a = array([1, 2, 3])
    result = a + 5
    assert result.tolist() == [6.0, 7.0, 8.0]

  def test_addition_array(self):
    a = array([1, 2, 3])
    b = array([4, 5, 6])
    result = a + b
    assert result.tolist() == [5.0, 7.0, 9.0]

  def test_subtraction_scalar(self):
    a = array([5, 6, 7])
    result = a - 2
    assert result.tolist() == [3.0, 4.0, 5.0]

  def test_subtraction_array(self):
    a = array([5, 6, 7])
    b = array([1, 2, 3])
    result = a - b
    assert result.tolist() == [4.0, 4.0, 4.0]

  def test_multiplication_scalar(self):
    a = array([1, 2, 3])
    result = a * 3
    assert result.tolist() == [3.0, 6.0, 9.0]

  def test_multiplication_array(self):
    a = array([1, 2, 3])
    b = array([2, 3, 4])
    result = a * b
    assert result.tolist() == [2.0, 6.0, 12.0]

  def test_division_scalar(self):
    a = array([6, 8, 10])
    result = a / 2
    assert result.tolist() == [3.0, 4.0, 5.0]

  def test_division_array(self):
    a = array([6, 8, 10])
    b = array([2, 2, 2])
    result = a / b
    assert result.tolist() == [3.0, 4.0, 5.0]

  def test_negation(self):
    a = array([1, -2, 3])
    result = -a
    assert result.tolist() == [-1.0, 2.0, -3.0]

  def test_power_scalar(self):
    a = array([2, 3, 4])
    result = a ** 2
    assert result.tolist() == [4.0, 9.0, 16.0]

  def test_reverse_power_scalar(self):
    a = array([2, 3, 4])
    result = 2 ** a
    assert result.tolist() == [4.0, 8.0, 16.0]

  def test_equality_scalar(self):
    a = array([1, 2, 3])
    result = a == 2
    # assuming boolean arrays return as lists of 1s and 0s
    assert len(result.tolist()) == 3

  def test_equality_array(self):
    a = array([1, 2, 3])
    b = array([1, 0, 3])
    result = a == b
    assert len(result.tolist()) == 3

class TestMathFunctions:
  def test_log(self):
    a = array([1, 2.718, 7.389])
    result = a.log()
    assert len(result.tolist()) == 3

  def test_sqrt(self):
    a = array([1, 4, 9])
    result = a.sqrt()
    expected = [1.0, 2.0, 3.0]
    for i, val in enumerate(result.tolist()):
      assert abs(val - expected[i]) < 0.001

  def test_exp(self):
    a = array([0, 1, 2])
    result = a.exp()
    assert len(result.tolist()) == 3

  def test_abs(self):
    a = array([-1, 2, -3])
    result = a.abs()
    assert result.tolist() == [1.0, 2.0, 3.0]

  def test_sin(self):
    a = array([0, 1.57, 3.14])
    result = a.sin()
    assert len(result.tolist()) == 3

  def test_cos(self):
    a = array([0, 1.57, 3.14])
    result = a.cos()
    assert len(result.tolist()) == 3

  def test_tan(self):
    a = array([0, 0.785, 1.57])
    result = a.tan()
    assert len(result.tolist()) == 3

  def test_sinh(self):
    a = array([0, 1, 2])
    result = a.sinh()
    assert len(result.tolist()) == 3

  def test_cosh(self):
    a = array([0, 1, 2])
    result = a.cosh()
    assert len(result.tolist()) == 3

  def test_tanh(self):
    a = array([0, 1, 2])
    result = a.tanh()
    assert len(result.tolist()) == 3

class TestLinearAlgebra:
  def test_matmul_2d(self):
    a = array([[1, 2], [3, 4]])
    b = array([[5, 6], [7, 8]])
    result = a @ b
    assert result.shape == (2, 2)

  def test_dot_product(self):
    a = array([1, 2, 3])
    b = array([4, 5, 6])
    result = a.dot(b)
    # dot product should be scalar: 1*4 + 2*5 + 3*6 = 32
    assert isinstance(result.tolist(), (int, float))

class TestArrayManipulation:
  def test_transpose_2d(self):
    a = array([[1, 2, 3], [4, 5, 6]])
    result = a.transpose()
    expected = [[1, 4], [2, 5], [3, 6]]
    assert result.tolist() == expected

  def test_reshape(self):
    a = array([1, 2, 3, 4, 5, 6])
    result = a.reshape((2, 3))
    assert result.shape == (2, 3)
    assert result.size == 6

  def test_reshape_invalid(self):
    a = array([1, 2, 3])
    with pytest.raises(ValueError):
      a.reshape((2, 3))  # incompatible size

  def test_squeeze_all(self):
    a = array([[[1], [2]], [[3], [4]]])
    result = a.squeeze()
    assert 1 not in result.shape

  def test_squeeze_axis(self):
    a = array([[[1, 2, 3]]])  # shape (1, 1, 3)
    result = a.squeeze(0)
    assert result.shape[0] != 1

  def test_expand_dims(self):
    a = array([1, 2, 3])
    result = a.expand_dims(0)
    assert result.ndim == a.ndim + 1

  def test_flatten(self):
    a = array([[1, 2], [3, 4]])
    result = a.flatten()
    assert result.shape == (4,)
    assert result.tolist() == [1.0, 2.0, 3.0, 4.0]

class TestReductions:
  def test_sum_all(self):
    a = array([1, 2, 3, 4])
    result = a.sum()
    assert result.tolist() == 10.0

  def test_sum_axis(self):
    a = array([[1, 2], [3, 4]])
    result = a.sum(axis=0)
    assert result.tolist() == [4.0, 6.0]

  def test_sum_keepdims(self):
    a = array([[1, 2], [3, 4]])
    result = a.sum(axis=0, keepdims=True)
    assert result.ndim == 2

  def test_mean_all(self):
    a = array([1, 2, 3, 4])
    result = a.mean()
    assert result.tolist() == 2.5

  def test_mean_axis(self):
    a = array([[1, 2], [3, 4]])
    result = a.mean(axis=0)
    assert result.tolist() == [2.0, 3.0]

  def test_max_all(self):
    a = array([1, 5, 2, 3])
    result = a.max()
    assert result.tolist() == 5.0

  def test_max_axis(self):
    a = array([[1, 5], [2, 3]])
    result = a.max(axis=0)
    assert result.tolist() == [2.0, 5.0]

  def test_min_all(self):
    a = array([1, 5, 2, 3])
    result = a.min()
    assert result.tolist() == 1.0

  def test_min_axis(self):
    a = array([[1, 5], [2, 3]])
    result = a.min(axis=0)
    assert result.tolist() == [1.0, 3.0]

  def test_var(self):
    a = array([1, 2, 3, 4])
    result = a.var()
    assert isinstance(result.tolist(), float)

  def test_std(self):
    a = array([1, 2, 3, 4])
    result = a.std()
    assert isinstance(result.tolist(), float)

class TestTypeConversion:
  def test_astype_int(self):
    a = array([1.5, 2.7, 3.1])
    result = a.astype("int32")
    assert result.dtype == "int32"

  def test_astype_float(self):
    a = array([1, 2, 3], dtype="int32")
    result = a.astype("float64")
    assert result.dtype == "float64"

class TestArrayCopy:
  def test_contiguous(self):
    a = array([1, 2, 3])
    b = a.contiguous()
    assert b.tolist() == a.tolist()

  def test_make_contiguous(self):
    a = array([1, 2, 3])
    a.make_contiguous()
    assert a.is_contiguous()

  def test_view(self):
    a = array([1, 2, 3])
    b = a.view()
    assert b.tolist() == a.tolist()

class TestUtilityFunctions:
  def test_zeros(self):
    a = zeros(3, 4)
    assert a.shape == (3, 4)
    assert all(x == 0 for row in a.tolist() for x in row)

  def test_ones(self):
    a = ones(2, 3)
    assert a.shape == (2, 3)
    assert all(x == 1 for row in a.tolist() for x in row)

  def test_zeros_like(self):
    a = array([[1, 2], [3, 4]])
    b = zeros_like(a)
    assert b.shape == a.shape
    assert all(x == 0 for row in b.tolist() for x in row)

  def test_ones_like(self):
    a = array([[1, 2], [3, 4]])
    b = ones_like(a)
    assert b.shape == a.shape
    assert all(x == 1 for row in b.tolist() for x in row)

  def test_randn(self):
    a = randn(3, 4)
    assert a.shape == (3, 4)
    assert a.dtype == "float32"

  def test_randint(self):
    a = randint(0, 10, 3, 4)
    assert a.shape == (3, 4)
    assert a.dtype == "int32"

  def test_uniform(self):
    a = uniform(0, 1, 3, 4)
    assert a.shape == (3, 4)
    assert a.dtype == "float32"

  def test_fill(self):
    a = fill(7, 2, 3)
    assert a.shape == (2, 3)
    assert all(x == 7 for row in a.tolist() for x in row)

  def test_linspace(self):
    a = linspace(0, 1, 10, 5)
    assert a.shape == (5,)
    assert a.dtype == "float32"

class TestEdgeCases:
  def test_empty_array(self):
    a = array([])
    assert a.size == 0
    assert a.tolist() == []

  def test_single_element_2d(self):
    a = array([[5]])
    assert a.shape == (1, 1)
    assert a.tolist() == [[5.0]]

  def test_chain_operations(self):
    a = array([1, 2, 3])
    result = ((a + 1) * 2).sqrt()
    assert len(result.tolist()) == 3

  def test_mixed_dtype_operations(self):
    a = array([1, 2, 3], dtype="int32")
    b = array([1.5, 2.5, 3.5], dtype="float32")
    result = a + b
    assert len(result.tolist()) == 3

class TestErrorHandling:
  def test_invalid_squeeze_axis(self):
    a = array([[1, 2], [3, 4]])  # no dimension of size 1
    with pytest.raises(ValueError):
      a.squeeze(0)

  def test_transpose_high_dim(self):
    a = array([[[[1]]]])  # 4d array
    with pytest.raises(AssertionError):
      a.transpose()

if __name__ == "__main__":
  pytest.main([__file__, "-v"])