def get_shape(data: list):
  if isinstance(data, list):
    return [len(data), ] + get_shape(data[0])
  else:
    return []

def flatten(data: list) -> list:
  if isinstance(data, list):
    return [item for sublist in data for item in flatten(sublist)]
  else:
    return [data]

def get_size(shape:tuple) -> list:
  out = 1
  for dim in shape:
    out *= dim
  return out

def get_strides(shape:tuple) -> list:
  strides = [1]
  for size in reversed(shape[:-1]):
    strides.append(strides[-1] * size)
  return list(reversed(strides))

def transposed_shape(shape):
  ndim = len(shape)
  if ndim == 1:
    return shape  # no transpose for 1D
  elif ndim == 2:
    rows, cols = shape
    return [cols, rows]
  elif ndim == 3:
    batch, rows, cols = shape
    return [batch, cols, rows]
  else:
    raise ValueError(f"Unsupported shape dimension: {ndim}")