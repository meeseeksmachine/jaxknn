# Use it under `jit`, `vmap` and `grad`

[`jaxknn.knn`][] and [`jaxknn.brute_knn`][] are ordinary JAX functions: they trace, batch and differentiate.

## `jit`

`k` (and `leaf_size`, `frontier`) choose shapes, so keep them static, for example by closing over them:

```python
import jax
import jax.numpy as jnp
import jaxknn

points = jax.random.normal(jax.random.key(0), (500, 2))

knn5 = jax.jit(lambda p: jaxknn.knn(p, 5))
idx, dist = knn5(points)

# Same answer as the eager call, ties included.
assert (idx == jaxknn.knn(points, 5)[0]).all()
```

Eagerly, `knn` pads the number of points to a size bucket (2ʲ or 1.5·2ʲ), so inputs of different sizes share compiled code. Under `jit` your shapes are used as they are.

## `vmap`

```python
clouds = jax.random.normal(jax.random.key(1), (4, 200, 3))  # 4 independent clouds
idx, dist = jax.vmap(lambda p: jaxknn.knn(p, 3))(clouds)
assert idx.shape == (4, 200, 3)
```

## `grad`

Distances are differentiable with respect to the coordinates, with the neighbour selection held fixed. At coincident points the gradient is 0, not NaN.

```python
def mean_spacing(p):
    _, dist = jaxknn.knn(p, 1)
    return jnp.mean(dist)

g = jax.grad(mean_spacing)(points)
assert g.shape == points.shape
assert jnp.isfinite(g).all()
```

## Errors under tracing

NaN or inf input is an error. It's a `ValueError` when called eagerly. Under `jit` the check runs at execution time instead (`equinox.error_if`), so it surfaces as a runtime error with the same message.
