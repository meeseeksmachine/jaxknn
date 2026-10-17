# Tune for large `k` or high dimension

The defaults (`leaf_size=16`, `frontier=16`) suit low-dimensional data and small `k`. Two settings help outside that range:

| situation | setting | measured effect |
|---|---|---|
| large `k` (≥ 32) | `frontier` near `k`, e.g. `frontier=32` at `k=32` | 3–15% faster |
| high dimension (`d` ≥ 6) | `leaf_size=32` | about 1.45× faster |

```python
import jax
import jaxknn

points = jax.random.normal(jax.random.key(0), (2000, 8))  # d = 8
idx, dist = jaxknn.knn(points, 32, leaf_size=32, frontier=32)
assert idx.shape == (2000, 32)
```

The results are exact for any setting; these only change speed. In high dimension every query overflows the default frontier, and the tree is 30–50× slower than SciPy's `cKDTree`. If you don't need tracing, see [when to use SciPy instead](../when-to-use-scipy-instead.md).
