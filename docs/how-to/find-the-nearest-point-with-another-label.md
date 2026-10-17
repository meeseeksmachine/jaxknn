# Find the nearest point with another label

The raw tree, [`jaxknn.tree`](../api/tree.md), can exclude every point that shares the query's label. It prunes whole subtrees that hold only that label. A typical use is bridging the components of a graph: give every point its component label, and each point's result is its nearest point in *another* component.

```python
import jax
import jax.numpy as jnp
from jaxknn import tree

points = jax.random.normal(jax.random.key(0), (300, 2))
labels = (points[:, 0] > 0).astype(jnp.int32)  # two "components"

t = tree.build_tree(points)
idx, d2 = tree.knn(t, points, 1, exclude=(labels, labels))

# Every point's nearest neighbour lies in the other component.
assert (labels[idx[:, 0]] != labels).all()
```

Notes:
- `exclude=(point_labels, query_labels)`: labels must be integers in `[0, len(points))`. Out-of-range labels raise `ValueError` eagerly.
- The raw tree returns **squared** distances, and does no input validation or coordinate scaling. Scale coordinates with [`jaxknn.pow2_scale`][] yourself if they may overflow float32.
- A query whose label every point shares gets the missing-neighbour sentinel: index `len(points)`, distance `inf`.
