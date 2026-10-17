"""Exact k-nearest neighbours in JAX: a kd-tree that traces under jit/vmap/grad.

The entry points are :func:`knn` (kd-tree) and :func:`brute_knn`. Both
validate inputs, scale coordinates by an exact power of two when float32
would overflow or underflow, and give ties to the lower index. :mod:`jaxknn.tree`
holds the raw tree for custom queries.
"""

__all__: tuple[str, ...] = (
    "__version__",
    "brute_knn",
    "bucket",
    "far_rows",
    "knn",
    "pow2_scale",
    "tree",
)

from . import tree
from ._api import brute_knn, bucket, far_rows, knn, pow2_scale
from ._version import version as __version__
