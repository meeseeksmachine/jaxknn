"""The raw exact kd-tree: build, query, label-excluding queries, brute force.

Lower level than :func:`jaxknn.knn`: no input validation, no coordinate
scaling and no eager size buckets, and distances are squared. Use it for
custom queries such as ``knn(tree, q, k, exclude=(point_labels, query_labels))``.
"""

__all__: tuple[str, ...] = (
    "Tree",
    "all_knn",
    "brute_knn",
    "build_tree",
    "knn",
    "locate_leaves",
    "node_labels",
)

from ._brute import brute_knn
from ._build import Tree, build_tree
from ._query import all_knn, knn, locate_leaves, node_labels
