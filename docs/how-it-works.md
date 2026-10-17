# How it works

## The tree

[`jaxknn.tree.build_tree`][] builds a balanced, pointer-free kd-tree with static shapes:
- **Splits.** Each node splits at the exact median of its widest dimension.
- **Leaves.** Points are padded to a power-of-two number of leaves, each holding `leaf_size` points.
- **Layout.** Node `i` at level `l` is a contiguous slice of the permuted points, so the tree is just arrays: no pointers, no recursion.

## A query

For each query:

1. **Bound.** The k-th distance to the points in an aligned block of leaves around the query's own leaf bounds the true k-th distance.
2. **Candidates.** A level-by-level descent keeps every leaf whose cell lies within the bound, up to a static `frontier` cap.
3. **Merge.** Distances are computed by direct differences (not `|a|² + |b|² − 2a·b`, which cancels) to the candidate points, then an exact k-smallest selection that sends ties to the lower index.
4. **Overflow.** Queries whose candidates overflowed `frontier` are compacted and rerun in wider tiers (128, then 512 leaves), and finally by brute force. Each tier tightens the bound, so the answer is always exact.

Nothing uses `lax.cond`: under `vmap` it becomes a `select` that runs both branches.

## The safe layer

[`jaxknn.knn`][] wraps the tree:

- **Validation.** `k ≥ 1`, `(n, d)` shapes, and finite inputs: `ValueError` eagerly, a runtime error under `jit`.
- **Exact scaling.** [`jaxknn.pow2_scale`][] divides the coordinates by a power of two when their magnitude leaves 2⁻²⁰ to 2⁶⁰. Dividing by a power of two is exact, so neighbour selection, ties included, is unchanged. It only stops float32 squared distances from overflowing or underflowing.
- **Size buckets.** Eager calls pad `n` up to a [`jaxknn.bucket`][] size with [`jaxknn.far_rows`][]. Those rows are farther from every real point than the data's diameter, so they never displace a real neighbour, and inputs of different sizes reuse compiled code.
- **Gradients.** The tree selects indices on stop-gradiented inputs, then distances are recomputed by a plain gather, so they differentiate with the selection held fixed.
