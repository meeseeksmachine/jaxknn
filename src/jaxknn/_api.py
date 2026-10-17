"""The safe kNN entry points: validation, exact scaling, eager size buckets."""

__all__: tuple[str, ...] = ("brute_knn", "bucket", "far_rows", "knn", "pow2_scale")

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, Float, Int

from . import tree as _tree

KnnOut = tuple[Int[Array, "m k"], Float[Array, "m k"]]

_NOT_FINITE = "kNN inputs must be finite (no NaN or inf)."

# Coordinates whose largest magnitude has binary exponent in this window are
# left unscaled; outside it they are moved to its nearer edge. 2**60 keeps
# float32 squared distances finite in up to 64 dimensions (d * 2**122 < 2**128);
# 2**-20 keeps them well above float32's smallest normal (2**-126).
_SCALE_WINDOW = (-20, 60)


def _traced(*xs: object) -> bool:
    """Whether any input is a JAX tracer (inside jit/vmap/grad)."""
    return any(isinstance(x, jax.core.Tracer) for x in xs)


def _value_error_if[T](x: T, pred: object, msg: str, /) -> T:
    """``ValueError`` when ``pred`` is concrete and true; ``eqx.error_if`` traced."""
    if _traced(pred):
        return eqx.error_if(x, pred, msg)
    if bool(pred):
        raise ValueError(msg)
    return x


def _as_float(x: object, /) -> Array:
    """Promote integer / low-precision inputs to at least float32."""
    x = jnp.asarray(x)
    return x.astype(jnp.promote_types(x.dtype, jnp.float32))


def _safe_sqrt(d2: Array, /) -> Array:
    """Euclidean distance with gradient 0 (not NaN) at coincident points."""
    pos = d2 > 0
    return jnp.where(pos, jnp.sqrt(jnp.where(pos, d2, 1.0)), 0.0)


def _gathered_distance(points: Array, queries: Array | None, ii: Array, /) -> Array:
    """Euclidean distance to each selected neighbour, recomputed from coordinates.

    The tree selects indices on stop-gradiented inputs (its overflow tiers use
    ``lax.while_loop``, which reverse-mode autodiff cannot cross); recomputing the
    distances by a plain gather makes them differentiable w.r.t. the points
    with the neighbour selection held fixed. Sentinel indices give ``inf``.
    """
    n = points.shape[0]
    q = points if queries is None else queries
    nb = points.at[ii].get(mode="fill", fill_value=0)
    diff = q[:, None] - nb
    return jnp.where(ii >= n, jnp.inf, _safe_sqrt(jnp.sum(diff * diff, -1)))


def _check_args(points: ArrayLike, k: object, queries: ArrayLike | None, /) -> None:
    """Check ``k >= 1`` and (n, d) shapes."""
    if isinstance(k, bool) or not isinstance(k, int | np.integer) or k < 1:
        msg = f"k must be an integer >= 1, got {k!r}."
        raise ValueError(msg)
    if jnp.ndim(points) != 2:
        msg = f"points must have shape (n, d), got {jnp.shape(points)}."
        raise ValueError(msg)
    if queries is not None and (
        jnp.ndim(queries) != 2 or jnp.shape(queries)[1] != jnp.shape(points)[1]
    ):
        msg = (
            f"queries must have shape (m, {jnp.shape(points)[1]}) to match "
            f"points, got {jnp.shape(queries)}."
        )
        raise ValueError(msg)


def _check_sizes(**sizes: int) -> None:
    for name, value in sizes.items():
        if value < 1:
            msg = f"{name} must be >= 1, got {value}."
            raise ValueError(msg)


def pow2_scale(
    points: Float[Array, "n d"], queries: Float[Array, "m d"] | None = None, /
) -> Float[Array, ""]:
    """Power of two to divide coordinates by so squared distances stay finite.

    1 (no scaling) unless the largest coordinate magnitude falls outside
    ``2**-20 .. 2**60``. Dividing by a power of two is exact, so neighbour
    selection, ties included, is unchanged; it only keeps squared distances
    from overflowing float32 (coordinate gaps above ~1.8e19, e.g. metres at
    kpc scale) or underflowing to zero for tiny coordinates. Scaling only out
    of range, rather than always normalising, keeps a tight cluster next to a
    distant outlier from underflowing. Gradient-free.
    """
    m = jnp.max(jnp.abs(points), initial=0.0)
    if queries is not None:
        m = jnp.maximum(m, jnp.max(jnp.abs(queries), initial=0.0))
    _, e = jnp.frexp(m)  # m = f * 2**e, f in [0.5, 1); e = 0 for m = 0
    lo, hi = _SCALE_WINDOW
    shift = e - jnp.clip(e, lo, hi)
    return jax.lax.stop_gradient(jnp.ldexp(jnp.ones((), points.dtype), shift))


def _prep(
    points: ArrayLike, k: int, queries: ArrayLike | None, /
) -> tuple[Array, Array | None, Array]:
    """Validate, cast and rescale inputs: ``(points, queries, scale)``."""
    _check_args(points, k, queries)
    q = None if queries is None else _as_float(queries)
    points = _as_float(points)
    bad = ~jnp.all(jnp.isfinite(points))
    if q is not None:
        bad = bad | ~jnp.all(jnp.isfinite(q))
    points = _value_error_if(points, bad, _NOT_FINITE)
    scale = pow2_scale(points, q)
    return points / scale, None if q is None else q / scale, scale


def bucket(n: int, /) -> int:
    """Smallest value in {2**j, 3 * 2**(j-1)} that is >= n (and >= 1).

    The eager size buckets of :func:`knn`: inputs whose sizes share a bucket
    share compiled code.
    """
    if n <= 1:
        return 1
    p = 1 << (n - 1).bit_length()  # next power of two >= n
    return p * 3 // 4 if p >= 4 and p * 3 // 4 >= n else p


def far_rows(points: Float[Array, "n d"], count: int, /) -> Float[Array, "count d"]:
    """``count`` distinct rows farther from every row of ``points`` than its diameter.

    Padding a search with these never changes the k nearest real neighbours of
    any query that was included in the array passed here (so pass the union of
    points and queries), while at least k real points exist. Placed relative to
    the data (no absolute offset, so tiny data is not swamped) and one ``span``
    apart: identical rows made every padding query's own search overflow into
    brute force (~20x slower eager :func:`knn` just above a bucket size).
    """
    d = points.shape[1]
    lo, hi = points.min(0), points.max(0)
    # Cover the magnitude too: in float32, hi + c * spread rounds back to hi
    # once |hi| / spread >~ 1e7, which would put "far" rows on real points.
    span = jnp.maximum(jnp.max(hi - lo), jnp.max(jnp.abs(jnp.stack([lo, hi]))))
    span = jnp.where(span > 0, span, 1.0)  # all-zero data
    offset = (2.0 * math.sqrt(d) + 1.0 + jnp.arange(count, dtype=points.dtype)) * span
    return jnp.broadcast_to(lo, (count, d)).at[:, 0].set(hi[0] + offset)


def _no_neighbors(m: int, k: int, dtype: object, /) -> KnnOut:
    """Return the empty-search result: index 0 (= n), distance inf, every slot."""
    return jnp.zeros((m, k), jnp.int32), jnp.full((m, k), jnp.inf, dtype)


def _knn_core(
    points: Array, queries: Array | None, k: int, leaf_size: int, frontier: int, /
) -> KnnOut:
    if queries is None:
        return _tree.all_knn(points, k, leaf_size=leaf_size, frontier=frontier)
    t = _tree.build_tree(points, leaf_size=leaf_size)
    return _tree.knn(t, queries, k, frontier=frontier)


_knn_core_jit = jax.jit(_knn_core, static_argnums=(2, 3, 4))


def knn(
    points: ArrayLike,
    /,
    k: int,
    *,
    queries: ArrayLike | None = None,
    leaf_size: int = 16,
    frontier: int = 16,
) -> KnnOut:
    """Exact k nearest neighbours with a kd-tree. Traceable under jit/vmap/grad.

    ``knn(points, k)`` returns, for every point, its k nearest *other* points
    (self excluded by index); ``knn(points, k, queries=q)`` the k nearest
    points to each query. Both give ``(indices, distances)``: Euclidean
    distances, rows sorted ascending; a missing neighbour (fewer than k
    candidates) is index ``len(points)`` with distance ``inf``. Equidistant
    neighbours go to the lower index (identically eager and under jit, for
    indices below 2**24). Distances are differentiable w.r.t. the coordinates,
    with the neighbour selection held fixed.

    Inputs must be finite (``ValueError`` eagerly, a runtime error traced).
    Integer and low-precision inputs are promoted to float32; coordinates are
    divided by an exact power of two (:func:`pow2_scale`) when needed, so
    float32 coordinates up to ~3e38 or down to ~1e-38 keep their neighbours.

    Eager calls pad ``n`` (and the number of queries) to a size
    :func:`bucket` with :func:`far_rows`, so differing sizes share compiled
    code; under ``jit`` the caller's shapes are used as-is.

    Tuning: for large ``k`` (``>= 32``) set ``frontier`` near ``k``; in high
    dimension (``d >= 6``) ``leaf_size=32`` measured ~1.45x faster.
    """
    _check_sizes(leaf_size=leaf_size, frontier=frontier)
    points, queries, scale = _prep(points, k, queries)
    n = points.shape[0]
    m = n if queries is None else queries.shape[0]
    if m == 0 or n == 0:
        return _no_neighbors(m, k, points.dtype)
    if _traced(points, queries):
        sq = None if queries is None else jax.lax.stop_gradient(queries)
        ii, _ = _knn_core_jit(jax.lax.stop_gradient(points), sq, k, leaf_size, frontier)
        return ii, _gathered_distance(points, queries, ii) * scale
    extent = points if queries is None else jnp.concat([points, queries])
    padded = jnp.concat([points, far_rows(extent, bucket(n) - n)])
    qpad = None
    if queries is not None:
        filler = jnp.broadcast_to(queries[:1], (bucket(m) - m, points.shape[1]))
        qpad = jnp.concat([queries, filler])
    ii, d2 = _knn_core_jit(padded, qpad, k, leaf_size, frontier)
    ii, d2 = ii[:m], d2[:m]
    fake = ii >= n  # only when fewer than k real candidates exist
    dist = _safe_sqrt(jnp.where(fake, jnp.inf, d2)) * scale
    return jnp.where(fake, n, ii), dist


def brute_knn(
    points: ArrayLike,
    /,
    k: int,
    *,
    queries: ArrayLike | None = None,
    chunk: int = 1024,
) -> KnnOut:
    """Exact k nearest neighbours by brute force, ``chunk`` queries at a time.

    The same contract as :func:`knn` (validation, scaling, ties to the lower
    index, sentinels). Traceable; faster than :func:`knn` for small inputs.
    """
    _check_sizes(chunk=chunk)
    points, q, scale = _prep(points, k, queries)
    ii, d2 = _tree.brute_knn(points, k, queries=q, chunk=chunk)
    return ii, _safe_sqrt(d2) * scale
