"""Tests for the safe entry points ``jaxknn.knn`` and ``jaxknn.brute_knn``."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests._compat import enable_x64

import jaxknn
from jaxknn import _api

FUNCS = [
    pytest.param(jaxknn.knn, id="knn"),
    pytest.param(jaxknn.brute_knn, id="brute"),
]


@pytest.fixture(params=FUNCS)
def search(request):
    """Each public kNN function."""
    return request.param


def _ref(points, k, queries=None):
    q = points if queries is None else queries
    d2 = ((q[:, None].astype(np.float64) - points[None]) ** 2).sum(-1)
    if queries is None:
        np.fill_diagonal(d2, np.inf)
    if points.shape[0] < k:
        d2 = np.concat([d2, np.full((len(q), k - points.shape[0]), np.inf)], axis=1)
    return np.sqrt(np.sort(d2, 1)[:, :k])


def _assert_indices_give_distances(points, queries, idx, dist):
    """Each finite neighbour index reproduces its reported distance."""
    q = points if queries is None else queries
    fin = np.isfinite(dist)
    rows = np.nonzero(fin)[0]
    got = np.linalg.norm(q[rows].astype(np.float64) - points[idx[fin]], axis=1)
    np.testing.assert_allclose(got, dist[fin], rtol=1e-5, atol=1e-6)


class TestContract:
    """Both functions honour the same contract."""

    @pytest.mark.parametrize("n", [0, 1, 2, 5, 11, 300])  # 5: padded, n < k
    def test_all_points(self, search, n):
        """Euclidean distances, sorted, self excluded, sentinel n when short."""
        p = np.random.default_rng(n).normal(size=(n, 3)).astype(np.float32)
        idx, dist = map(np.asarray, search(jnp.asarray(p), 10))
        ref = _ref(p, 10)
        np.testing.assert_array_equal(np.isfinite(dist), np.isfinite(ref))
        fin = np.isfinite(ref)
        np.testing.assert_allclose(dist[fin], ref[fin], rtol=1e-5, atol=1e-6)
        assert np.all(idx[~fin] == n)
        assert not np.any(idx == np.arange(n)[:, None])
        _assert_indices_give_distances(p, None, idx, dist)

    @pytest.mark.parametrize("k", [5, 40])
    def test_duplicates_excluded_by_index(self, search, k):
        """Coincident points are neighbours at distance 0; self never is."""
        p = np.random.default_rng(3).normal(size=(80, 3)).astype(np.float32)
        p[10:40] = p[0]
        idx, dist = map(np.asarray, search(jnp.asarray(p), k))
        assert not np.any(idx == np.arange(80)[:, None])
        assert np.sum(dist[0] == 0) == min(k, 30)
        _assert_indices_give_distances(p, None, idx, dist)

    def test_queries(self, search):
        """Bichromatic queries, k=1 and k=10."""
        rng = np.random.default_rng(7)
        p = rng.normal(size=(300, 3)).astype(np.float32)
        q = rng.normal(size=(40, 3)).astype(np.float32)
        for k in (1, 10):
            idx, dist = search(jnp.asarray(p), k, queries=jnp.asarray(q))
            np.testing.assert_allclose(
                np.asarray(dist), _ref(p, k, q), rtol=1e-5, atol=1e-6
            )
            _assert_indices_give_distances(p, q, np.asarray(idx), np.asarray(dist))

    def test_far_queries(self, search):
        """Queries farther from the data than its diameter still find real points."""
        p = np.random.default_rng(11).normal(size=(100, 3)).astype(np.float32)
        q = np.array([[50, 0, 0], [0, -40, 0]], np.float32)
        idx, dist = search(jnp.asarray(p), 3, queries=jnp.asarray(q))
        np.testing.assert_allclose(
            np.asarray(dist), _ref(p, 3, q), rtol=1e-5, atol=1e-6
        )
        assert not np.any(np.asarray(idx) == 100)

    @pytest.mark.parametrize("dtype", [jnp.int32, jnp.bfloat16], ids=["int32", "bf16"])
    def test_integer_and_low_precision_inputs(self, search, dtype):
        """Integers / bfloat16 behave as their float32 values (no inf rows)."""
        rng = np.random.default_rng(5)
        p = jnp.asarray(rng.integers(-50, 50, size=(120, 3))).astype(dtype)
        q = jnp.asarray(rng.integers(-60, 60, size=(15, 3))).astype(dtype)
        p32, q32 = p.astype(jnp.float32), q.astype(jnp.float32)
        for qq, qq32 in ((None, None), (q, q32)):
            got_i, got_d = search(p, 4, queries=qq)
            _, want_d = search(p32, 4, queries=qq32)
            assert jnp.issubdtype(got_d.dtype, jnp.floating)
            assert np.all(np.isfinite(np.asarray(got_d)))
            np.testing.assert_allclose(np.asarray(got_d), np.asarray(want_d), rtol=1e-5)
            assert np.all(np.asarray(got_i) < 120)

    def test_signature(self, search):
        """``points`` is positional-only; ``k`` may be bound by keyword (for jit)."""
        p = jnp.asarray(np.random.default_rng(1).normal(size=(20, 2)), jnp.float32)
        idx, _ = search(p, k=3)
        assert idx.shape == (20, 3)
        with pytest.raises(TypeError, match="positional"):
            search(points=p, k=3)

    def test_float64(self, search):
        """Under x64, float64 stays float64 and nothing scatters int64 into int32."""
        p = np.random.default_rng(9).normal(size=(200, 3))
        q = p[:20] * 3
        with enable_x64(new_val=True):
            for qq in (None, jnp.asarray(q)):
                _, dist = search(jnp.asarray(p), 4, queries=qq)
                assert dist.dtype == jnp.float64
                ref = _ref(p, 4, None if qq is None else q)
                np.testing.assert_allclose(np.asarray(dist), ref, rtol=1e-12)

    def test_empty_queries(self, search):
        """Zero queries give (0, k) outputs."""
        p = jnp.asarray(np.random.default_rng(0).normal(size=(30, 3)), jnp.float32)
        idx, dist = search(p, 4, queries=jnp.zeros((0, 3), jnp.float32))
        assert idx.shape == dist.shape == (0, 4)

    def test_empty_points(self, search):
        """Zero points: every query gets sentinel index 0 (= n) at distance inf."""
        q = jnp.asarray(np.random.default_rng(0).normal(size=(5, 3)), jnp.float32)
        idx, dist = search(jnp.zeros((0, 3), jnp.float32), 4, queries=q)
        assert idx.shape == dist.shape == (5, 4)
        assert np.all(np.asarray(idx) == 0)
        assert np.all(np.isinf(np.asarray(dist)))

    @pytest.mark.parametrize(
        ("args", "kwargs"),
        [
            ((np.ones((5, 2)), 0), {}),
            ((np.ones((5, 2)), -1), {}),
            ((np.ones((5, 2)), 2.0), {}),
            ((np.ones((5, 2)), True), {}),
            ((np.ones(5), 2), {}),
            ((np.ones((5, 2)), 2), {"queries": np.ones((3, 3))}),
        ],
        ids=["k0", "k-1", "kfloat", "kbool", "1d", "query-dim"],
    )
    def test_invalid_args_raise(self, search, args, kwargs):
        """Bad k or shapes raise ValueError.

        A float ``k`` can be a TypeError instead: under runtime type checking
        (as here) the ``k: int`` annotation rejects it first.
        """
        with pytest.raises((ValueError, TypeError), match=r"must|Type-check error"):
            search(*args, **kwargs)

    def test_non_finite_queries_raise(self, search):
        """NaN in queries is caught too, as ValueError when eager."""
        p = np.random.default_rng(0).normal(size=(50, 3)).astype(np.float32)
        q = p[:4].copy()
        q[2, 0] = np.nan
        with pytest.raises(ValueError, match="finite"):
            search(jnp.asarray(p), 3, queries=jnp.asarray(q))

    @pytest.mark.parametrize("scale", [1e19, 3e19, 1e30])
    def test_huge_coordinates_float32(self, search, scale):
        """Squared distances that overflow float32 still give real neighbours.

        1 kpc is ~3.1e19 m: unscaled, these d2 overflow to inf and came back as
        sentinels. Coordinates are scaled by a power of two (exact) first.
        """
        p = (np.random.default_rng(4).normal(size=(30, 3)) * scale).astype(np.float32)
        idx, dist = map(np.asarray, search(jnp.asarray(p), 2))
        assert np.all(idx < 30)
        np.testing.assert_allclose(dist, _ref(p, 2), rtol=1e-5)

    @pytest.mark.parametrize("scale", [1e-14, 1e-30])
    def test_tiny_coordinates_float32(self, search, scale):
        """Tiny coordinates keep their neighbours (no squared-distance underflow)."""
        p = (np.random.default_rng(6).normal(size=(30, 3)) * scale).astype(np.float32)
        idx, dist = map(np.asarray, search(jnp.asarray(p), 2))
        np.testing.assert_allclose(dist, _ref(p, 2), rtol=1e-5)
        _assert_indices_give_distances(p, None, idx, dist)

    def test_near_float32_max_coordinates(self, search):
        """Coordinates near float32's maximum do not overflow the scale factor."""
        p = np.array([[3e38], [1e38], [2e38]], np.float32)
        idx, dist = map(np.asarray, search(jnp.asarray(p), 1))
        np.testing.assert_array_equal(idx[:, 0], [2, 2, 1])
        assert np.all(np.isfinite(dist))

    def test_tight_cluster_beside_distant_outlier(self, search):
        """A 1e-15 cluster beside a 1e15 outlier keeps its own neighbours.

        Normalising every coordinate below 1 underflowed the cluster's squared
        distances to 0; in range, coordinates are left unscaled.
        """
        rng = np.random.default_rng(7)
        p = np.concat([rng.normal(size=(50, 2)) * 1e-15, [[1e15, 0.0]]])
        p = p.astype(np.float32)
        idx, dist = map(np.asarray, search(jnp.asarray(p), 3))
        np.testing.assert_allclose(dist, _ref(p, 3), rtol=1e-5)
        _assert_indices_give_distances(p, None, idx, dist)

    def test_huge_queries_beside_small_points(self, search):
        """Scaling covers the queries too: huge queries against unit points."""
        p = np.random.default_rng(0).normal(size=(50, 3)).astype(np.float32)
        q = np.array([[3e19, 0, 0], [0, -1e25, 0]], np.float32)
        idx, dist = map(np.asarray, search(jnp.asarray(p), 2, queries=jnp.asarray(q)))
        assert np.all(idx < 50)
        np.testing.assert_allclose(dist, _ref(p, 2, q), rtol=1e-5)

    def test_non_finite_with_empty_input_raises(self, search):
        """NaN is caught even when the other input is empty."""
        nan = jnp.full((2, 3), jnp.nan, jnp.float32)
        empty = jnp.zeros((0, 3), jnp.float32)
        for pts, qs in ((empty, nan), (nan, empty)):
            with pytest.raises(ValueError, match="finite"):
                search(pts, 2, queries=qs)

    def test_non_finite_raises(self, search):
        """NaN input is an error, not a silently wrong answer."""
        p = np.random.default_rng(0).normal(size=(50, 3)).astype(np.float32)
        p[3, 1] = np.nan
        with pytest.raises(ValueError, match="finite"):
            search(jnp.asarray(p), 5)

    @pytest.mark.parametrize(
        "kwargs",
        [{"leaf_size": 0}, {"frontier": 0}],
        ids=["leaf_size", "frontier"],
    )
    def test_bad_knn_sizes(self, kwargs):
        """Sizes below 1 raise ValueError."""
        with pytest.raises(ValueError, match=">= 1"):
            jaxknn.knn(jnp.ones((5, 2)), 2, **kwargs)

    def test_bad_chunk(self):
        """A chunk below 1 raises ValueError."""
        with pytest.raises(ValueError, match=">= 1"):
            jaxknn.brute_knn(jnp.ones((5, 2)), 2, chunk=0)


class TestTracing:
    """Both functions trace under jit, vmap and grad."""

    def test_jit_matches_eager(self, search):
        """Under jit the result equals the eager one."""
        p = jnp.asarray(np.random.default_rng(1).normal(size=(200, 3)), jnp.float32)
        idx, _ = jax.jit(lambda x: search(x, 5))(p)
        np.testing.assert_array_equal(np.asarray(idx), np.asarray(search(p, 5)[0]))

    @pytest.mark.parametrize("k", [3, 8])
    def test_ties_independent_of_jit_and_padding(self, k):
        """On a grid (many equidistant neighbours) eager, jit and brute agree."""
        g = np.stack(np.meshgrid(np.arange(7.0), np.arange(5.0)), -1).reshape(-1, 2)
        g = jnp.asarray(g, jnp.float32)
        eager = jaxknn.knn(g, k)[0]
        jitted = jax.jit(lambda x: jaxknn.knn(x, k))(g)[0]
        brute = jaxknn.brute_knn(g, k)[0]
        np.testing.assert_array_equal(np.asarray(eager), np.asarray(jitted))
        np.testing.assert_array_equal(np.asarray(eager), np.asarray(brute))

    def test_gradient_matches_brute_force(self):
        """Distance gradients equal brute force's (tie-free data)."""
        p = jnp.asarray(np.random.default_rng(6).normal(size=(64, 3)), jnp.float32)

        def grad(search):
            return jax.grad(lambda x: jnp.sum(search(x, 4)[1] ** 1.5))(p)

        np.testing.assert_allclose(
            np.asarray(grad(jaxknn.knn)),
            np.asarray(grad(jaxknn.brute_knn)),
            rtol=1e-5,
            atol=1e-6,
        )

    def test_non_finite_raises_under_jit(self, search):
        """Traced NaN input is a runtime error too, not a silently wrong answer."""
        p = np.random.default_rng(0).normal(size=(50, 3)).astype(np.float32)
        p[3, 1] = np.nan
        with pytest.raises(Exception, match="finite"):
            jax.block_until_ready(jax.jit(lambda x: search(x, 5))(p))

    @pytest.mark.parametrize("queries", [False, True])
    def test_vmap_matches_loop(self, search, queries):
        """Vmap over a batch of clouds equals separate eager calls."""
        rng = np.random.default_rng(8)
        pb = jnp.asarray(rng.normal(size=(3, 120, 3)), jnp.float32)
        qb = jnp.asarray(rng.normal(size=(3, 9, 3)), jnp.float32)
        if queries:
            got = jax.vmap(lambda a, b: search(a, 4, queries=b))(pb, qb)
        else:
            got = jax.vmap(lambda a: search(a, 4))(pb)
        for i in range(3):
            want = search(pb[i], 4, queries=qb[i] if queries else None)
            np.testing.assert_array_equal(np.asarray(got[0][i]), np.asarray(want[0]))
            np.testing.assert_allclose(
                np.asarray(got[1][i]), np.asarray(want[1]), rtol=1e-6
            )

    def test_gradient_finite_at_coincident_points(self, search):
        """Distances differentiate with neighbour selection fixed; no NaN at d=0."""
        p = np.random.default_rng(2).normal(size=(64, 3)).astype(np.float32)
        p[1] = p[0]

        def loss(x):
            return jnp.sum(search(x, 4)[1])

        g = np.asarray(jax.grad(loss)(jnp.asarray(p)))
        assert np.all(np.isfinite(g))
        assert np.any(g != 0)


class TestBucketing:
    """Eager size buckets for ``knn``."""

    def test_bucket_sizes(self):
        """Buckets are 2**j or 1.5 * 2**j, never smaller than n."""
        sizes = [jaxknn.bucket(n) for n in [1, 2, 3, 5, 7, 100, 1000]]
        assert sizes == [1, 2, 3, 6, 8, 128, 1024]

    def test_same_bucket_reuses_compilation(self, monkeypatch):
        """N in an already-compiled bucket does not recompile."""
        traces = []
        all_knn = _api._tree.all_knn

        def counting(*args, **kwargs):  # runs only while jit traces
            traces.append(args[0].shape)
            return all_knn(*args, **kwargs)

        monkeypatch.setattr(_api._tree, "all_knn", counting)
        # An unusual leaf_size, and a cleared cache, so the first call traces.
        _api._knn_core_jit.clear_cache()
        rng = np.random.default_rng(3)
        jaxknn.knn(
            jnp.asarray(rng.normal(size=(1000, 3)), jnp.float32), 10, leaf_size=13
        )
        jaxknn.knn(
            jnp.asarray(rng.normal(size=(990, 3)), jnp.float32), 10, leaf_size=13
        )
        assert traces == [(1024, 3)]

    def test_far_rows_never_win(self):
        """Padding rows are farther than the data's diameter from every point."""
        p = jnp.asarray(np.random.default_rng(4).normal(size=(100, 3)), jnp.float32)
        far = np.asarray(jaxknn.far_rows(p, 7))
        diam = np.max(
            np.linalg.norm(np.asarray(p)[:, None] - np.asarray(p)[None], axis=-1)
        )
        dmin = np.min(np.linalg.norm(np.asarray(p)[:, None] - far[None], axis=-1))
        assert dmin > diam

    @pytest.mark.parametrize("exponent", [-53, 0, 53])
    def test_far_rows_are_relative_to_the_data(self, exponent):
        """Scaling the data by 2**e scales the padding rows by exactly 2**e.

        An absolute ``+1`` offset swamped tiny data's range, and the coordinate
        scaling then underflowed it.
        """
        p = np.random.default_rng(5).normal(size=(100, 3)).astype(np.float32)
        s = np.float32(2.0**exponent)
        far = np.asarray(jaxknn.far_rows(jnp.asarray(p), 50))
        far_s = np.asarray(jaxknn.far_rows(jnp.asarray(p * s), 50))
        np.testing.assert_array_equal(far_s, far * s)

    def test_far_rows_are_distinct(self):
        """Identical padding rows sent every padding query into brute force."""
        p = jnp.asarray(np.random.default_rng(6).normal(size=(100, 3)), jnp.float32)
        far = np.asarray(jaxknn.far_rows(p, 200))
        assert len(np.unique(far, axis=0)) == 200

    def test_far_rows_beyond_the_data_at_large_magnitude(self):
        """|x| >> spread in float32: the far row still lies beyond the data."""
        p = np.c_[np.full(41, 1e9), np.linspace(0, 0.5, 41)].astype(np.float32)
        far = np.asarray(jaxknn.far_rows(jnp.asarray(p), 7), np.float64)
        gap = np.linalg.norm(p.astype(np.float64)[:, None] - far[None], axis=-1)
        assert gap.min() > 0.5  # farther than the data's diameter

    def test_far_rows_large_magnitude_float32(self):
        """|x| >> spread in float32: far rows must not round onto real points."""
        p = np.c_[np.full(41, 1e9), np.linspace(0, 0.5, 41)].astype(np.float32)
        idx, dist = jaxknn.knn(jnp.asarray(p), 3)
        assert np.all(np.asarray(idx) < 41)
        np.testing.assert_allclose(np.asarray(dist), _ref(p, 3), rtol=1e-5)

    def test_pow2_scale_is_exact_and_windowed(self):
        """In range no scaling; out of range a power of two, covering queries."""
        p = jnp.ones((4, 2), jnp.float32)
        assert float(jaxknn.pow2_scale(p)) == 1.0
        s = float(jaxknn.pow2_scale(p, p * 1e30))
        assert s > 1.0
        assert np.log2(s) == int(np.log2(s))
