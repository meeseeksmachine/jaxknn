"""All-points kNN scaling: jaxknn vs jaxkd (both under jit) vs SciPy cKDTree.

Writes results.json incrementally (one row per measurement).
"""

import json
import sys
import time

import jax
import jax.numpy as jnp
import jaxkd
import jaxknn
import numpy as np
from scipy.spatial import cKDTree

K = 10
SIZES = [1_000, 3_000, 10_000, 30_000, 100_000, 300_000, 1_000_000]
NAMES = ["stream", "blob", "interlopers"]
GIVE_UP = 60.0  # s; skip larger n for a (lib, data) once one steady call exceeds this
OUT = sys.argv[1]


def data(name, n):  # same generator as phasecurvefit's old tests/benchmarks/test_kdtree.py
    rng = np.random.default_rng(0)
    if name == "blob":
        return rng.normal(size=(n, 3)).astype(np.float32)
    t = np.linspace(0, 1, n)
    p = np.c_[10 * t, np.sin(3 * t), 0.3 * np.cos(2 * t)] + rng.normal(0, 0.02, (n, 3))
    if name == "interlopers":
        i = rng.choice(n, n // 100, replace=False)
        pad = np.array([0.0, 3.0, 3.0])
        p[i] = rng.uniform(p.min(0) - pad, p.max(0) + pad, (len(i), 3))
    return p[rng.permutation(n)].astype(np.float32)


knn_jit = jax.jit(lambda x: jaxknn.knn(x, K))
kd_jit = jax.jit(lambda x: jaxkd.query_neighbors(jaxkd.build_tree(x), x, k=K + 1))


def scipy_knn(x):
    return cKDTree(x).query(x, k=K + 1, workers=-1)


def timed(f, x, reps):
    t0 = time.perf_counter()
    jax.block_until_ready(f(x))
    first = time.perf_counter() - t0
    best = first
    for _ in range(reps if first < 20 else 1):
        t0 = time.perf_counter()
        jax.block_until_ready(f(x))
        best = min(best, time.perf_counter() - t0)
    return first, best


libs = {"jaxknn": knn_jit, "jaxkd": kd_jit, "scipy": scipy_knn}
dead = set()
rows = []
for n in SIZES:
    for name in NAMES:
        x_np = data(name, n)
        x = jnp.asarray(x_np)
        ref = np.sort(scipy_knn(x_np)[0][:, 1:], axis=1)
        for lib, f in libs.items():
            if (lib, name) in dead:
                continue
            arg = x_np if lib == "scipy" else x
            reps = 3 if n <= 100_000 else 1
            first, best = timed(f, arg, reps)
            # sanity: neighbour distances match SciPy's
            if lib == "jaxknn":
                d = np.sort(np.asarray(f(x)[1]), axis=1)
                assert np.allclose(d, ref, rtol=1e-4, atol=1e-6), (lib, name, n)
            row = dict(lib=lib, data=name, n=n, first_s=first, steady_s=best)
            rows.append(row)
            print(row, flush=True)
            if best > GIVE_UP:
                dead.add((lib, name))
            with open(OUT, "w") as fh:
                json.dump(rows, fh, indent=1)
