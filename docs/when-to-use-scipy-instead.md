# When to use SciPy instead

jaxknn makes the kNN *traceable*; on a CPU it doesn't make it faster. SciPy's `cKDTree` queries with every core, while jaxknn's query is only partly parallel on CPU.

All-points kNN (k=10, a 3-D stream with 2% interlopers): jaxknn compiled under `jit`, steady state, against SciPy called eagerly. It was measured on an Apple M2 Pro under load, so the ratios matter more than the absolute times.

| n | `jit` compile (first call) | jaxknn under `jit` | SciPy, eager | jaxknn / SciPy |
|---|---|---|---|---|
| 2,000 | 9.4 s | 17.8 ms | 6.0 ms | 2.9× |
| 10,000 | 11.6 s | 106 ms | 16.4 ms | 6.5× |
| 100,000 | 13.2 s | 1,261 ms | 148 ms | 8.5× |
| 1,000,000 | 26.8 s | 11,719 ms | 1,990 ms | 5.9× |

So:

- **Inside `jit`, `vmap` or `grad`, or on an accelerator:** use jaxknn. SciPy can't run on traced values.
- **Eager CPU code:** SciPy is faster and compiles nothing.

GPU wasn't measured.
