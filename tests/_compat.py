"""Shims for the oldest supported JAX."""

import jax

if hasattr(jax, "enable_x64"):  # jax >= 0.8
    enable_x64 = jax.enable_x64
else:  # pragma: no cover
    from jax.experimental import enable_x64  # noqa: F401
