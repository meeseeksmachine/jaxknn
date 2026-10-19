import json
import sys

import matplotlib.pyplot as plt

rows = json.load(open(sys.argv[1]))
TITLES = {"stream": "stream", "blob": "Gaussian blob", "interlopers": "stream + 1% interlopers"}
STYLE = {
    "jaxknn": dict(color="#2a6fdb", label="jaxknn (jit)"),
    "jaxkd": dict(color="#d9822b", label="jaxkd (jit)"),
    "scipy": dict(color="#7a7a7a", ls="--", label="SciPy cKDTree (eager, all cores)"),
}

fig, axes = plt.subplots(1, 4, figsize=(17, 4.2))
for ax in axes[1:3]:
    ax.sharey(axes[0])
for ax, name in zip(axes, TITLES):
    for lib, st in STYLE.items():
        pts = sorted((r["n"], r["steady_s"]) for r in rows if r["lib"] == lib and r["data"] == name)
        if pts:
            ax.loglog(*zip(*pts), marker="o", ms=4, **st)
    ax.set_title(TITLES[name])
    ax.set_xlabel("number of points n")
    ax.grid(True, which="major", alpha=0.3)
axes[0].set_ylabel("all-points kNN, k = 10 (s, steady state)")
axes[0].legend(frameon=False)
ax = axes[3]
for lib, st in STYLE.items():
    if lib == "scipy":
        continue
    pts = sorted((r["n"], r["first_s"]) for r in rows if r["lib"] == lib and r["data"] == "stream")
    ax.semilogx(*zip(*pts), marker="o", ms=4, **st)
ax.set_title("first call (compile + run), stream")
ax.set_xlabel("number of points n")
ax.set_ylabel("s")
ax.set_ylim(bottom=0)
ax.grid(True, which="major", alpha=0.3)
fig.suptitle("Exact all-points kNN on CPU (3-D, float32; Apple M2 Pro, 10 cores, load average ~4)", fontsize=11)
fig.tight_layout()
fig.savefig(sys.argv[2], dpi=150)
