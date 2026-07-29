"""Execute every code cell of research.ipynb in order, in one namespace.

Emulates what a Jupyter kernel does, without needing jupyter installed.
Reports per-cell timing so slow cells surface, and stops on the first failure
with a full traceback since later cells depend on earlier state.
"""

import json
import sys
import time
import traceback

import matplotlib

matplotlib.use("Agg")  # headless: plt.show() becomes a no-op

nb = json.load(open("research.ipynb"))
code_cells = [(i, c) for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code"]

ns: dict = {"__name__": "__main__"}
timings = []
failed = None

print(f"executing {len(code_cells)} code cells\n" + "=" * 72)

for n, (idx, cell) in enumerate(code_cells, 1):
    src = cell["source"]
    if isinstance(src, list):
        src = "".join(src)
    first = next((ln for ln in src.splitlines() if ln.strip()), "")
    label = first.strip()[:58]

    t0 = time.perf_counter()
    try:
        compiled = compile(src, f"<cell {n}>", "exec")
        exec(compiled, ns)
    except Exception:
        dt = time.perf_counter() - t0
        print(f"[{n:2d}/{len(code_cells)}] FAIL  {dt:6.2f}s  {label}")
        print("-" * 72)
        traceback.print_exc(file=sys.stdout)
        print("-" * 72)
        failed = n
        break
    dt = time.perf_counter() - t0
    timings.append((n, dt, label))
    flag = "  <-- SLOW" if dt > 20 else ""
    print(f"[{n:2d}/{len(code_cells)}] ok    {dt:6.2f}s  {label}{flag}", flush=True)

print("=" * 72)
if failed:
    print(f"FAILED at code cell {failed}")
    sys.exit(1)

total = sum(t for _, t, _ in timings)
print(f"ALL {len(code_cells)} CODE CELLS EXECUTED  (total {total:.1f}s)")
print("\nslowest cells:")
for n, dt, label in sorted(timings, key=lambda x: -x[1])[:5]:
    print(f"  cell {n:2d}  {dt:6.2f}s  {label}")

# Confirm the notebook actually produced the objects it claims to.
print("\nkey objects in the final namespace:")
for name in ["panel", "results", "table", "mean_ic", "t_ic", "sw", "wf",
             "plan", "hc", "full", "best_lb", "best_h"]:
    obj = ns.get(name)
    kind = type(obj).__name__ if obj is not None else "MISSING"
    extra = ""
    if hasattr(obj, "shape"):
        extra = f" shape={obj.shape}"
    elif isinstance(obj, dict):
        extra = f" keys={len(obj)}"
    print(f"  {name:10s} {kind}{extra}")
