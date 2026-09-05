"""Execute notebooks/analysis.ipynb top to bottom in one fresh namespace.

The benchmark env `py311` has IPython but not jupyter, nbconvert, nbformat or
ipykernel, so `jupyter nbconvert --to notebook --execute` cannot run there and
installing them into a shared lab env is not something this benchmark does on
its own. This script is the no-install route to the same check: it reads the
notebook's code cells and runs them in order in a single namespace, which is
faithful to a fresh-kernel top-to-bottom run because the notebook contains no
IPython magics and no shell escapes.

It stops at the first failing cell and prints that cell's source and traceback,
the way a kernel stops. Figures and tables are written by the notebook itself,
to `figures/` and `tables/`.

    python scripts/run_notebook.py
    python scripts/run_notebook.py --dry-run

If jupyter is available in some other env, the equivalent and equally valid
check is:

    cd notebooks && jupyter nbconvert --to notebook --execute analysis.ipynb
"""

import argparse
import json
import os
import sys
import time
import traceback

DEFAULT_NB = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "notebooks",
    "analysis.ipynb",
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--notebook", default=DEFAULT_NB, help="path to the .ipynb")
    ap.add_argument("--dry-run", action="store_true",
                    help="list the cells that would run and exit")
    args = ap.parse_args()

    nb_path = os.path.abspath(args.notebook)
    if not os.path.exists(nb_path):
        print(f"no notebook at {nb_path}", file=sys.stderr)
        return 2

    with open(nb_path) as fh:
        cells = [c for c in json.load(fh)["cells"] if c["cell_type"] == "code"]

    # Reject IPython-only syntax rather than failing obscurely part way through.
    for i, cell in enumerate(cells):
        for line in "".join(cell["source"]).split("\n"):
            if line.strip()[:1] in ("%", "!"):
                print(f"cell {i} uses IPython syntax and needs a real kernel:\n"
                      f"  {line.strip()}", file=sys.stderr)
                return 2

    if args.dry_run:
        print(f"would run {len(cells)} code cells from {nb_path}")
        print(f"would chdir to {os.path.dirname(nb_path)}")
        print("would write figures/ and tables/; reads results/*.csv and "
              "data/scaffold_manifest.csv")
        for i, cell in enumerate(cells):
            first = next((ln for ln in "".join(cell["source"]).split("\n")
                          if ln.strip()), "")
            print(f"  [{i:02d}] {first[:70]}")
        return 0

    os.chdir(os.path.dirname(nb_path))
    ns = {"__name__": "__main__", "__file__": nb_path}

    print(f"executing {len(cells)} code cells from {nb_path}", flush=True)
    started = time.time()
    for i, cell in enumerate(cells):
        src = "".join(cell["source"])
        if not src.strip():
            continue
        t0 = time.time()
        try:
            exec(compile(src, f"<cell {i}>", "exec"), ns)
        except Exception:
            print(f"[{i:02d}] FAILED after {time.time() - t0:.1f}s", flush=True)
            print("---- source ----", flush=True)
            print(src, flush=True)
            print("---- traceback ----", flush=True)
            traceback.print_exc()
            return 1
        print(f"[{i:02d}] ok {time.time() - t0:6.1f}s", flush=True)

    print(f"all {len(cells)} cells ok in {time.time() - started:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
