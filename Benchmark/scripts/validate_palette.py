#!/usr/bin/env python3
"""Colourblind-safety check for the figure palette.

A Python port of the checks in the `dataviz` skill's `validate_palette.js`,
written because that script needs Node and Amarel has no node interpreter. The
constants, the Machado (2009) CVD transforms at severity 1.0, the OKLab
conversion and the thresholds are copied from it so the two agree:

    CVD_TARGET   8.0   OKLab dE x100, min(protan, deutan), adjacent pairs
    CVD_FLOOR    6.0   legal only with a secondary encoding
    NORMAL_FLOOR 15.0  worst pair, unsimulated; below this is a hard fail

The point is that palette safety is computed, not eyeballed. Run it whenever the
figure palette changes:

    python scripts/validate_palette.py
    python scripts/validate_palette.py --pairs all
"""

from __future__ import annotations

import argparse
import itertools
import math

CVD_TARGET, CVD_FLOOR, NORMAL_FLOOR = 8.0, 6.0, 15.0

MACHADO = {
    "protan": ((0.152286, 1.052583, -0.204868),
               (0.114503, 0.786281, 0.099216),
               (-0.003882, -0.048116, 1.051998)),
    "deutan": ((0.367322, 0.860646, -0.227968),
               (0.280085, 0.672501, 0.047413),
               (-0.011820, 0.042940, 0.968881)),
}


def hex_to_linear(h: str) -> tuple[float, float, float]:
    h = h.strip().lstrip("#")
    srgb = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    return tuple(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
                 for c in srgb)


def oklab(linear) -> tuple[float, float, float]:
    r, g, b = linear
    l = (0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b) ** (1 / 3)
    m = (0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b) ** (1 / 3)
    s = (0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b) ** (1 / 3)
    return (0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
            1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
            0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s)


def simulate(linear, kind: str):
    m = MACHADO[kind]
    return tuple(max(0.0, min(1.0, sum(m[i][j] * linear[j] for j in range(3))))
                 for i in range(3))


def delta_e(a, b) -> float:
    return 100 * math.dist(oklab(a), oklab(b))


def check(colors: list[str], pairs: str = "adjacent") -> int:
    linears = [hex_to_linear(c) for c in colors]
    idx = (list(zip(range(len(colors) - 1), range(1, len(colors))))
           if pairs == "adjacent"
           else list(itertools.combinations(range(len(colors)), 2)))

    worst_normal = (float("inf"), None)
    worst_cvd = (float("inf"), None)
    for i, j in idx:
        d_norm = delta_e(linears[i], linears[j])
        if d_norm < worst_normal[0]:
            worst_normal = (d_norm, (i, j))
        d_cvd = min(delta_e(simulate(linears[i], k), simulate(linears[j], k))
                    for k in MACHADO)
        if d_cvd < worst_cvd[0]:
            worst_cvd = (d_cvd, (i, j))

    print(f"palette ({len(colors)} slots, {pairs} pairs, {len(idx)} comparisons)")
    for n, c in enumerate(colors, 1):
        print(f"  slot {n}: {c}")
    ok = True
    d, pair = worst_normal
    verdict = "PASS" if d >= NORMAL_FLOOR else "FAIL"
    ok &= d >= NORMAL_FLOOR
    print(f"  worst normal-vision dE : {d:5.1f}  (slots {pair[0]+1},{pair[1]+1})"
          f"  floor {NORMAL_FLOOR}  {verdict}")
    d, pair = worst_cvd
    verdict = ("PASS" if d >= CVD_TARGET else
               "WARN, needs secondary encoding" if d >= CVD_FLOOR else "FAIL")
    ok &= d >= CVD_FLOOR
    print(f"  worst protan/deutan dE : {d:5.1f}  (slots {pair[0]+1},{pair[1]+1})"
          f"  target {CVD_TARGET}, floor {CVD_FLOOR}  {verdict}")
    print("  RESULT: " + ("usable" if ok else "NOT usable, re-step the palette"))
    return 0 if ok else 1


# The dataviz skill's documented categorical palette, first six slots in its
# documented order. Order matters: the certification is for that order.
DEFAULT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--colors", default=",".join(DEFAULT))
    parser.add_argument("--pairs", choices=["adjacent", "all"], default="adjacent")
    args = parser.parse_args()
    raise SystemExit(check([c.strip() for c in args.colors.split(",") if c.strip()],
                           args.pairs))


if __name__ == "__main__":
    main()
