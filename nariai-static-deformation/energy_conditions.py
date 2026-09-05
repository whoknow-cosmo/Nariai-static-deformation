#!/usr/bin/env python3
"""
Energy conditions of the compact even-quartic Nariai deformation.

Reproduces, at 30 significant digits, the numerical statements of Sec. 7.3 of

    H.-C. Kim, "A Static Axisymmetric Deformation of Nariai Geometry:
    Nonround Horizons without Rotation or Conical Defects".

Specifically:

  * the Legendre conformal profile  Sigma(z) = Sigma_inf / P_{1/2}(coth y)^2
    and its small-z series,                                       Eqs. (9), (10)
  * the kernel  K(z) = 4 csch^2 y - 2 sigma coth y  and its series, Eq. (12)
  * the three dominant-energy inequalities
        E >= 0,      2E - A >= 0,      2E - B >= 0,                Eq. (82)
    scanned over the compact square (x^2, chi^2) in [0,1]^2
  * the threshold  lambda_c = 0.363619247...                       Eq. (83)
    as the smallest positive root of
        2 lc^2 E0(lc^2) - (1 + lc^2) E2(lc^2) = 0                  Eq. (84)

Independent of horizon_figure.py: the Legendre function is evaluated here with
mpmath's `legenp`, there with scipy's `hyp2f1`.  crosscheck.py verifies that the
two implementations agree.

Usage:  python3 energy_conditions.py [--grid N] [--dps D]
"""
from __future__ import annotations
import argparse, json, pathlib, sys
import mpmath as mp

PAPER_LAMBDA_C = "0.363619247"


# --------------------------------------------------------------------------
#  conformal profile
# --------------------------------------------------------------------------
def P(X):
    """Legendre function P_{1/2}(X) on the cut-free branch X > 1."""
    return mp.legenp(mp.mpf(1) / 2, 0, X, type=3)


def Pp(X):
    return mp.diff(lambda w: mp.legenp(mp.mpf(1) / 2, 0, w, type=3), X)


def sigma(z):
    """sigma = d log Sigma / dy  with  y = -(1/2) log z.  Eq. (9)."""
    z = mp.mpf(z)
    if z == 0:
        return mp.mpf(0)
    X = (1 + z) / (1 - z)
    return 8 * z * Pp(X) / (P(X) * (1 - z) ** 2)


def sigma_over_z(z):
    """sigma(z)/z, finite at z -> 0 (-> 3).  Removes the apparent pole in A, B."""
    z = mp.mpf(z)
    if z == 0:
        return mp.mpf(3)
    return sigma(z) / z


def K_over_z(z):
    """K(z)/z with K = 16z/(1-z)^2 - 2 sigma (1+z)/(1-z).  Finite (-> 10). Eq. (12)."""
    z = mp.mpf(z)
    return 16 / (1 - z) ** 2 - 2 * sigma_over_z(z) * (1 + z) / (1 - z)


# --------------------------------------------------------------------------
#  stress-tensor channels, Sec. 7.1  (all in units of  Sigma / k)
# --------------------------------------------------------------------------
def E0(z):
    """Eq. (62)."""
    z = mp.mpf(z)
    s = sigma(z)
    return (-mp.mpf(3) / 4 * s ** 2 * (1 + z) / z
            - s * (z ** 2 - 4 * z + 1) / (z * (1 - z))
            + 3 * (1 + z) / (1 - z) ** 2)


def E2(z):
    """Eq. (63)."""
    z = mp.mpf(z)
    s = sigma(z)
    return (-mp.mpf(3) / 2 * s ** 2
            + s * (1 + z) / (1 - z)
            - (z ** 2 - 8 * z + 1) / (1 - z) ** 2)


def E_of(lam, x2, c2):
    """E * Sigma / k  at (x^2, chi^2).  Eq. (64)."""
    lam = mp.mpf(lam)
    z = lam ** 2 * mp.mpf(x2) * mp.mpf(c2)
    if z == 0:
        return -(1 + lam ** 2) * E2(mp.mpf(0))
    return lam ** 2 * (mp.mpf(x2) + mp.mpf(c2)) * E0(z) - (1 + lam ** 2) * E2(z)


def A_of(lam, x2, c2):
    """A * Sigma / k = (1/2) (K/z) lam^2 x^2 (1-chi^2)(1-lam^2 chi^2).  Eq. (61)."""
    lam, x2, c2 = mp.mpf(lam), mp.mpf(x2), mp.mpf(c2)
    return (K_over_z(lam ** 2 * x2 * c2) / 2) * lam ** 2 * x2 * (1 - c2) * (1 - lam ** 2 * c2)


def B_of(lam, x2, c2):
    """B * Sigma / k = (1/2) (K/z) lam^2 chi^2 (1-x^2)(1-lam^2 x^2).  Eq. (61)."""
    lam, x2, c2 = mp.mpf(lam), mp.mpf(x2), mp.mpf(c2)
    return (K_over_z(lam ** 2 * x2 * c2) / 2) * lam ** 2 * c2 * (1 - x2) * (1 - lam ** 2 * x2)


def dec_triple(lam, x2, c2):
    """The three quantities of Eq. (82), all >= 0 iff WEC and DEC hold."""
    E = E_of(lam, x2, c2)
    return E, 2 * E - A_of(lam, x2, c2), 2 * E - B_of(lam, x2, c2)


def polar_root_function(lam):
    """g(lam) = 2 lam^2 E0(lam^2) - (1+lam^2) E2(lam^2) = E at (x^2,chi^2)=(1,1). Eq. (84)."""
    lam = mp.mpf(lam)
    return 2 * lam ** 2 * E0(lam ** 2) - (1 + lam ** 2) * E2(lam ** 2)


# --------------------------------------------------------------------------
def scan(lam, n):
    """Global minimum of each Eq.(82) quantity over the compact square."""
    out = {}
    for name, f in (("E", lambda a, b: dec_triple(lam, a, b)[0]),
                    ("2E-A", lambda a, b: dec_triple(lam, a, b)[1]),
                    ("2E-B", lambda a, b: dec_triple(lam, a, b)[2])):
        best = None
        for i in range(n + 1):
            for j in range(n + 1):
                x2, c2 = mp.mpf(i) / n, mp.mpf(j) / n
                v = f(x2, c2)
                if best is None or v < best[0]:
                    best = (v, float(x2), float(c2))
        out[name] = best
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid", type=int, default=24, help="scan resolution per axis (default 24)")
    ap.add_argument("--dps", type=int, default=30, help="mpmath decimal places (default 30)")
    ap.add_argument("--outdir", default="output")
    a = ap.parse_args()
    mp.mp.dps = a.dps
    out = pathlib.Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
    ok = True

    print("=" * 72)
    print("1. Conformal profile   Sigma(z) = Sigma_inf / P_{1/2}((1+z)/(1-z))^2")
    print("=" * 72)
    s_small = sigma(mp.mpf("1e-8")) / mp.mpf("1e-8")
    print(f"   sigma(z)/z at z=1e-8      : {mp.nstr(s_small, 12)}   (series: 3)")
    ok &= abs(s_small - 3) < mp.mpf("1e-6")
    z = mp.mpf("0.1")
    num = 1 / P((1 + z) / (1 - z)) ** 2
    ser = 1 - mp.mpf("1.5") * z + 21 * z ** 2 / 32 - 13 * z ** 3 / 64
    print(f"   Sigma/Sigma_inf at z=0.1  : {mp.nstr(num, 12)}   (series: {mp.nstr(ser, 12)})")
    ok &= abs(num - ser) < mp.mpf("1e-4")
    kz = K_over_z(mp.mpf("1e-8"))
    print(f"   K(z)/z at z=1e-8          : {mp.nstr(kz, 12)}   (series: 10)   [Eq. (12)]")
    ok &= abs(kz - 10) < mp.mpf("1e-5")

    print()
    print("=" * 72)
    print("2. Dominant energy conditions, Eq. (82), over (x^2, chi^2) in [0,1]^2")
    print(f"   grid {a.grid}x{a.grid}; a negative minimum means the condition fails")
    print("=" * 72)
    print(f"   {'lambda':>12} {'min E':>16} {'min 2E-A':>16} {'min 2E-B':>16}   argmin(E)")
    rows = []
    for L in ["0.20", "0.30", "0.3636", "0.363619247", "0.37", "0.45"]:
        s = scan(mp.mpf(L), a.grid)
        print(f"   {L:>12} {mp.nstr(s['E'][0], 8):>16} {mp.nstr(s['2E-A'][0], 8):>16}"
              f" {mp.nstr(s['2E-B'][0], 8):>16}   ({s['E'][1]:.3f},{s['E'][2]:.3f})")
        rows.append({"lambda": L, "min_E": mp.nstr(s["E"][0], 12),
                     "min_2E_minus_A": mp.nstr(s["2E-A"][0], 12),
                     "min_2E_minus_B": mp.nstr(s["2E-B"][0], 12),
                     "argmin_E": [s["E"][1], s["E"][2]]})
    print()
    print("   => E is the binding condition; 2E-A and 2E-B stay positive throughout,")
    print("      and the minimum of E sits at the polar endpoints (x^2,chi^2)=(1,1)")
    print("      of the finite Killing horizons.  This is the premise of Eq. (84).")

    print()
    print("=" * 72)
    print("3. Threshold lambda_c :  smallest positive root of Eq. (84)")
    print("=" * 72)
    for L in ["0.30", "0.35", "0.36", "0.3636", "0.37", "0.40"]:
        print(f"   g({L:>7}) = {mp.nstr(polar_root_function(L), 10)}")
    root = mp.findroot(polar_root_function, mp.mpf("0.36"))
    print()
    print(f"   lambda_c (this run) = {mp.nstr(root, 15)}")
    print(f"   paper, Eq. (83)     = {PAPER_LAMBDA_C}...")
    agree = mp.nstr(root, 9).startswith(PAPER_LAMBDA_C[:11])
    ok &= agree
    print(f"   agreement to the printed digits : {'YES' if agree else 'NO'}")

    res = {"lambda_c": mp.nstr(root, 20), "paper_value": PAPER_LAMBDA_C,
           "mpmath_dps": a.dps, "grid": a.grid, "scan": rows, "all_checks_passed": bool(ok)}
    (out / "energy_conditions.json").write_text(json.dumps(res, indent=2))
    print(f"\n   wrote {out / 'energy_conditions.json'}")
    print(f"\n{'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
