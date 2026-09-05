#!/usr/bin/env python3
"""
Independent cross-check of the two Legendre implementations used in this
repository.

energy_conditions.py evaluates the conformal profile through mpmath's
`legenp` at 30 decimal places; horizon_figure.py evaluates it through scipy's
Gauss hypergeometric function, using

    P_{1/2}(w) = 2F1(-1/2, 3/2; 1; (1-w)/2),      w = (1+z)/(1-z),
    (1-w)/2 = -z/(1-z).

The two paths share no code.  This script checks that they agree, and that both
reproduce the small-z series quoted in the paper,

    Sigma/Sigma_inf = 1 - (3/2) z + (21/32) z^2 + O(z^3),           Eq. (10)
    sigma           = 3z + (15/8) z^2 + O(z^3),                     Eq. (10)
    K               = 10z + (65/4) z^2 + O(z^3).                    Eq. (12)

Usage:  python3 crosscheck.py [--tol 1e-12]
"""
from __future__ import annotations
import argparse, sys
import mpmath as mp
from scipy.special import hyp2f1

import energy_conditions as ec


def P_scipy(z: float) -> float:
    return float(hyp2f1(-0.5, 1.5, 1.0, -z / (1 - z)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tol", type=float, default=1e-12)
    a = ap.parse_args()
    mp.mp.dps = 30
    ok = True

    print("P_{1/2}((1+z)/(1-z)) :  mpmath legenp  vs  scipy hyp2f1")
    print(f"{'z':>10} {'mpmath':>22} {'scipy':>22} {'|rel diff|':>12}")
    for z in [1e-6, 1e-3, 0.01, 0.05, 0.1, 0.25, 0.4, 0.6, 0.8, 0.9, 0.99]:
        m = ec.P(mp.mpf(z) and (1 + mp.mpf(z)) / (1 - mp.mpf(z)))
        s = P_scipy(z)
        rel = abs(float(m) - s) / abs(s)
        flag = "" if rel < a.tol else "   <-- MISMATCH"
        ok &= rel < a.tol
        print(f"{z:>10.6g} {float(m):>22.15f} {s:>22.15f} {rel:>12.3e}{flag}")

    print("\nsmall-z series, Eqs. (10) and (12)")
    print(f"{'z':>10} {'Sigma/Sigma_inf':>18} {'series':>18} | "
          f"{'sigma':>14} {'series':>14} | {'K':>14} {'series':>14}")
    for z in [1e-4, 1e-3, 1e-2, 3e-2]:
        zz = mp.mpf(z)
        prof = 1 / ec.P((1 + zz) / (1 - zz)) ** 2
        prof_s = 1 - mp.mpf(3) / 2 * zz + mp.mpf(21) / 32 * zz ** 2
        sig = ec.sigma(zz)
        sig_s = 3 * zz + mp.mpf(15) / 8 * zz ** 2
        Kv = ec.K_over_z(zz) * zz
        K_s = 10 * zz + mp.mpf(65) / 4 * zz ** 2
        print(f"{z:>10.4g} {float(prof):>18.12f} {float(prof_s):>18.12f} | "
              f"{float(sig):>14.9f} {float(sig_s):>14.9f} | "
              f"{float(Kv):>14.9f} {float(K_s):>14.9f}")
        ok &= abs(prof - prof_s) < 5 * zz ** 3
        ok &= abs(sig - sig_s) < 50 * zz ** 3
        ok &= abs(Kv - K_s) < 500 * zz ** 3

    print(f"\n{'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
