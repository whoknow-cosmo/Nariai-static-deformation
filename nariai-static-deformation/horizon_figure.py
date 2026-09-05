#!/usr/bin/env python3
"""
Horizon geometry of the compact even-quartic Nariai deformation.

Builds the figure of Sec. 5.3 of

    H.-C. Kim, "A Static Axisymmetric Deformation of Nariai Geometry:
    Nonround Horizons without Rotation or Conical Defects".

The horizon cross-section metric is                                    Eq. (37)

    ds_H^2 = L^2 Q_lam(x) [ dx^2/(1-x^2) + (1-x^2) dphi^2 ],
    Q_lam(x) = Omega^2(lam^2 x^2) / (1 - lam^2 x^2),
    Omega^2(z) = 1 / P_{1/2}((1+z)/(1-z))^2,

with x = cos(theta) and lam = |a|/s_+ in [0,1).  Setting L = 1, the script

  (a) embeds the cross-section in Euclidean R^3 as a surface of revolution
      with cylindrical radius rho and axial height h -- note that h is the
      embedding height, NOT the similarity variable z = lam^2 x^2 chi^2 --
          rho(x)    = sqrt(Q_lam (1-x^2)),
          (dh/dx)^2 = [4Q^2 - (Q'(1-x^2) - 2xQ)^2] / [4Q(1-x^2)],
      and colours it by the Gaussian curvature K_H of Sec. 5.3;
  (b) overlays the meridional profiles;
  (c) plots L^2 K_H(x);
  (d) tracks h_pole/rho_eq, the area A/4piL^2 of Eq. (38), and the invariant
      horizon distortion  D_H = 1 - K_pole/K_eq  of Eq. (distortion),
      against lam, marking the energy-condition bound lam_c of Eq. (83).

A table of the panel-(d) quantities is written to output/horizon_table.csv.

The Legendre function is evaluated here through the hypergeometric
representation  P_{1/2}(w) = 2F1(-1/2, 3/2; 1; (1-w)/2)  in scipy; the
independent mpmath evaluation used by energy_conditions.py is compared against
it in crosscheck.py.

Usage:  python3 horizon_figure.py [--outdir DIR] [--dpi N] [--no-png]
"""
from __future__ import annotations
import argparse, csv, pathlib, sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.gridspec import GridSpec
from scipy.special import hyp2f1
from scipy.integrate import quad, cumulative_trapezoid

LAMBDA_C = 0.363619247          # Eq. (83); reproduced by energy_conditions.py

# ---------------------------------------------------------------- geometry --
Om2 = lambda z: 1.0 / hyp2f1(-0.5, 1.5, 1.0, -z / (1 - z)) ** 2      # Omega^2
Q = lambda x, l: Om2((l * x) ** 2) / (1 - (l * x) ** 2)              # Eq. (37), L=1


def Qp(x, l, h=1e-6):
    return (Q(x + h, l) - Q(x - h, l)) / (2 * h)


def dhdx2(x, l):
    """(dh/dx)^2 for the surface-of-revolution embedding, h = axial height.

    Strictly positive for every 0 <= lam < 1, so each cross-section admits a
    global isometric embedding in Euclidean R^3.
    """
    q, qp = Q(x, l), Qp(x, l)
    num = 4 * q ** 2 - (qp * (1 - x ** 2) - 2 * x * q) ** 2
    den = 4 * q * (1 - x ** 2)
    return np.where(np.abs(1 - x ** 2) < 1e-10, q, num / np.where(den == 0, 1e-300, den))


def KH(x, l, h=1e-5):
    """Gaussian curvature L^2 K_H(x) of the cross-section, Sec. 5.3."""
    g = lambda s: (1 - s ** 2) * 0.5 * Qp(s, l) / Q(s, l)
    return (1 - (g(x + h) - g(x - h)) / (2 * h)) / Q(x, l)


def meridian(l, n=1201):
    x = np.linspace(-1, 1, n)
    q = np.array([Q(xi, l) for xi in x])
    rho = np.sqrt(np.clip(q * (1 - x ** 2), 0, None))
    a = np.clip(np.array([dhdx2(xi, l) for xi in x]), 0, None)
    h = np.concatenate([[0.0], cumulative_trapezoid(np.sqrt(a), x)])
    h -= h[n // 2]
    return x, rho, h


area = lambda l: 2 * np.pi * quad(lambda s: Q(s, l), -1, 1, limit=200)[0]   # Eq. (38)

# ----------------------------------------------------------------- palette --
RAMP = ['#86b6ef', '#5598e7', '#2a78d6', '#1c5cab', '#104281']   # ordinal, one hue
CAT = ['#eb6834', '#1baf7a', '#4a3aa7']                          # categorical
INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#d8d7d2', '#fcfcfb'
DIV = LinearSegmentedColormap.from_list(
    'K', ['#0d366b', '#2a78d6', '#9ec5f4', '#f0efec', '#f2a5a4', '#e34948', '#8e2b2a'])
LAM = [0.0, LAMBDA_C, 0.6, 0.8, 0.95]
LBL = [r'$\lambda=0$', r'$\lambda=\lambda_c$', r'$\lambda=0.6$',
       r'$\lambda=0.8$', r'$\lambda=0.95$']
LAM3D = [0.0, LAMBDA_C, 0.7, 0.95]


def build(outdir: pathlib.Path, dpi: int, png: bool) -> None:
    plt.rcParams.update({
        'font.family': 'serif', 'font.serif': ['DejaVu Serif'], 'mathtext.fontset': 'cm',
        'font.size': 9, 'axes.edgecolor': INK2, 'axes.labelcolor': INK,
        'xtick.color': INK2, 'ytick.color': INK2, 'axes.linewidth': .7,
        'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF})

    fig = plt.figure(figsize=(9.6, 6.4))
    gs = GridSpec(2, 4, figure=fig, height_ratios=[.80, 1.0], hspace=.46, wspace=.40,
                  left=.065, right=.975, top=.955, bottom=.095)

    # (a) embeddings ---------------------------------------------------------
    norm = TwoSlopeNorm(vmin=0.2, vcenter=1.0, vmax=1.8)
    for j, l in enumerate(LAM3D):
        ax = fig.add_subplot(gs[0, j], projection='3d')
        x, rho, h = meridian(l, 241)
        ph = np.linspace(0, 2 * np.pi, 121)
        R, PH = np.meshgrid(rho, ph)
        H, _ = np.meshgrid(h, ph)
        K = np.array([KH(xi, l) for xi in np.clip(x, -.9999, .9999)])
        ax.plot_surface(R * np.cos(PH), R * np.sin(PH), H,
                        facecolors=DIV(norm(np.meshgrid(K, ph)[0])),
                        rstride=2, cstride=2, linewidth=0, antialiased=True, shade=False)
        ax.set_box_aspect((1, 1, 1)); ax.view_init(elev=16, azim=-60)
        for A_ in (ax.xaxis, ax.yaxis, ax.zaxis):
            A_.set_pane_color((1, 1, 1, 0)); A_.line.set_color((1, 1, 1, 0))
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([]); ax.grid(False)
        ax.set_xlim(-1.02, 1.02); ax.set_ylim(-1.02, 1.02); ax.set_zlim(-1.02, 1.02)
        p0 = ax.get_position(); xc = p0.x0 + p0.width / 2
        p = p0.expanded(1.52, 1.52)
        ax.set_position([p.x0, p.y0 - 0.040, p.width, p.height])
        tag = r'$\lambda=\lambda_c=0.364$' if j == 1 else rf'$\lambda={l:g}$'
        fig.text(xc, 0.955, tag, fontsize=10, color=INK, ha='center', va='top')
    cax = fig.add_axes([.315, .585, .37, .014])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=DIV), cax=cax,
                      orientation='horizontal', ticks=[0.2, 0.6, 1.0, 1.4, 1.8])
    cb.set_label(r'Gaussian curvature  $L^2K_H$   (grey $=1$: round sphere)',
                 fontsize=8.5, color=INK2, labelpad=2)
    cb.ax.tick_params(labelsize=7.5, color=INK2, labelcolor=INK2, length=2)
    cb.outline.set_visible(False)
    fig.text(.012, .985, '(a)', fontsize=10, color=INK, weight='bold')

    # (b) meridional profiles ------------------------------------------------
    ax = fig.add_subplot(gs[1, 0:2])
    for l, c in zip(LAM, RAMP):
        x, rho, h = meridian(l)
        ls = (0, (4, 2.5)) if l == 0 else '-'
        ax.plot(np.concatenate([rho, -rho[::-1]]), np.concatenate([h, h[::-1]]),
                color=c, lw=2.0, ls=ls, zorder=3 + LAM.index(l))
        ax.plot([0], [h[-1]], marker='o', ms=3.4, color=c, zorder=9)
    ax.axhline(0, color=GRID, lw=.7, zorder=1); ax.axvline(0, color=GRID, lw=.7, zorder=1)
    ax.set_aspect('equal'); ax.set_xlim(-1.62, 1.62); ax.set_ylim(-1.15, 1.15)
    ax.set_xlabel(r'$\rho/L$   (equatorial)'); ax.set_ylabel(r'$h/L$   (axis)')
    ax.spines[['top', 'right']].set_visible(False)
    ax.annotate('equator fixed:\n' + r'$\rho_{\rm eq}=L$  $\forall\lambda$', xy=(1.005, 0.0),
                xytext=(1.42, 0.30), fontsize=8, color=INK2, ha='center', va='center',
                arrowprops=dict(arrowstyle='-', lw=.7, color=INK2, shrinkA=2, shrinkB=1))
    ax.annotate('poles flatten\n' + r'as $\lambda\to1$', xy=(-0.30, 0.90), xytext=(-1.36, 0.34),
                fontsize=8, color=INK2, ha='center', va='center',
                arrowprops=dict(arrowstyle='-', lw=.7, color=INK2, shrinkA=2, shrinkB=1))
    for l, c, lab in zip(LAM, RAMP, LBL):
        ax.plot([], [], color=c, lw=2, ls=(0, (4, 2.5)) if l == 0 else '-', label=lab)
    ax.legend(loc='lower left', fontsize=7.8, frameon=False, labelcolor=INK,
              handlelength=1.7, borderpad=.1, labelspacing=.30, bbox_to_anchor=(-0.02, -0.02))
    fig.text(.012, .505, '(b)', fontsize=10, color=INK, weight='bold')

    # (c) Gaussian curvature -------------------------------------------------
    ax = fig.add_subplot(gs[1, 2])
    xs = np.linspace(-.9995, .9995, 401)
    ax.axhline(1, color=GRID, lw=1.0, zorder=1)
    ax.text(-0.985, 1.035, r'$\lambda=0$', fontsize=7.6, color=INK2, ha='left', va='bottom')
    for l, c in zip(LAM, RAMP):
        ax.plot(xs, [KH(xi, l) for xi in xs], color=c, lw=2.0,
                ls=(0, (4, 2.5)) if l == 0 else '-', zorder=3)
    ax.set_xlim(-1, 1); ax.set_xticks([-1, -.5, 0, .5, 1]); ax.set_ylim(0.0, 1.68)
    ax.set_xlabel(r'$x=\cos\theta$'); ax.set_ylabel(r'$L^2K_H(x)$')
    ax.spines[['top', 'right']].set_visible(False)
    ax.text(0.0, 1.60, 'equator', fontsize=7.8, color=INK2, ha='center')
    ax.text(0.0, 0.06, 'poles', fontsize=7.8, color=INK2, ha='center')
    fig.text(.512, .505, '(c)', fontsize=10, color=INK, weight='bold')

    # (d) scalars vs lambda --------------------------------------------------
    ax = fig.add_subplot(gs[1, 3])
    ls_ = np.linspace(0, 0.985, 60)
    ob = [meridian(l, 401)[2][-1] for l in ls_]                 # h_pole / rho_eq
    ar = [area(l) / (4 * np.pi) for l in ls_]                    # A / 4 pi L^2
    dh = [1.0 - KH(.99999, l) / KH(0., l) for l in ls_]          # D_H, Eq. (distortion)
    for y, c in [(ob, CAT[0]), (ar, CAT[1]), (dh, CAT[2])]:
        ax.plot(ls_, y, color=c, lw=2.0, zorder=3)
    ax.axvline(LAMBDA_C, color=INK2, lw=.8, ls=(0, (2, 2)), zorder=2)
    ax.text(LAMBDA_C, 1.105, r'$\lambda_c$ (WEC/DEC)', fontsize=8,
            color=INK2, ha='center')
    ax.set_xlim(0, 1.0); ax.set_ylim(0, 1.17); ax.set_xlabel(r'$\lambda=|a|/s_+$')
    ax.set_xticks([0, .25, .5, .75, 1.0]); ax.spines[['top', 'right']].set_visible(False)
    ax.legend([plt.Line2D([], [], color=c, lw=2) for c in CAT],
              [r'$h_{\rm pole}/\rho_{\rm eq}$', r'$A/4\pi L^2$', r'$\mathcal{D}_H$'],
              loc='center left', fontsize=7.8, labelcolor=INK,
              frameon=True, facecolor=SURF, edgecolor='none', framealpha=1.0,
              handlelength=1.5, borderpad=.25, labelspacing=.34,
              bbox_to_anchor=(-.02, .58))
    fig.text(.762, .505, '(d)', fontsize=10, color=INK, weight='bold')

    fig.savefig(outdir / 'horizon_shape.pdf')
    if png:
        fig.savefig(outdir / 'horizon_shape.png', dpi=dpi)

    with open(outdir / 'horizon_table.csv', 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['lambda', 'h_pole_over_rho_eq', 'area_over_4piL2',
                    'K_equator', 'K_pole', 'D_H', 'min_dhdx_squared'])
        for l in [0.0, 0.1, 0.2, 0.3, LAMBDA_C, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]:
            xg = np.linspace(-0.999999, 0.999999, 4001)
            keq, kpol = KH(0.0, l), KH(0.999999, l)
            w.writerow([f'{l:.9f}', f'{meridian(l, 2001)[2][-1]:.8f}',
                        f'{area(l) / (4 * np.pi):.8f}', f'{keq:.8f}', f'{kpol:.8f}',
                        f'{1.0 - kpol / keq:.8f}',
                        f'{np.array([dhdx2(xi, l) for xi in xg]).min():.8f}'])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default="output")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--no-png", action="store_true")
    a = ap.parse_args()
    out = pathlib.Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
    build(out, a.dpi, not a.no_png)
    print(f"wrote {out/'horizon_shape.pdf'}")
    if not a.no_png:
        print(f"wrote {out/'horizon_shape.png'}")
    print(f"wrote {out/'horizon_table.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
