# Numerical companion — A Static Axisymmetric Deformation of Nariai Geometry

Scripts reproducing the numerical results of

> H.-C. Kim, *A Static Axisymmetric Deformation of Nariai Geometry: Nonround Horizons without Rotation or Conical Defects* (2026).
> <!-- TODO: arXiv:XXXX.XXXXX -->

The paper constructs an exact one-parameter family of static axisymmetric deformations of the Nariai geometry $dS_2\times S^2$, labelled by $\lambda=|a|/s_+\in[0,1)$. Two of its statements are numerical, and this repository reproduces both from scratch.

| Script | Reproduces | Runtime |
|---|---|---|
| `energy_conditions.py` | the weak/dominant energy-condition threshold $\lambda_c = 0.363619247\ldots$, Sec. 7.3, Eqs. (82)–(84) | ~2 s |
| `horizon_figure.py` | the horizon-geometry figure and table, Sec. 5.3, Eqs. (37)–(38) | ~40 s |
| `crosscheck.py` | agreement of the two independent Legendre-function implementations | ~1 s |

## Quick start

```bash
git clone https://github.com/USER/nariai-static-deformation   # TODO: your URL
cd nariai-static-deformation
pip install -r requirements.txt
make all          # or run the three scripts individually
```

Outputs are written to `output/`:

| File | Contents |
|---|---|
| `horizon_shape.pdf` / `.png` | the Sec. 5.3 figure |
| `horizon_table.csv` | $h_{\rm pole}/\rho_{\rm eq}$, $A/4\pi L^2$, $K_{\rm eq}$, $K_{\rm pole}$, $\mathcal{D}_H$, $\min(dh/dx)^2$ vs $\lambda$ |
| `energy_conditions.json` | $\lambda_c$ to 20 digits and the energy-condition scan |

## What each script establishes

### `energy_conditions.py` — the threshold $\lambda_c$

Evaluates the Legendre conformal profile $\Sigma(z)=\Sigma_\infty/P_{1/2}\!\big(\tfrac{1+z}{1-z}\big)^2$ with `mpmath` at 30 decimal places, then

1. verifies the small-$z$ series of Eqs. (10) and (12) — $\sigma\to3z$, $\Sigma/\Sigma_\infty\to1-\tfrac32 z$, $\mathcal K\to10z$;
2. scans **all three** dominant-energy inequalities of Eq. (82),
   $\mathcal E\ge0$, $2\mathcal E-\mathfrak A\ge0$, $2\mathcal E-\mathfrak B\ge0$,
   over the compact square $(x^2,\chi^2)\in[0,1]^2$, confirming that $\mathcal E$ is the binding condition and that its minimum sits at the polar endpoints $(x^2,\chi^2)=(1,1)$ of the finite Killing horizons — the premise on which Eq. (84) rests;
3. solves Eq. (84), $2\lambda_c^2\mathcal E_0(\lambda_c^2)-(1+\lambda_c^2)\mathcal E_2(\lambda_c^2)=0$.

Expected output (abridged):

```
   0.363619247    9.0720062e-10     1.8144012e-9     1.8144012e-9   (1.000,1.000)
   0.37          -0.033332166      -0.066664331     -0.066664331   (1.000,1.000)

   lambda_c (this run) = 0.363619247174825
   paper, Eq. (83)     = 0.363619247...
```

### `horizon_figure.py` — the horizon geometry

The horizon cross-section metric is Eq. (37),

$$ds_H^2 = L^2 Q_\lambda(x)\Big[\frac{dx^2}{1-x^2}+(1-x^2)\,d\phi^2\Big],\qquad Q_\lambda(x)=\frac{\Omega^2(\lambda^2x^2)}{1-\lambda^2x^2},$$

with $x=\cos\theta$. The script embeds it in Euclidean $\mathbb E^3$ as a surface of revolution with cylindrical radius $\rho$ and axial height $h$ (not to be confused with the similarity variable $z=\lambda^2x^2\chi^2$),

$$\rho(x)=\sqrt{Q_\lambda(1-x^2)},\qquad \Big(\frac{dh}{dx}\Big)^{2}=\frac{4Q^2-\big(Q'(1-x^2)-2xQ\big)^2}{4Q(1-x^2)},$$

colours it by the Gaussian curvature $K_H$, and tracks the shape against $\lambda$.

Three facts the table makes explicit:

* the equatorial radius is **fixed** at $\rho_{\rm eq}=L$ for every $\lambda$, because $Q_\lambda(0)=1$; the deformation is a pure polar flattening;
* the horizon is **oblate** — $K_{\rm eq}>K_{\rm pole}$, with $h_{\rm pole}/\rho_{\rm eq}$ falling from $1$ to $0.738$ as $\lambda\to1$ — without any rotation, the invariant distortion $\mathcal{D}_H=1-K_{\rm pole}/K_{\rm eq}$ rising from $0$ to $0.96$;
* $(dh/dx)^2>0$ throughout for every $0\le\lambda<1$, so each cross-section is strictly convex and admits a **global** isometric embedding in $\mathbb E^3$. (Contrast the Kerr horizon, whose polar Gauss curvature turns negative at $a/M>\sqrt3/2$, obstructing the Euclidean embedding — L. Smarr, *Phys. Rev. D* **7**, 289 (1973).)

### `crosscheck.py` — independent implementations

The two analysis scripts share **no code**. `energy_conditions.py` evaluates $P_{1/2}$ with `mpmath.legenp` at 30 dps; `horizon_figure.py` uses the hypergeometric representation $P_{1/2}(w)={}_2F_1(-\tfrac12,\tfrac32;1;\tfrac{1-w}{2})$ in `scipy`. `crosscheck.py` confirms they agree to machine precision (relative difference $\lesssim10^{-15}$ across $z\in(0,1)$) and that both match the paper's series.

## Requirements

Python ≥ 3.10 with `numpy`, `scipy`, `matplotlib`, `mpmath` (see `requirements.txt`). No compiled extensions, no data files, no network access. A GitHub Actions workflow (`.github/workflows/ci.yml`) runs all three scripts on Python 3.10 and 3.12 and uploads `output/` as a build artifact.

## Citing

Please cite both the paper and this software.

<!-- TODO: replace with the DOI Zenodo mints for the first release -->
```bibtex
@software{Kim_nariai_numerics_2026,
  author  = {Kim, Hyeong-Chan},
  title   = {Numerical companion to: A Static Axisymmetric Deformation of Nariai Geometry},
  year    = {2026},
  version = {1.0.0},
  doi     = {10.5281/zenodo.XXXXXXX},
  url     = {https://github.com/USER/nariai-static-deformation}
}
```

`CITATION.cff` carries the same metadata in machine-readable form; GitHub renders it as a "Cite this repository" button.

## License

MIT — see [`LICENSE`](LICENSE).

## Acknowledgment

Supported by the National Research Foundation of Korea (NRF), grant RS-2026-25483539.
