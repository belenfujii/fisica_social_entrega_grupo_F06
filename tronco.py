# =====================================================================================
# tronco.py — el modelo de Boltzmann con "tronco común" elegido por lasso
#
# Pesos de cada vertical: β_v = μ + δ_v. Se minimiza
#     J = (1/G) Σ_v NLL_v(μ + δ_v) + λ ( Σ_k |μ_k| + r Σ_v Σ_k |δ_v,k| )
# (data shared lasso; Gross y Tibshirani 2016). Para rankear se usa SOLO el tronco μ:
# en 2024S1 rinde igual que μ + δ_v y sirve para cualquier vertical, también una nueva.
# λ = FRAC · λ_max y r se eligieron con las dos ventanas de validación (exp_lasso.py).
# El L1 se resuelve de forma exacta con L-BFGS-B separando θ = θ⁺ − θ⁻ (θ± ≥ 0).
#
# Se usa igual que boltzmann_seleccion.Boltzmann:  Tronco(variables).fit({V: [pool, ...]})
# y después .predict(X) y .tabla_beta().
# =====================================================================================
import numpy as np
import pandas as pd
from scipy.optimize import minimize

import boltzmann_seleccion as bs

FRAC, R = 0.002, 1.0      # elegidos en validación (exp_lasso.py)
EPS, TOL0 = 1e-4, 1e-6     # ridge mínimo para unicidad; |β| menor que TOL0 se considera cero


def _nll_grad(X, ini, grupo, y, beta):
    eta = (X @ beta.astype(np.float32)).astype(np.float64)
    mx = np.maximum.reduceat(eta, ini)
    ex = np.exp(eta - mx[grupo])
    Z = np.add.reduceat(ex, ini)
    pr = ex / Z[grupo]
    return -(y @ eta - (np.log(Z) + mx).sum()), (X.T @ (pr - y).astype(np.float32)).astype(np.float64)


class Tronco(bs.Boltzmann):
    def __init__(self, variables, frac=FRAC, r=R, negativos=200, semilla=0):
        super().__init__(variables, l2=0.0, negativos=negativos, semilla=semilla)
        self.frac, self.r = frac, r

    # ---------- datos: ganadores + muestra de negativos (McFadden), por vertical ----------
    def _armar(self, pools_por_vertical):
        rng = np.random.default_rng(self.semilla)
        bloques = []
        for pools in pools_por_vertical.values():
            partes = []
            for k, p in enumerate(pools):
                p = p[p.groupby("q").ganador.transform("max").values > 0]
                o = pd.Series(rng.random(len(p))).groupby(p.q.values).rank(method="first").values
                p = p[(p.ganador.values == 1) | (o <= self.negativos)]
                partes.append((p.q.values + 1_000_000 * k, p.ganador.values,
                               self._transformar(p[self.variables].values).astype(np.float32)))
            g = np.concatenate([a for a, _, _ in partes])
            orden = np.argsort(g, kind="stable")
            g = g[orden]
            y = np.concatenate([b for _, b, _ in partes])[orden].astype(np.float64)
            X = np.ascontiguousarray(np.concatenate([c for _, _, c in partes])[orden])
            ini = np.flatnonzero(np.r_[True, g[1:] != g[:-1]])
            grupo = np.repeat(np.arange(len(ini)), np.diff(np.r_[ini, len(g)]))
            y = y / np.bincount(grupo, weights=y)[grupo]
            bloques.append((X, ini, grupo, y))
        return bloques

    def _ajustar(self, bloques, lam):
        P, nv = len(self.variables), len(bloques)
        G = sum(len(b[1]) for b in bloques)
        n = P * (nv + 1)
        w = np.r_[np.ones(P), self.r * np.ones(P * nv)]
        w[self.variables.index("p_b0")] = 0.0            # el B0 no se penaliza en el tronco

        def f(z):
            th = z[:n] - z[n:]
            mu = th[:P]
            tot, g = 0.0, np.zeros(n)
            for i, (X, ini, grupo, y) in enumerate(bloques):
                val, gr = _nll_grad(X, ini, grupo, y, mu + th[P * (i + 1):P * (i + 2)])
                tot += val
                g[:P] += gr
                g[P * (i + 1):P * (i + 2)] += gr
            tot, g = tot / G, g / G
            tot += 0.5 * EPS * th @ th + lam * w @ (z[:n] + z[n:])
            g += EPS * th
            return tot, np.r_[g + lam * w, -g + lam * w]

        res = minimize(f, np.zeros(2 * n), jac=True, method="L-BFGS-B", bounds=[(0, None)] * (2 * n),
                       options={"maxiter": 3000, "maxfun": 6000})
        th = res.x[:n] - res.x[n:]
        th[np.abs(th) < TOL0] = 0.0
        return th[:P], th[P:].reshape(nv, P)

    def fit(self, pools_por_vertical):
        """pools_por_vertical: {vertical: [pool de cada semestre de entrenamiento]}."""
        rng = np.random.default_rng(0)
        muestra = [p[self.variables].values[rng.choice(len(p), min(len(p), 150_000), replace=False)]
                   for pools in pools_por_vertical.values() for p in pools]
        self._ajustar_escalas(np.concatenate(muestra))   # escalas sólo con entrenamiento
        bloques = self._armar(pools_por_vertical)
        # λ_max: con penalización enorme sólo queda p_b0; el mayor |gradiente| del tronco ahí
        mu0, _ = self._ajustar(bloques, 10.0)
        g = sum(_nll_grad(X, ini, grupo, y, mu0)[1] for X, ini, grupo, y in bloques) / sum(len(b[1]) for b in bloques)
        g[self.variables.index("p_b0")] = 0.0
        self.lam = self.frac * np.abs(g).max()
        self.beta, self.delta = self._ajustar(bloques, self.lam)
        self.convergio = True
        return self
