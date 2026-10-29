import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # módulos de la raíz
# =====================================================================================
# exp_bayes.py
#
# MODELO DE BOLTZMANN JERÁRQUICO (bayesiano): "cuánto agrega saber la vertical", con números.
#
# Cada vertical v tiene sus pesos β_v, pero no son libres: vienen de una distribución común,
#       β_v = μ + δ_v ,        δ_v ~ Normal(0, σ² I)
# (modelo jerárquico o de "agrupamiento parcial"; Gelman et al., Bayesian Data Analysis, cap. 5;
#  en elección discreta, Allenby y Rossi 1998). μ es lo que todas las verticales tienen en común
# y δ_v lo propio de cada una. σ es la medida buscada: si σ → 0 todas comparten los mismos pesos
# (modelo global); si σ → ∞ cada vertical va por su lado (modelo por vertical).
#
#   - σ se elige con las dos ventanas de validación (bayes empírico por validación).
#   - Se reportan intervalos de credibilidad del 95% para μ y para δ_v con la aproximación
#     de Laplace (MacKay 1992): la posterior se aproxima por una gaussiana centrada en el
#     máximo, con covarianza = inversa del hessiano de -log posterior.
#   - Una vertical nueva usa β = μ (lo único que se sabe de ella es lo común).
#
# Necesita los pools de exp_universal.py --preparar (carpeta pools_universal/).
# Uso:  uv run experimentos/exp_bayes.py
# =====================================================================================
import numpy as np
import pandas as pd
from scipy.optimize import minimize

import boltzmann_seleccion as bs
import modelo_v2 as m
from exp_universal import SEIS, TE, TR, VENTANAS, VERTICALES, cargar, Semestre  # Semestre: para unpickle

log = m.log
SIGMAS = [0.03, 0.1, 0.3, 1.0, 3.0]          # desvío entre verticales de los β (en unidades de β)
L2_MU = 0.001                                 # prior muy débil sobre lo común


class Datos:
    """Pools de entrenamiento de varias verticales, ya transformados (asinh + estandarización global)."""
    def __init__(self, S, claves_por_v, pre, negativos=200, semilla=0):
        rng = np.random.default_rng(semilla)
        self.v = []
        for V, claves in claves_por_v.items():
            partes = []
            for k, cl in enumerate(claves):
                p = S[cl].pool
                p = p[p.groupby("q").ganador.transform("max").values > 0]
                o = pd.Series(rng.random(len(p))).groupby(p.q.values).rank(method="first").values
                p = p[(p.ganador.values == 1) | (o <= negativos)]
                partes.append(pd.DataFrame({"g": p.q.values + 1_000_000 * k, "y": p.ganador.values,
                                            **{c: p[c].values for c in SEIS}}))
            t = pd.concat(partes, ignore_index=True).sort_values("g", kind="stable")
            X = pre._transformar(t[SEIS].values)
            g = t.g.values
            ini = np.flatnonzero(np.r_[True, g[1:] != g[:-1]])
            grupo = np.repeat(np.arange(len(ini)), np.diff(np.r_[ini, len(g)]))
            y = t.y.values.astype(float)
            y = y / np.bincount(grupo, weights=y)[grupo]
            self.v.append((V, X, ini, grupo, y))
        self.G = sum(len(ini) for _, _, ini, _, _ in self.v)


def nll_y_grad(X, ini, grupo, y, beta):
    eta = X @ beta
    mx = np.maximum.reduceat(eta, ini)
    ex = np.exp(eta - mx[grupo])
    Z = np.add.reduceat(ex, ini)
    prob = ex / Z[grupo]
    return -(y @ eta - (np.log(Z) + mx).sum()), X.T @ (prob - y)


def ajustar(D, sigma):
    """MAP del modelo jerárquico. Parámetros: μ (6) y δ_v (6 por vertical)."""
    p, nv = len(SEIS), len(D.v)

    def f(th):
        mu, dl = th[:p], th[p:].reshape(nv, p)
        tot, g_mu, g_dl = 0.0, np.zeros(p), np.zeros((nv, p))
        for i, (_, X, ini, grupo, y) in enumerate(D.v):
            v, g = nll_y_grad(X, ini, grupo, y, mu + dl[i])
            tot += v; g_mu += g; g_dl[i] += g
        tot, g_mu, g_dl = tot / D.G, g_mu / D.G, g_dl / D.G
        tot += 0.5 * L2_MU * mu @ mu + 0.5 / sigma ** 2 * (dl * dl).sum() / D.G
        g_mu += L2_MU * mu
        g_dl += dl / sigma ** 2 / D.G
        return tot, np.r_[g_mu, g_dl.ravel()]

    r = minimize(f, np.zeros(p * (nv + 1)), jac=True, method="L-BFGS-B", options={"maxiter": 2000})
    return r.x[:p], r.x[p:].reshape(nv, p), f, r.x


def laplace(f, th, D, eps=1e-4):
    """Covarianza de la posterior ≈ inversa del hessiano de -log posterior (en unidades de licitaciones)."""
    n = len(th)
    H = np.zeros((n, n))
    for i in range(n):
        e = np.zeros(n); e[i] = eps
        H[:, i] = (f(th + e)[1] - f(th - e)[1]) / (2 * eps)
    H = 0.5 * (H + H.T) * D.G                  # f está dividida por G: -log posterior = G·f
    return np.linalg.inv(H)


def puntaje(S, clave, pre, beta):
    return pre._transformar(S[clave].pool[SEIS].values) @ beta


def main():
    S = cargar()
    # estandarización común (se ajusta con los semestres de entrenamiento de todas las verticales)
    pre = bs.Boltzmann(SEIS)
    pre._ajustar_escalas(pd.concat([S[(V, s)].pool[SEIS] for V in VERTICALES for s in TR]).values)

    # ---- elegir σ con las dos ventanas de validación ----
    val = {}
    for sig in SIGMAS:
        rs = []
        for tr, va in VENTANAS:
            D = Datos(S, {V: [(V, s) for s in tr] for V in VERTICALES}, pre)
            mu, dl, _, _ = ajustar(D, sig)
            rs += [bs.rr_rapido(S[(V, va)], puntaje(S, (V, va), pre, mu + dl[i])).mean() for i, V in enumerate(VERTICALES)]
        val[sig] = np.mean(rs)
        log(f"   σ = {sig}: MRR medio de validación {val[sig]:.4f}")
    sig = max(val, key=val.get)
    log(f"σ elegido: {sig}")

    # ---- modelo final en 2024S1 ----
    D = Datos(S, {V: [(V, s) for s in TR] for V in VERTICALES}, pre)
    mu, dl, f, th = ajustar(D, sig)
    C = laplace(f, th, D)
    se = np.sqrt(np.clip(np.diag(C), 0, None))
    p = len(SEIS)
    filas = []
    for i, V in enumerate(VERTICALES):
        r_j = bs.rr_rapido(S[(V, TE)], puntaje(S, (V, TE), pre, mu + dl[i]))
        r_mu = bs.rr_rapido(S[(V, TE)], puntaje(S, (V, TE), pre, mu))
        # vertical NO vista: se reentrena sin ella y se usa sólo μ
        Dn = Datos(S, {W: [(W, s) for s in TR] for W in VERTICALES if W != V}, pre)
        mu_n, _, _, _ = ajustar(Dn, sig)
        r_n = bs.rr_rapido(S[(V, TE)], puntaje(S, (V, TE), pre, mu_n))
        lo, hi = bs.ic_bootstrap(r_n - r_j)
        filas.append({"vertical": V, "jerárquico (μ+δ_v)": r_j.mean(), "sólo lo común (μ)": r_mu.mean(),
                      "vertical no vista (μ sin ella)": r_n.mean(),
                      "no vista − jerárquico (IC95)": f"{(r_n - r_j).mean():+.4f} ({lo:+.4f}, {hi:+.4f})"})
    T = pd.DataFrame(filas)
    B = pd.DataFrame({"μ (común)": mu, "IC95 μ": [f"±{1.96 * s:.2f}" for s in se[:p]]}, index=SEIS)
    for i, V in enumerate(VERTICALES):
        B[f"δ_{V}"] = [f"{dl[i, k]:+.2f} ±{1.96 * se[p * (i + 1) + k]:.2f}" for k in range(p)]
    # fracción de la "variación" de los β que es propia de cada vertical
    frac = (dl ** 2).sum() / ((dl ** 2).sum() + len(VERTICALES) * (mu ** 2).sum())
    T.to_csv("bayes_resultados.csv", index=False)
    B.to_csv("bayes_betas.csv")
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.float_format", "{:.4f}".format):
        print(f"\n=== MODELO DE BOLTZMANN JERÁRQUICO (6 pistas), σ = {sig} ===")
        print(T.to_string(index=False))
        print("\nPesos: lo común (μ) y lo propio de cada vertical (δ_v), con IC95 de Laplace")
        print(B.to_string())
        print(f"\nFracción de la norma de los β que es propia de cada vertical: {frac:.1%}")


if __name__ == "__main__":
    main()
