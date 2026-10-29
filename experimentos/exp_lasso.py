import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # módulos de la raíz
# =====================================================================================
# exp_lasso.py
#
# LASSO "TRONCAL + ESPECÍFICO" SOBRE EL BOLTZMANN DE 44 PISTAS
# (objetivo: medir importancia de pistas, conjunto mínimo, y cuáles son propias
#  de cada vertical, sin sobreajustar y con una metodología que sirva para datos nuevos)
#
# Cada vertical v tiene pesos  β_v = μ + δ_v.
#   μ   = TRONCO común a todas las verticales (lo que se aprende "viendo los datos").
#   δ_v = CORRECCIÓN propia de la vertical v (lo que agrega saber la vertical).
# Se minimiza
#   J = (1/G) Σ_v NLL_v(μ + δ_v) + λ ( Σ_k |μ_k| + r Σ_v Σ_k |δ_v,k| ) + (ε/2)|θ|²
# Es el "data shared lasso" (Gross y Tibshirani 2016; Ollier y Viallon 2017): la L1 sobre μ
# elige las pistas del tronco y la L1 sobre δ_v deja en cero las correcciones que no hacen
# falta. r > 1 hace más caro especializarse; r < 1, más barato. p_b0 no se penaliza en μ
# (es el modelo de referencia, igual que en el Sparse Group Lasso, exp_sgl.py).
#
# Protocolo (sin data leakage):
#   - estandarización ajustada sólo con semestres de entrenamiento;
#   - λ y r se eligen con las dos ventanas de validación (2022S2→2023S1, 2022S2+2023S1→2023S2);
#   - regla anti-sobreajuste: el modelo MÁS CHICO cuyo MRR medio de validación quede a menos
#     de 0,002 del mejor (umbral del equipo);
#   - 2024S1 se mide una sola vez con el modelo elegido.
# El L1 se resuelve exacto con L-BFGS-B separando θ = θ⁺ − θ⁻, θ± ≥ 0.
#
# Necesita los pools de exp_universal.py --preparar (carpeta pools_universal/).
# Uso:  uv run experimentos/exp_lasso.py
# =====================================================================================
import numpy as np
import pandas as pd
from scipy.optimize import minimize

import boltzmann_seleccion as bs
import modelo_v2 as m
from exp_universal import TE, TR, VENTANAS, VERTICALES, SEIS, cargar, Semestre  # Semestre: unpickle
from pista_nombre import VARIABLES_NOMBRE

log = m.log
bs.FAMILIAS["nombre"] = VARIABLES_NOMBRE
bs.FAMILIA_DE.update({v: "nombre" for v in VARIABLES_NOMBRE})
TODAS = bs.variables_de(bs.FAMILIAS)
P = len(TODAS)
I_B0 = TODAS.index("p_b0")
FRACS = [0.3, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]   # λ / λ_max
RS = [0.5, 1.0, 2.0]
EPS = 1e-4          # ridge mínimo (sólo para que el óptimo sea único)
TOL0 = 1e-6         # |β| menor que esto = cero
UMBRAL = 0.002


class Datos:
    def __init__(self, S, claves_por_v, pre, negativos=200, semilla=0):
        rng = np.random.default_rng(semilla)
        self.nombres, self.v = list(claves_por_v), []
        for V, claves in claves_por_v.items():
            partes = []
            for k, cl in enumerate(claves):
                p = S[cl].pool
                p = p[p.groupby("q").ganador.transform("max").values > 0]
                o = pd.Series(rng.random(len(p))).groupby(p.q.values).rank(method="first").values
                p = p[(p.ganador.values == 1) | (o <= negativos)]
                t = pd.DataFrame({"g": p.q.values + 1_000_000 * k, "y": p.ganador.values})
                t = pd.concat([t, pd.DataFrame(pre._transformar(p[TODAS].values).astype(np.float32),
                                               columns=TODAS)], axis=1)
                partes.append(t)
            t = pd.concat(partes, ignore_index=True).sort_values("g", kind="stable")
            X = np.ascontiguousarray(t[TODAS].values, dtype=np.float32)
            g = t.g.values
            ini = np.flatnonzero(np.r_[True, g[1:] != g[:-1]])
            grupo = np.repeat(np.arange(len(ini)), np.diff(np.r_[ini, len(g)]))
            y = t.y.values.astype(np.float64)
            y = y / np.bincount(grupo, weights=y)[grupo]
            self.v.append((X, ini, grupo, y))
        self.G = sum(len(x[1]) for x in self.v)


def nll_grad(X, ini, grupo, y, beta):
    eta = (X @ beta.astype(np.float32)).astype(np.float64)
    mx = np.maximum.reduceat(eta, ini)
    ex = np.exp(eta - mx[grupo])
    Z = np.add.reduceat(ex, ini)
    pr = ex / Z[grupo]
    return -(y @ eta - (np.log(Z) + mx).sum()), (X.T @ (pr - y).astype(np.float32)).astype(np.float64)


def ajustar(D, lam, r, z0=None, soporte=None):
    """Devuelve μ (P) y δ (nv×P). soporte: máscara de θ permitidos (para el reajuste)."""
    nv = len(D.v)
    n = P * (nv + 1)
    w = np.r_[np.ones(P), r * np.ones(P * nv)]
    w[I_B0] = 0.0

    def f(z):
        th = z[:n] - z[n:]
        mu, dl = th[:P], th[P:].reshape(nv, P)
        tot, g = 0.0, np.zeros(n)
        for i, (X, ini, grupo, y) in enumerate(D.v):
            val, gr = nll_grad(X, ini, grupo, y, mu + dl[i])
            tot += val
            g[:P] += gr
            g[P * (i + 1):P * (i + 2)] += gr
        tot, g = tot / D.G, g / D.G
        tot += 0.5 * EPS * th @ th + lam * w @ (z[:n] + z[n:])
        g += EPS * th
        return tot, np.r_[g + lam * w, -g + lam * w]

    cota = [(0, None)] * (2 * n)
    if soporte is not None:
        cota = [(0, None) if s else (0, 0) for s in np.r_[soporte, soporte]]
    z0 = np.zeros(2 * n) if z0 is None else z0
    r_ = minimize(f, z0, jac=True, method="L-BFGS-B", bounds=cota,
                  options={"maxiter": 3000, "maxfun": 6000})
    th = r_.x[:n] - r_.x[n:]
    th[np.abs(th) < TOL0] = 0.0
    return th[:P], th[P:].reshape(nv, P), r_.x


def lam_max(D):
    """λ a partir del cual sólo sobrevive p_b0: máximo |gradiente| del tronco con μ = (p_b0 sólo)."""
    mu, _, _ = ajustar(D, 10.0, 1.0)
    g = np.zeros(P)
    for X, ini, grupo, y in D.v:
        g += nll_grad(X, ini, grupo, y, mu)[1]
    g /= D.G
    g[I_B0] = 0
    return np.abs(g).max()


def puntaje(S, clave, pre, beta):
    return pre._transformar(S[clave].pool[TODAS].values) @ beta


def n_activas(mu, dl):
    return int((mu != 0).sum()), int((dl != 0).sum())


def main():
    S = cargar()
    pre = bs.Boltzmann(TODAS)
    # escalas con una muestra de 150 mil filas por semestre de entrenamiento (cabe en memoria)
    rng = np.random.default_rng(0)
    muestra = []
    for V in VERTICALES:
        for s in TR:
            p = S[(V, s)].pool
            muestra.append(p[TODAS].values[rng.choice(len(p), min(len(p), 150_000), replace=False)])
    pre._ajustar_escalas(np.concatenate(muestra))
    del muestra

    # ---------------- 1. camino de regularización en las dos ventanas ----------------
    filas = []
    for w_i, (tr, va) in enumerate(VENTANAS, 1):
        D = Datos(S, {V: [(V, s) for s in tr] for V in VERTICALES}, pre)
        lm = lam_max(D)
        log(f"ventana v{w_i}: λ_max = {lm:.4g}")
        for r in RS:
            z = None
            for fr in FRACS:
                mu, dl, z = ajustar(D, fr * lm, r, z)
                rrs = {V: bs.rr_rapido(S[(V, va)], puntaje(S, (V, va), pre, mu + dl[i])).mean()
                       for i, V in enumerate(VERTICALES)}
                nm, nd = n_activas(mu, dl)
                filas.append({"ventana": w_i, "r": r, "frac": fr, "n_tronco": nm, "n_especificas": nd,
                              "tronco": "|".join(np.array(TODAS)[mu != 0]),
                              **{f"mrr_{V}": x for V, x in rrs.items()}, "mrr_medio": np.mean(list(rrs.values()))})
                log(f"   r={r} λ/λmax={fr}: tronco {nm}, específicas {nd}, MRR medio {filas[-1]['mrr_medio']:.4f}")
        del D
    C = pd.DataFrame(filas)
    C.to_csv("lasso_camino.csv", index=False)
    R = (C.groupby(["r", "frac"]).agg(mrr=("mrr_medio", "mean"), n_tronco=("n_tronco", "mean"),
                                      n_esp=("n_especificas", "mean")).reset_index())
    R["n_total"] = R.n_tronco + R.n_esp
    mejor = R.mrr.max()
    cand = R[R.mrr >= mejor - UMBRAL].sort_values(["n_total", "mrr"], ascending=[True, False])
    r_sel, fr_sel = cand.iloc[0].r, cand.iloc[0].frac
    log(f"mejor MRR validación {mejor:.4f}; elegido (más chico a < {UMBRAL}): r={r_sel}, λ/λmax={fr_sel}")

    # estabilidad: pistas del tronco en cada ventana con (r, λ) elegidos
    est = C[(C.r == r_sel) & (C.frac == fr_sel)].set_index("ventana").tronco
    t1, t2 = set(est[1].split("|")), set(est[2].split("|"))

    # ---------------- 2. modelo final y prueba única en 2024S1 ----------------
    D = Datos(S, {V: [(V, s) for s in TR] for V in VERTICALES}, pre)
    lm = lam_max(D)
    mu, dl, _ = ajustar(D, fr_sel * lm, r_sel)
    sop = np.r_[mu != 0, (dl != 0).ravel()]
    sop[I_B0] = True
    mu_r, dl_r, _ = ajustar(D, 0.0, r_sel, soporte=sop)      # reajuste sin L1 sobre el soporte
    mu_l2, dl_l2, _ = ajustar(D, 0.0, 1.0, soporte=np.r_[np.ones(P, bool), np.zeros(P * len(VERTICALES), bool)])
    filas = []
    for i, V in enumerate(VERTICALES):
        k = (V, TE)
        r_j = bs.rr_rapido(S[k], puntaje(S, k, pre, mu + dl[i]))
        r_t = bs.rr_rapido(S[k], puntaje(S, k, pre, mu))
        r_rf = bs.rr_rapido(S[k], puntaje(S, k, pre, mu_r + dl_r[i]))
        r_g = bs.rr_rapido(S[k], puntaje(S, k, pre, mu_l2))
        Dn = Datos(S, {W: [(W, s) for s in TR] for W in VERTICALES if W != V}, pre)
        mu_n, _, _ = ajustar(Dn, fr_sel * lam_max(Dn), r_sel)
        r_n = bs.rr_rapido(S[k], puntaje(S, k, pre, mu_n))
        del Dn
        lo, hi = bs.ic_bootstrap(r_rf - r_g)
        lo2, hi2 = bs.ic_bootstrap(r_n - r_j)
        filas.append({"vertical": V, "lasso tronco+δ": r_j.mean(), "lasso sólo tronco": r_t.mean(),
                      "lasso reajustado": r_rf.mean(), "global 44 (L2 mínima)": r_g.mean(),
                      "vertical no vista (tronco)": r_n.mean(),
                      "reajustado − global44 (IC95)": f"{(r_rf - r_g).mean():+.4f} ({lo:+.4f}, {hi:+.4f})",
                      "no vista − tronco+δ (IC95)": f"{(r_n - r_j).mean():+.4f} ({lo2:+.4f}, {hi2:+.4f})"})
        log(f"   {V}: {filas[-1]}")
    T = pd.DataFrame(filas)
    B = pd.DataFrame({"familia": [bs.FAMILIA_DE.get(v, "") for v in TODAS], "tronco μ": mu,
                      **{f"δ_{V}": dl[i] for i, V in enumerate(VERTICALES)},
                      "en tronco v1": [v in t1 for v in TODAS], "en tronco v2": [v in t2 for v in TODAS],
                      "en escalera (6)": [v in SEIS for v in TODAS]}, index=TODAS)
    B = B[(B["tronco μ"] != 0) | (B.filter(like="δ_") != 0).any(axis=1)]
    B = B.reindex(B["tronco μ"].abs().sort_values(ascending=False).index)
    T.to_csv("lasso_resultados.csv", index=False)
    B.to_csv("lasso_betas.csv")
    R.to_csv("lasso_resumen_camino.csv", index=False)
    nm, nd = n_activas(mu, dl)
    with pd.option_context("display.width", 250, "display.max_columns", 40, "display.float_format", "{:.4f}".format):
        print("\n=== CAMINO (promedio de las dos ventanas de validación) ===")
        print(R.to_string(index=False))
        print(f"\nElegido: r = {r_sel}, λ/λmax = {fr_sel}  →  tronco {nm} pistas, {nd} correcciones por vertical")
        print("\n=== PRUEBA ÚNICA EN 2024S1 ===")
        print(T.to_string(index=False))
        print("\n=== PESOS DEL MODELO ELEGIDO (sólo los no nulos) ===")
        print(B.round(3).to_string())


if __name__ == "__main__":
    main()
