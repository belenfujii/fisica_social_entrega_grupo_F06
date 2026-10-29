import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # módulos de la raíz
# =====================================================================================
# exp_sgl.py — Sparse Group Lasso sobre el MISMO pool y protocolo,
# para comparar en igualdad de condiciones con el tronco lasso (exp_lasso.py).
#
#   J(β) = NLL(β)/G + λ [ α Σ_k |β_k| + (1−α) Σ_g √|g| ‖β_g‖₂ ] + (ε/2)|β|²
#   (Simon, Friedman, Hastie y Tibshirani 2013). Grupos = familias de pistas.
#   p_b0 y log_puesto_b0 no se penalizan.
# Se resuelve con gradiente proximal acelerado (FISTA) con búsqueda de paso.
#
# Dos variantes:
#   global      : un solo β para las cinco verticales (como el modelo final).
#   por vertical: un β por vertical (la versión por rubro).
# λ y α se eligen con las dos ventanas de validación (máximo MRR medio). 2024S1 se mide una vez.
# =====================================================================================
import numpy as np
import pandas as pd

import boltzmann_seleccion as bs
import modelo_v2 as m
from exp_lasso import TODAS, Datos, nll_grad, puntaje
from exp_universal import TE, TR, VENTANAS, VERTICALES, cargar, Semestre  # Semestre: unpickle

log = m.log
P = len(TODAS)
GRUPOS = {}
for i, v in enumerate(TODAS):
    GRUPOS.setdefault(bs.FAMILIA_DE.get(v, "nombre"), []).append(i)
LIBRES = [TODAS.index("p_b0"), TODAS.index("log_puesto_b0")]
FRACS = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
ALFAS = [0.4, 0.8]
EPS = 1e-4


def f_grad(D, b):
    tot, g = 0.0, np.zeros(P)
    for X, ini, grupo, y in D.v:
        v, gr = nll_grad(X, ini, grupo, y, b)
        tot += v; g += gr
    return tot / D.G + 0.5 * EPS * b @ b, g / D.G + EPS * b


def prox(v, t, lam, alfa):
    out = v.copy()
    for fam, idx in GRUPOS.items():
        idx = [i for i in idx if i not in LIBRES]
        if not idx:
            continue
        z = np.sign(v[idx]) * np.maximum(np.abs(v[idx]) - t * lam * alfa, 0)
        nz = np.linalg.norm(z)
        k = t * lam * (1 - alfa) * np.sqrt(len(idx))
        out[idx] = 0 if nz <= k else z * (1 - k / nz)
    return out


def ajustar(D, lam, alfa, b0=None, iters=400):
    b = np.zeros(P) if b0 is None else b0.copy()
    yv, tk, L = b.copy(), 1.0, 1.0
    fb = None
    for _ in range(iters):
        fy, gy = f_grad(D, yv)
        while True:
            bn = prox(yv - gy / L, 1 / L, lam, alfa)
            d = bn - yv
            fn, _ = f_grad(D, bn)
            if fn <= fy + gy @ d + 0.5 * L * d @ d + 1e-12:
                break
            L *= 2
        tn = (1 + np.sqrt(1 + 4 * tk * tk)) / 2
        yv = bn + (tk - 1) / tn * (bn - b)
        if fb is not None and abs(fb - fn) < 1e-7:
            b = bn; break
        b, tk, fb = bn, tn, fn
        L *= 0.9
    b[np.abs(b) < 1e-6] = 0
    return b


def lam_max(D):
    b = np.zeros(P)
    b = ajustar(D, 100.0, 0.5, iters=200)          # sólo quedan las pistas libres del B0
    g = f_grad(D, b)[1]
    g[LIBRES] = 0
    return np.abs(g).max()


def camino(D, lm):
    res = {}
    for a in ALFAS:
        b = None
        for fr in FRACS:
            b = ajustar(D, fr * lm, a, b)
            res[(a, fr)] = b.copy()
    return res


def main():
    S = cargar()
    pre = bs.Boltzmann(TODAS)
    rng = np.random.default_rng(0)
    pre._ajustar_escalas(np.concatenate([S[(V, s)].pool[TODAS].values[rng.choice(len(S[(V, s)].pool), min(len(S[(V, s)].pool), 150_000), replace=False)]
                                         for V in VERTICALES for s in TR]))
    filas = []
    for modo in ["global", "por vertical"]:
        grupos_v = [VERTICALES] if modo == "global" else [[V] for V in VERTICALES]
        for verts in grupos_v:
            val = {}
            for tr, va in VENTANAS:
                D = Datos(S, {"todo": [(V, s) for V in verts for s in tr]}, pre)
                res = camino(D, lam_max(D))
                for k, b in res.items():
                    val.setdefault(k, []).append(np.mean([bs.rr_rapido(S[(V, va)], puntaje(S, (V, va), pre, b)).mean() for V in verts]))
                del D
            k = max(val, key=lambda q: np.mean(val[q]))
            D = Datos(S, {"todo": [(V, s) for V in verts for s in TR]}, pre)
            lm = lam_max(D)
            b = None
            for fr in FRACS:                      # mismo camino con arranque en caliente hasta el elegido
                b = ajustar(D, fr * lm, k[0], b)
                if fr == k[1]:
                    break
            del D
            for V in verts:
                r = bs.rr_rapido(S[(V, TE)], puntaje(S, (V, TE), pre, b))
                filas.append({"modo": modo, "vertical": V, "alfa": k[0], "frac": k[1], "val": np.mean(val[k]),
                              "pistas": int((b != 0).sum()), "mrr_2024S1": r.mean()})
                np.save(f"sgl_rr_{modo.replace(' ', '_')}_{V}.npy", r)
                log(f"   {filas[-1]}")
    T = pd.DataFrame(filas)
    T.to_csv("sgl_resultados.csv", index=False)
    with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
        print(T.to_string(index=False))
        print(T.groupby("modo").mrr_2024S1.mean())


if __name__ == "__main__":
    main()
