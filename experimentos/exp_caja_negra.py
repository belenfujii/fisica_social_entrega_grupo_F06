import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # módulos de la raíz
# =====================================================================================
# exp_caja_negra.py   (CORRER EN LA COMPUTADORA DEL EQUIPO: necesita lightgbm)
#
# COMPARACIÓN CON UNA "CAJA NEGRA": ¿cuánto se pierde por usar un modelo interpretable?
# Compara, con el mismo pool y el mismo protocolo, tres modelos GLOBALES (una sola fórmula
# para todas las verticales):
#   A. Boltzmann tronco (lasso, exp_lasso.py: r = 1, λ/λmax = 0.002, sólo μ).
#   B. LambdaMART global con las 44 pistas (incluye la sorpresa del nombre).
#   C. Mezcla:  s = w · z(Boltzmann) + (1 − w) · z(LambdaMART),  z = puntaje estandarizado
#      dentro de cada licitación. w se elige con las dos ventanas de validación.
# 2024S1 se mide una sola vez con el w elegido.
#
# Antes: uv run experimentos/exp_universal.py --preparar d   (y it, j, r, rd) → carpeta pools_universal/
# Uso:   uv run experimentos/exp_caja_negra.py     (necesita lightgbm: uv sync --group dev)
# =====================================================================================
import numpy as np
import pandas as pd

import boltzmann_seleccion as bs
import modelo_v2 as m
from exp_lasso import TODAS, Datos, ajustar, lam_max, puntaje
from exp_universal import TE, TR, VENTANAS, VERTICALES, cargar, Semestre  # Semestre: unpickle

log = m.log
FRAC, R_LASSO = 0.002, 1.0
PESOS = [0.0, 0.25, 0.5, 0.75, 1.0]


def lambdamart(S, claves, negativos=200, semilla=0):
    import lightgbm as lgb
    rng = np.random.default_rng(semilla)
    partes = []
    for k, cl in enumerate(claves):
        p = S[cl].pool
        p = p[p.groupby("q").ganador.transform("max").values > 0]
        o = pd.Series(rng.random(len(p))).groupby(p.q.values).rank(method="first").values
        p = p[(p.ganador.values == 1) | (o <= negativos)]
        partes.append(pd.DataFrame(p[TODAS].values, columns=TODAS).assign(
            g=p.q.values + 1_000_000 * k, y=p.ganador.values))
    t = pd.concat(partes, ignore_index=True).sort_values("g", kind="stable")
    mod = lgb.LGBMRanker(objective="lambdarank", n_estimators=600, learning_rate=0.03, num_leaves=31,
                         min_child_samples=50, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                         verbose=-1, random_state=0)
    mod.fit(t[TODAS].values, t.y.values, group=t.groupby("g", sort=False).size().values)
    return mod


def z_por_licitacion(q, s):
    s = pd.Series(s)
    g = s.groupby(q)
    return ((s - g.transform("mean")) / g.transform("std").replace(0, 1).fillna(1)).values


def puntajes(S, pre, mu, lm, clave):
    p = S[clave].pool
    sb = puntaje(S, clave, pre, mu)
    sl = lm.predict(p[TODAS].values)
    return z_por_licitacion(p.q.values, sb), z_por_licitacion(p.q.values, sl)


def entrenar(S, pre, semestres):
    D = Datos(S, {V: [(V, s) for s in semestres] for V in VERTICALES}, pre)
    mu, _, _ = ajustar(D, FRAC * lam_max(D), R_LASSO)
    del D
    lm = lambdamart(S, [(V, s) for V in VERTICALES for s in semestres])
    return mu, lm


def main():
    S = cargar()
    pre = bs.Boltzmann(TODAS)
    rng = np.random.default_rng(0)
    muestra = [S[(V, s)].pool[TODAS].values[rng.choice(len(S[(V, s)].pool),
               min(len(S[(V, s)].pool), 150_000), replace=False)] for V in VERTICALES for s in TR]
    pre._ajustar_escalas(np.concatenate(muestra))
    del muestra

    # ---- elegir w en validación ----
    val = {w: [] for w in PESOS}
    for tr, va in VENTANAS:
        mu, lm = entrenar(S, pre, tr)
        for V in VERTICALES:
            zb, zl = puntajes(S, pre, mu, lm, (V, va))
            for w in PESOS:
                val[w].append(bs.rr_rapido(S[(V, va)], w * zb + (1 - w) * zl).mean())
    for w in PESOS:
        log(f"   w = {w}: MRR medio de validación {np.mean(val[w]):.4f}")
    w_sel = max(PESOS, key=lambda w: np.mean(val[w]))
    log(f"w elegido = {w_sel}  (1 = sólo Boltzmann, 0 = sólo LambdaMART)")

    # ---- prueba única en 2024S1 ----
    mu, lm = entrenar(S, pre, TR)
    filas = []
    for V in VERTICALES:
        zb, zl = puntajes(S, pre, mu, lm, (V, TE))
        rb = bs.rr_rapido(S[(V, TE)], zb)
        rl = bs.rr_rapido(S[(V, TE)], zl)
        rm = bs.rr_rapido(S[(V, TE)], w_sel * zb + (1 - w_sel) * zl)
        f = {"vertical": V, "Boltzmann tronco": rb.mean(), "LambdaMART global": rl.mean(),
             f"mezcla (w={w_sel})": rm.mean()}
        for et, x in (("LambdaMART − Boltzmann", rl - rb), ("mezcla − Boltzmann", rm - rb)):
            lo, hi = bs.ic_bootstrap(x)
            f[f"{et} (IC95)"] = f"{x.mean():+.4f} ({lo:+.4f}, {hi:+.4f})"
        filas.append(f)
    T = pd.DataFrame(filas)
    T.to_csv("caja_negra_resultados.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 20, "display.float_format", "{:.4f}".format):
        print("\n=== CAJA NEGRA: MRR en 2024S1 (prueba única) ===")
        print(T.to_string(index=False))
        print("promedio:", T.select_dtypes("number").mean().round(4).to_dict())


if __name__ == "__main__":
    main()
