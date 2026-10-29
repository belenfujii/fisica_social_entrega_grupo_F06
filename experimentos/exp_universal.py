import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # módulos de la raíz
# =====================================================================================
# exp_universal.py
#
# ¿CUÁNTO AGREGA SABER LA VERTICAL? (objetivo: una metodología que ande igual de bien
# en cualquier vertical, también en una que el modelo nunca vio, sin sobreajustar)
#
# Se comparan tres formas de entrenar los MISMOS modelos (6 pistas y 44 pistas):
#   A. por vertical        : β entrenados con los semestres de entrenamiento de esa vertical.
#   B. global              : un solo juego de β, entrenado con las cinco verticales juntas.
#   C. vertical no vista   : β entrenados con las OTRAS cuatro verticales y aplicados a la
#                            quinta (dejar-una-afuera). Es el caso de datos de un sector nuevo.
# Todo se mide en 2024S1. A − C = cuánto se gana por conocer la vertical.
# La L2 de cada caso se elige con las dos ventanas de validación de las verticales de
# entrenamiento (nunca con la vertical de prueba ni con 2024S1).
#
# Uso (dos pasos, para no tener las cinco verticales en memoria a la vez):
#   uv run experimentos/exp_universal.py --preparar j      (y lo mismo con r, it, d, rd)
#   uv run experimentos/exp_universal.py --analizar
# =====================================================================================
import argparse
import os
import pickle

import numpy as np
import pandas as pd

import boltzmann_seleccion as bs
import modelo_v2 as m
from pista_nombre import VARIABLES_NOMBRE, PistaNombre

log = m.log
VERTICALES = ["d", "it", "j", "r", "rd"]
VENTANAS = [(["2022S2"], "2023S1"), (["2022S2", "2023S1"], "2023S2")]
TR, TE = ["2022S2", "2023S1", "2023S2"], "2024S1"
L2S = [0.001, 0.01, 0.1]
SEIS = ["p_b0", "log_puesto_tfidf", "sim_hist", "js_temas", "nombre_sorpresa", "exceso_fd"]
CARPETA = "pools_universal"


class Semestre:
    """Lo mínimo para entrenar y medir: el pool y cuántos ganadores tiene cada licitación."""
    def __init__(self, pool, n_gan, q):
        self.pool, self.n_gan, self.q = pool, n_gan, q


def preparar(V, K=300):
    D = m.Datos(V)
    D.pista_extra = PistaNombre(D)
    os.makedirs(CARPETA, exist_ok=True)
    for s in bs.SEMESTRES:
        d = m.preparar(D, s, K=K, fisica=True)
        bs.agregar_pistas(D, d)
        bs.FAMILIAS["nombre"] = VARIABLES_NOMBRE
        cols = ["q", "c", "ganador", "puesto_min"] + bs.variables_de(bs.FAMILIAS)
        p = d.pool[cols].copy()
        for c in cols[4:]:
            p[c] = p[c].astype(np.float32)
        with open(f"{CARPETA}/{V}_{s}.pkl", "wb") as f:
            pickle.dump(Semestre(p, d.n_gan, d.q[["id"]].copy()), f)
        log(f"{V} {s}: guardado ({len(p):,} filas)")


def cargar():
    S = {}
    for V in VERTICALES:
        for s in bs.SEMESTRES:
            with open(f"{CARPETA}/{V}_{s}.pkl", "rb") as f:
                S[(V, s)] = pickle.load(f)
    return S


def ajustar(S, claves, vv, l2):
    return bs.Boltzmann(vv, l2=l2).fit([S[k].pool for k in claves])


def rr(S, k, mod, vv):
    return bs.rr_rapido(S[k], mod.predict(S[k].pool[vv].values))


def elegir_l2(S, verts, vv):
    """L2 con las dos ventanas de validación, promediando sobre las verticales de entrenamiento."""
    def media(l2):
        vals = []
        for tr, va in VENTANAS:
            mod = ajustar(S, [(V, s) for V in verts for s in tr], vv, l2)
            vals += [rr(S, (V, va), mod, vv).mean() for V in verts]
        return np.mean(vals)
    return max(L2S, key=media)


def analizar():
    S = cargar()
    bs.FAMILIAS["nombre"] = VARIABLES_NOMBRE
    bs.FAMILIA_DE.update({v: "nombre" for v in VARIABLES_NOMBRE})
    TODAS = bs.variables_de(bs.FAMILIAS)
    filas, betas = [], {}
    for nombre_mod, vv in (("6 pistas", SEIS), ("44 pistas", TODAS)):
        # B. global: una sola L2 y un solo juego de β para todas
        l2g = elegir_l2(S, VERTICALES, vv)
        glob = ajustar(S, [(V, s) for V in VERTICALES for s in TR], vv, l2g)
        if nombre_mod == "6 pistas":
            betas["global"] = glob.tabla_beta().set_index("pista").beta
        for V in VERTICALES:
            # A. por vertical
            l2a = elegir_l2(S, [V], vv)
            a = ajustar(S, [(V, s) for s in TR], vv, l2a)
            # C. vertical no vista
            otras = [x for x in VERTICALES if x != V]
            l2c = elegir_l2(S, otras, vv)
            c = ajustar(S, [(x, s) for x in otras for s in TR], vv, l2c)
            ra, rb, rc = rr(S, (V, TE), a, vv), rr(S, (V, TE), glob, vv), rr(S, (V, TE), c, vv)
            if nombre_mod == "6 pistas":
                betas[f"{V} (por vertical)"] = a.tabla_beta().set_index("pista").beta
                betas[f"{V} (sin verla)"] = c.tabla_beta().set_index("pista").beta
            f = {"modelo": nombre_mod, "vertical": V, "A por vertical": ra.mean(), "B global": rb.mean(),
                 "C vertical no vista": rc.mean()}
            for et, x in (("B − A", rb - ra), ("C − A", rc - ra)):
                lo, hi = bs.ic_bootstrap(x)
                f[f"{et} (IC95)"] = f"{x.mean():+.4f} ({lo:+.4f}, {hi:+.4f})"
            filas.append(f)
            log(f"   {nombre_mod} {V}: A {ra.mean():.4f}  B {rb.mean():.4f}  C {rc.mean():.4f}")
    T = pd.DataFrame(filas)
    Bt = pd.DataFrame(betas).reindex(SEIS)
    T.to_csv("universal_resultados.csv", index=False)
    Bt.to_csv("universal_betas.csv")
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.float_format", "{:.4f}".format):
        print("\n=== ¿CUÁNTO AGREGA SABER LA VERTICAL? (MRR en 2024S1) ===")
        print(T.to_string(index=False))
        print("\n=== β del modelo de 6 pistas según cómo se entrena ===")
        print(Bt.round(2).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preparar")
    ap.add_argument("--analizar", action="store_true")
    a = ap.parse_args()
    if a.preparar:
        preparar(a.preparar)
    if a.analizar:
        analizar()


if __name__ == "__main__":
    main()
