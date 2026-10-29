import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # módulos de la raíz
# =====================================================================================
# exp_empresas_falsas.py
#
# MODELO NULO DE "EMPRESAS FALSAS" (control para el texto)
#
# Se fabrican empresas falsas: cada empresa conserva su nombre, sus códigos (PSC, NAICS,
# oficinas), su historial de adjudicaciones y su cantidad de contratos, pero los TEXTOS se
# reparten al azar:
#   - las descripciones de los contratos del historial se permutan entre todos los contratos;
#   - el perfil de texto de cada empresa (texto_awards) se permuta entre las empresas del semestre.
# Así se conservan exactamente las frecuencias de palabras (las columnas de la matriz TF-IDF
# tienen las mismas palabras) pero se rompe el vínculo entre lo que una empresa hizo y lo que
# dice su texto. Si el texto sólo aportara por azar o por efectos de tamaño, el MRR no cambiaría.
#
# Con las empresas falsas se recalculan TODAS las pistas, se reentrena el mismo Boltzmann de
# 44 pistas (2022S2–2023S2) y se mide en 2024S1. Se compara con el mismo modelo con los datos
# reales (pools de exp_universal.py).
#
# Uso:  uv run experimentos/exp_empresas_falsas.py --verticales d,it --semillas 0
# =====================================================================================
import argparse
import pickle

import numpy as np
import pandas as pd

import boltzmann_seleccion as bs
import modelo_v2 as m
from exp_universal import TE, TR, Semestre
from pista_nombre import VARIABLES_NOMBRE, PistaNombre

log = m.log
bs.FAMILIAS["nombre"] = VARIABLES_NOMBRE
TODAS = bs.variables_de(bs.FAMILIAS)
TEXTO = bs.variables_de(["tfidf", "titulo", "contrato_parecido", "bm25", "temas_nmf"])


def pools_falsos(V, semilla, K=300):
    D = m.Datos(V)
    D.pista_extra = PistaNombre(D)
    rng = np.random.default_rng(semilla)
    D.hist["desc_limpia"] = D.hist.desc_limpia.values[rng.permutation(len(D.hist))]
    for s, idx in D.emp.groupby("semestre").groups.items():
        idx = np.asarray(idx)
        D.emp.loc[idx, "texto_limpio"] = D.emp.loc[idx, "texto_limpio"].values[rng.permutation(len(idx))]
    S = {}
    for s in bs.SEMESTRES:
        d = m.preparar(D, s, K=K, fisica=True)
        bs.agregar_pistas(D, d)
        S[s] = Semestre(d.pool[["q", "c", "ganador", "puesto_min"] + TODAS].copy(), d.n_gan, d.q[["id"]].copy())
        log(f"{V} {s} falso: {len(d.pool):,} filas")
    return S


def reales(V):
    S = {}
    for s in bs.SEMESTRES:
        with open(f"pools_universal/{V}_{s}.pkl", "rb") as f:
            S[s] = pickle.load(f)
    return S


def medir(S, vv, l2=0.01):
    mod = bs.Boltzmann(vv, l2=l2).fit([S[s].pool for s in TR])
    return bs.rr_rapido(S[TE], mod.predict(S[TE].pool[vv].values))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verticales", default="d,it")
    ap.add_argument("--semillas", default="0")
    a = ap.parse_args()
    sin_texto = [v for v in TODAS if v not in TEXTO]
    filas = []
    for V in a.verticales.split(","):
        R = reales(V)
        r_real, r_sin = medir(R, TODAS), medir(R, sin_texto)
        del R
        for sem in map(int, a.semillas.split(",")):
            F = pools_falsos(V, sem)
            r_falso = medir(F, TODAS)
            cob = np.mean([F[TE].pool.groupby("q").ganador.max().mean()])
            del F
            lo, hi = bs.ic_bootstrap(r_falso - r_real)
            filas.append({"vertical": V, "semilla": sem, "real (44)": r_real.mean(),
                          "empresas falsas (44)": r_falso.mean(), "sin pistas de texto": r_sin.mean(),
                          "falso − real (IC95)": f"{(r_falso - r_real).mean():+.4f} ({lo:+.4f}, {hi:+.4f})",
                          "cobertura pool falso": cob})
            log(f"   {filas[-1]}")
            pd.DataFrame(filas).to_csv(f"empresas_falsas_{a.verticales.replace(',', '_')}.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 20, "display.float_format", "{:.4f}".format):
        print("\n=== MODELO NULO DE EMPRESAS FALSAS (MRR en 2024S1) ===")
        print(pd.DataFrame(filas).to_string(index=False))


if __name__ == "__main__":
    main()
