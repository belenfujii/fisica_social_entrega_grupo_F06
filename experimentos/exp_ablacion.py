import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # módulos de la raíz
# Ablación por idea sobre el Boltzmann global de 44 pistas: se saca una idea entera, se reentrena
# con las cinco verticales (2022S2–2023S2) y se mide en 2024S1. También "solo memoria" (B0 dentro
# del Boltzmann) como referencia nula, y la matriz de correlación de las 6 pistas de la escalera.
import numpy as np, pandas as pd
import boltzmann_seleccion as bs
from exp_universal import TE, TR, VERTICALES, SEIS, cargar, Semestre
from pista_nombre import VARIABLES_NOMBRE
bs.FAMILIAS["nombre"] = VARIABLES_NOMBRE
TODAS = bs.variables_de(bs.FAMILIAS)
IDEAS = {"memoria": bs.variables_de(["b0", "niveles_b0", "actividad"]) + ["ent_x_p_b0"],
         "texto": bs.variables_de(["tfidf", "titulo", "contrato_parecido", "bm25", "temas_nmf"]) + ["ent_x_tfidf", "ent_x_hist"],
         "nombre": VARIABLES_NOMBRE,
         "red": bs.variables_de(["difusion", "fermi_dirac", "correspondencias"])}
assert sorted(sum(IDEAS.values(), [])) == sorted(TODAS)
S = cargar()
def correr(vv):
    mod = bs.Boltzmann(vv, l2=0.01).fit([S[(V, s)].pool for V in VERTICALES for s in TR])
    return {V: bs.rr_rapido(S[(V, TE)], mod.predict(S[(V, TE)].pool[vv].values)) for V in VERTICALES}
full = correr(TODAS)
filas = [{"modelo": "completo (44)", **{V: full[V].mean() for V in VERTICALES}}]
for idea, vs in list(IDEAS.items()) + [("todo salvo memoria (solo B0 en Boltzmann)", None)]:
    vv = IDEAS["memoria"] if vs is None else [v for v in TODAS if v not in vs]
    r = correr(vv)
    f = {"modelo": ("sin " + idea) if vs is not None else "solo memoria"}
    for V in VERTICALES:
        lo, hi = bs.ic_bootstrap(full[V] - r[V])
        f[V] = r[V].mean(); f[f"{V}_perdida"] = f"{(full[V]-r[V]).mean():+.3f}{'*' if lo > 0 or hi < 0 else ''}"
    filas.append(f); print(f, flush=True)
T = pd.DataFrame(filas); T.to_csv("ablacion_ideas.csv", index=False)
print(T.to_string())
rng = np.random.default_rng(0)
X = pd.concat([S[(V, s)].pool[TODAS].sample(50000, random_state=0) for V in VERTICALES for s in TR])
C = X.corr(method="spearman")
print("\nSpearman escalera:\n", C.loc[SEIS, SEIS].round(2).to_string())
pares = [("log_puesto_tfidf", "log_puesto_titulo"), ("log_puesto_tfidf", "log_puesto_bm25"), ("log_puesto_tfidf", "log_puesto_hist"),
         ("sim_tfidf", "sim_bm25"), ("sim_hist", "log_puesto_hist"), ("p_b0", "log_puesto_b0"), ("exceso_fd", "p_nulo_fd")]
for a, b in pares: print(a, b, round(C.loc[a, b], 2))
C.to_csv("correlacion_spearman.csv")
