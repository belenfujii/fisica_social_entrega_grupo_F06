# =====================================================================================
# boltzmann_seleccion.py
#
# Qué hace este script, en una frase:
#   arma un modelo de Boltzmann (máxima entropía) sobre el pool y las pistas de modelo_v2.py,
#   le agrega pistas nuevas (BM25, temas NMF, análisis de correspondencias) y
#   decide, por ABLACIÓN, qué familias de pistas quedan y cuáles se sacan.
#
# Uso (desde la carpeta del repo):
#   uv sync --group dev                       (una sola vez; trae lightgbm)
#   uv run boltzmann_seleccion.py --vertical j
#
# Opciones:
#   --K 300             tamaño del pool por fuente
#   --umbral 0.002      una familia se saca si quitarla baja el MRR menos que esto
#   --sin-lambdamart    no entrena LambdaMART para comparar (más rápido)
#
# Regla de oro (data leakage):
#   - TODO lo que se calcula para un semestre usa sólo información anterior a su inicio
#     (el pool y las pistas de modelo_v2.preparar ya lo cumplen; las pistas nuevas de acá
#     también: el vocabulario, los temas NMF y la red oficina–empresa se arman con lo previo).
#   - La SELECCIÓN de familias se hace con validación: se entrena con 2022S2 + 2023S1 y se
#     mide en 2023S2. El semestre de prueba (2024S1) se mira UNA sola vez, al final.
#     Si elegimos mirando 2024S1, estaríamos usando la prueba para entrenar.
#
# Referencias:
#   - Jaynes (1957), Information theory and statistical mechanics, Phys. Rev. 106, 620.
#     La distribución de Boltzmann es la de máxima entropía que reproduce los promedios.
#   - McFadden (1974), Conditional logit analysis of qualitative choice behavior.
#     Es el mismo modelo, visto desde la economía.
#   - McFadden (1978), Modelling the choice of residential location. Muestra que se puede
#     ajustar usando, en cada licitación, el ganador más una muestra AL AZAR de las demás
#     candidatas, y la estimación sigue siendo consistente. Lo usamos para ir más rápido.
#   - Robertson y Zaragoza (2009), BM25.
#   - Lee y Seung (1999), Nature 401, 788: NMF. Pinto et al. (2019), Physica A 524, 614:
#     TF-IDF + NMF + Jensen-Shannon para agendas de medios.
#   - Greenacre (2007), Correspondence Analysis in Practice. del Pozo, Pinto et al. (2025),
#     Sci. Rep.: análisis de correspondencias sobre una red bipartita.
#
# Salidas (en la carpeta desde donde se corre):
#   ablacion_{V}.csv            cuánto baja el MRR de validación al sacar cada familia
#   seleccion_pasos_{V}.csv     la eliminación paso a paso
#   resultados_boltzmann_{V}.csv  MRR en 2024S1 de cada modelo, con IC95
#   comparaciones_{V}.csv       diferencias entre modelos con IC95 (bootstrap pareado)
#   beta_seleccionado_{V}.csv   los β del Boltzmann final, con su familia
#   beta_completo_{V}.csv       los β del Boltzmann con todas las familias
#   temas_{V}.csv               las palabras de cada tema NMF (para interpretar)
#   ranking_boltzmann_{V}.csv   el ranking final de 2024S1 en el formato de evaluar.py
# =====================================================================================
from __future__ import annotations

import argparse
import gc
import warnings

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize
from scipy.sparse.linalg import LinearOperator, svds
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import evaluar
import modelo_v2 as m

log = m.log
# NMF avisa cuando llega al máximo de iteraciones; para temas alcanza, no hace falta verlo
warnings.filterwarnings("ignore", message="Maximum number of iterations")
SEMESTRES = ["2022S2", "2023S1", "2023S2", "2024S1"]


# =====================================================================================
# 1. LAS FAMILIAS DE PISTAS
#    Una "familia" es un grupo de pistas que miden lo mismo. La ablación saca familias
#    enteras, no pistas sueltas: si sacáramos de a una, una pista "gemela" (por ejemplo
#    sim_tfidf y sim_tfidf_rel) taparía a la otra y ninguna parecería importante.
#
#    No entran p_b0_max_lic ni entropia_b0 solas: valen lo mismo para todas las candidatas
#    de una licitación, así que se cancelan con Z y su β siempre da 0.
#    La entropía entra multiplicada por otras pistas (interacciones), que sí varían.
# =====================================================================================
FAMILIAS = {
    # --- historial (lo que ya hace el B0) ---
    "b0":                ["p_b0", "log_puesto_b0"],
    "niveles_b0":        [f"nivel_{n}" for n in m.NOMBRES_NIVELES],
    "actividad":         ["log_n_awards", "log_n_rec", "log_dias_ultimo"],
    # --- texto ---
    "tfidf":             ["sim_tfidf", "sim_tfidf_rel", "log_puesto_tfidf"],
    "titulo":            ["sim_titulo", "sim_titulo_rel", "log_puesto_titulo"],
    "contrato_parecido": ["sim_hist", "sim_hist_rel", "log_puesto_hist"],
    "bm25":              ["sim_bm25", "sim_bm25_rel", "log_puesto_bm25"],
    "temas_nmf":         ["js_temas", "entropia_temas"],
    # --- red oficina–empresa ---
    "difusion":          ["dif_masa_oficina", "dif_calor_oficina", "dif_mixta_oficina",
                          "dif_masa_psc", "log_puesto_dif"],
    "fermi_dirac":       ["p_nulo_fd", "exceso_fd"],
    "correspondencias":  ["afinidad_ca", "residuo_ca"],
    # --- física: cuánto duda el B0 (interacciones con la entropía) ---
    "entropia":          ["ent_x_p_b0", "ent_x_tfidf", "ent_x_hist"],
}
FAMILIA_DE = {v: f for f, vs in FAMILIAS.items() for v in vs}

# Pistas con colas muy largas (unas pocas empresas tienen valores enormes). Se comprimen con
# asinh(x / escala), que es como un logaritmo pero acepta negativos y el cero.
COLAS_LARGAS = (["p_b0", "ent_x_p_b0", "exceso_fd", "afinidad_ca", "residuo_ca"]
                + [f"nivel_{n}" for n in m.NOMBRES_NIVELES]
                + ["dif_masa_oficina", "dif_calor_oficina", "dif_mixta_oficina", "dif_masa_psc"])


def variables_de(familias):
    return [v for f in familias for v in FAMILIAS[f]]


# =====================================================================================
# 2. PISTAS NUEVAS (las que modelo_v2.preparar no calcula)
#    Se agregan como columnas nuevas al pool de cada semestre.
# =====================================================================================
def bm25_matriz(cv, textos_q, textos_d, k1=1.2, b=0.75):
    """BM25 (Robertson y Zaragoza 2009), igual que en mejoras_papers.py.
    k1: saturación (repetir una palabra 100 veces no vale 100 veces más).
    b: corrección por largo (un texto enorme no gana sólo por ser enorme)."""
    D = cv.transform(textos_d).tocsr().astype(np.float32)          # cuentas por empresa
    N = D.shape[0]
    df = np.bincount(D.indices, minlength=D.shape[1])               # en cuántas empresas aparece
    idf = np.log1p((N - df + 0.5) / (df + 0.5)).astype(np.float32)
    largo = np.asarray(D.sum(axis=1)).ravel()
    prom = largo.mean() if largo.mean() > 0 else 1.0
    filas = np.repeat(np.arange(N), np.diff(D.indptr))
    tf = D.data
    D.data = (tf * (k1 + 1) / (tf + k1 * (1 - b + b * largo[filas] / prom))) * idf[D.indices]
    Q = cv.transform(textos_q).tocsr()
    Q.data[:] = 1                                                    # en la consulta: aparece o no
    return (Q.astype(np.float32) @ D.T).toarray().astype(np.float32)


def jensen_shannon(P, Q):
    """Distancia de Jensen-Shannon entre filas de P y Q (distribuciones sobre temas), en [0, 1].
    Es la que usan Pinto et al. (2019) para comparar agendas. 0 = mismos temas, 1 = nada en común.
    Si alguna de las dos filas es todo ceros (sin texto), vale 1."""
    M = 0.5 * (P + Q)
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.where(P > 0, P * np.log(P / M), 0.0).sum(axis=1)
        b = np.where(Q > 0, Q * np.log(Q / M), 0.0).sum(axis=1)
    js = np.sqrt(np.clip(0.5 * (a + b) / np.log(2), 0, 1))
    vacia = (P.sum(axis=1) == 0) | (Q.sum(axis=1) == 0)
    js[vacia] = 1.0
    return js.astype(np.float32)


def normalizar_filas(H):
    s = H.sum(axis=1, keepdims=True)
    return np.divide(H, s, out=np.zeros_like(H), where=s > 0)


def correspondencias(A, k):
    """Análisis de correspondencias (Greenacre 2007; del Pozo, Pinto et al. 2025) de la matriz
    oficinas x empresas A (contratos ganados antes del semestre, pesados como en el B0).

    Modelo nulo: si oficinas y empresas fueran independientes, la fracción esperada de
    contratos entre la oficina o y la empresa e sería r_o * c_e (masas de fila y columna).
    Los residuos estandarizados miden cuánto se apartan los datos de ese nulo:
        s_oe = (p_oe - r_o c_e) / sqrt(r_o c_e)
    La SVD de esa matriz da un espacio latente donde oficinas y empresas "afines" quedan
    cerca, AUNQUE nunca hayan trabajado juntas (si contratan / trabajan con los mismos).

    Devuelve F (coordenadas de oficinas), G (de empresas), y r, c, P para el residuo directo.
    afinidad(o, e) = F[o] · G[e]  ≈  p_oe / (r_o c_e) - 1, suavizado con k dimensiones."""
    N = A.sum()
    P = (A / N).tocsr()
    r = np.asarray(P.sum(axis=1)).ravel()
    c = np.asarray(P.sum(axis=0)).ravel()
    dr = np.divide(1.0, np.sqrt(r), out=np.zeros_like(r), where=r > 0)
    dc = np.divide(1.0, np.sqrt(c), out=np.zeros_like(c), where=c > 0)
    sr, sc = np.sqrt(r), np.sqrt(c)
    PT = P.T.tocsr()
    # S no se arma nunca como matriz densa (sería oficinas x ~100.000 empresas): sólo se
    # define cómo multiplica a un vector, y svds hace el resto.
    S = LinearOperator(P.shape, dtype=np.float64,
                       matvec=lambda x: dr * (P @ (dc * np.ravel(x))) - sr * (sc @ np.ravel(x)),
                       rmatvec=lambda y: dc * (PT @ (dr * np.ravel(y))) - sc * (sr @ np.ravel(y)))
    k = int(min(k, min(P.shape) - 2))
    if k < 1:
        return None
    try:
        U, s, Vt = svds(S, k=k, random_state=0)
    except Exception:                 # ARPACK a veces no converge (depende de la versión de scipy)
        log("   correspondencias: ARPACK no convergió, se usa una tolerancia más laxa")
        U, s, Vt = svds(S, k=k, random_state=0, tol=1e-6, maxiter=100_000)
    F = dr[:, None] * U * s           # coordenadas principales de las oficinas
    G = dc[:, None] * Vt.T            # coordenadas estándar de las empresas
    return F.astype(np.float32), G.astype(np.float32), r, c, P


def agregar_pistas(D, d, k_temas=20, k_ca=10, max_docs_nmf=40000):
    """Agrega al pool del semestre d las pistas BM25, temas NMF, correspondencias y las
    interacciones con la entropía. Todo con información anterior al inicio del semestre."""
    inicio, previo = m.semestre_inicio(d.sem)
    q, cand, pool = d.q, d.cand, d.pool
    qi, ci = pool.q.values, pool.c.values
    # mismo corpus que usa modelo_v2.preparar para el TF-IDF: licitaciones anteriores al
    # semestre y los perfiles de las empresas del semestre anterior
    corpus = pd.concat([D.lic[D.lic.t < inicio].texto, D.emp[D.emp.semestre == previo].texto_limpio])

    # ---------- BM25 ----------
    log(f"{d.sem}: BM25...")
    cv = CountVectorizer(stop_words="english", min_df=3, max_df=0.5, dtype=np.float32).fit(corpus)
    B = bm25_matriz(cv, q.texto, cand.texto_limpio)                  # licitaciones x candidatas
    mx = B.max(axis=1)
    pool["sim_bm25"] = B[qi, ci]
    pool["sim_bm25_rel"] = np.where(mx[qi] > 0, B[qi, ci] / np.maximum(mx[qi], 1e-12), 0).astype(np.float32)
    pool["log_puesto_bm25"] = np.log(m._puestos(B)[qi, ci])
    del B
    gc.collect()

    # ---------- Temas NMF (Lee y Seung 1999; Pinto et al. 2019) ----------
    # Cada texto pasa a ser una mezcla de k temas ("HVAC", "vehículos", ...).
    # La licitación y la empresa se comparan por sus mezclas con Jensen-Shannon.
    log(f"{d.sem}: temas NMF...")
    tf = TfidfVectorizer(stop_words="english", sublinear_tf=True, min_df=3, max_df=0.5,
                         max_features=20000, dtype=np.float32).fit(corpus)
    muestra = corpus.sample(min(len(corpus), max_docs_nmf), random_state=0)
    k_t = int(min(k_temas, len(tf.vocabulary_) - 1, len(muestra) - 1))
    nmf = NMF(n_components=k_t, init="nndsvda", random_state=0, max_iter=300).fit(tf.transform(muestra))
    Hq = normalizar_filas(nmf.transform(tf.transform(q.texto)).astype(np.float32))
    Hc = normalizar_filas(nmf.transform(tf.transform(cand.texto_limpio)).astype(np.float32))
    pool["js_temas"] = jensen_shannon(Hq[qi], Hc[ci])
    # entropía normalizada de la mezcla de temas de la empresa: 0 = especialista, 1 = hace de todo
    with np.errstate(divide="ignore", invalid="ignore"):
        ent_c = -np.where(Hc > 0, Hc * np.log(Hc), 0).sum(axis=1) / np.log(k_t)
    pool["entropia_temas"] = ent_c[ci].astype(np.float32)
    palabras = np.array(tf.get_feature_names_out())
    d.temas = pd.DataFrame({"semestre": d.sem, "tema": range(k_t),
                            "palabras": [" ".join(palabras[np.argsort(-w)[:10]]) for w in nmf.components_]})

    # ---------- Análisis de correspondencias en la red oficina–empresa ----------
    log(f"{d.sem}: análisis de correspondencias oficina–empresa...")
    A, idx_of = m.red_bipartita(D.prior, "office", inicio)
    col_hist = np.array([D.prior.idx.get(u, -1) for u in cand.parent_uei])  # candidata -> columna de A
    fila_of = np.array([idx_of.get(str(o), -1) for o in q.office.astype(str).str.upper()])
    o, e = fila_of[qi], col_hist[ci]
    ok = (o >= 0) & (e >= 0)
    afin = np.zeros(len(pool), np.float32)
    resid = np.zeros(len(pool), np.float32)
    ca = correspondencias(A, k_ca)
    if ca is not None and ok.any():
        F, G, r, c, P = ca
        afin[ok] = (F[o[ok]] * G[e[ok]]).sum(axis=1)
        p_oe = np.asarray(P[o[ok], e[ok]]).ravel()
        esperado = r[o[ok]] * c[e[ok]]
        resid[ok] = np.where(esperado > 0, (p_oe - esperado) / np.sqrt(np.maximum(esperado, 1e-300)), 0)
    pool["afinidad_ca"] = afin
    pool["residuo_ca"] = resid
    del A
    gc.collect()

    # ---------- Interacciones con la entropía del B0 ----------
    # entropia_b0 alta = el B0 duda (reparte su apuesta entre muchas empresas).
    # Un β positivo en ent_x_tfidf quiere decir: "cuando el B0 duda, el texto pesa más".
    ent = pool.entropia_b0.values
    pool["ent_x_p_b0"] = ent * pool.p_b0.values
    pool["ent_x_tfidf"] = ent * pool.sim_tfidf_rel.values
    pool["ent_x_hist"] = ent * pool.sim_hist_rel.values

    for v in variables_de(FAMILIAS):
        pool[v] = pool[v].astype(np.float32)
    d.n_gan = d.gan.groupby("id").size().reindex(q.id).fillna(0).values   # ganadores por licitación


# =====================================================================================
# 3. EL MODELO DE BOLTZMANN
#    Para cada licitación, cada candidata del pool es un "estado" con energía
#        E = -(β_1 f_1 + β_2 f_2 + ...)          (f = pistas estandarizadas)
#    y la probabilidad de ganar es la de Boltzmann:
#        P(empresa | licitación) = exp(-E) / Z,   Z = suma sobre las candidatas del pool.
#    Los β se eligen por máxima verosimilitud (lo más probable que hayan ganado los que
#    ganaron). En el óptimo, el promedio de cada pista bajo el modelo es igual al promedio
#    en los ganadores: es el principio de máxima entropía de Jaynes (1957).
#    l2 es un "resorte" chico que empuja los β hacia 0 (evita β enormes con pistas raras).
# =====================================================================================
class Boltzmann:
    def __init__(self, variables, l2=0.01, negativos=200, semilla=0):
        self.variables = list(variables)
        self.l2, self.negativos, self.semilla = l2, negativos, semilla

    # --- preprocesamiento: comprimir colas largas y estandarizar (media 0, desvío 1) ---
    def _ajustar_escalas(self, X):
        self.escala = np.ones(X.shape[1])
        for j, v in enumerate(self.variables):
            if v in COLAS_LARGAS:
                nz = np.abs(X[:, j][X[:, j] != 0])
                self.escala[j] = np.median(nz) if len(nz) else 1.0
        Z = self._comprimir(X)
        self.media = Z.mean(axis=0)
        self.desvio = Z.std(axis=0)
        self.desvio[self.desvio == 0] = 1.0          # pista constante: queda en 0 y no molesta

    def _comprimir(self, X):
        Z = np.asarray(X, dtype=np.float64).copy()
        for j, v in enumerate(self.variables):
            if v in COLAS_LARGAS:
                Z[:, j] = np.arcsinh(Z[:, j] / self.escala[j])
        return Z

    def _transformar(self, X):
        return (self._comprimir(X) - self.media) / self.desvio

    def fit(self, pools):
        """pools: lista de pools (uno por semestre de entrenamiento)."""
        rng = np.random.default_rng(self.semilla)
        partes = []
        for k, p in enumerate(pools):
            # sólo licitaciones con algún ganador en el pool: las otras no enseñan nada
            con_gan = p.groupby("q").ganador.transform("max").values > 0
            p = p[con_gan]
            # McFadden (1978): ganadores + una muestra al azar de las demás candidatas
            azar = rng.random(len(p))
            orden_azar = pd.Series(azar).groupby(p.q.values).rank(method="first").values
            elegir = (p.ganador.values == 1) | (orden_azar <= self.negativos)
            p = p[elegir]
            partes.append(pd.DataFrame({"g": p.q.values + 1_000_000 * k, "y": p.ganador.values,
                                        **{v: p[v].values for v in self.variables}}))
        t = pd.concat(partes, ignore_index=True).sort_values("g", kind="stable")
        Xraw = t[self.variables].values
        self._ajustar_escalas(Xraw)
        X = self._transformar(Xraw)
        g = t.g.values
        y = t.y.values.astype(np.float64)
        inicio_g = np.flatnonzero(np.r_[True, g[1:] != g[:-1]])
        G = len(inicio_g)
        de_grupo = np.repeat(np.arange(G), np.diff(np.r_[inicio_g, len(g)]))
        # si una licitación tiene varios ganadores, cada uno pesa 1/(cuántos son): la licitación
        # vale 1 en total, igual que en el MRR filtrado de evaluar.py
        y = y / np.bincount(de_grupo, weights=y)[de_grupo]
        objetivo = X.T @ y / G                       # promedio de cada pista en los ganadores

        def perdida(beta):
            eta = X @ beta                           # eta = -E
            mx = np.maximum.reduceat(eta, inicio_g)
            ex = np.exp(eta - mx[de_grupo])
            Zg = np.add.reduceat(ex, inicio_g)       # función de partición de cada licitación
            lnZ = np.log(Zg) + mx
            prob = ex / Zg[de_grupo]                 # probabilidad de Boltzmann de cada candidata
            nll = -(y @ eta - lnZ.sum()) / G + 0.5 * self.l2 * beta @ beta
            # gradiente = <f>_Boltzmann - <f>_ganadores (+ resorte)
            grad = (X.T @ prob) / G - objetivo + self.l2 * beta
            return nll, grad

        res = minimize(perdida, np.zeros(X.shape[1]), jac=True, method="L-BFGS-B",
                       options={"maxiter": 500})
        self.beta = res.x
        self.convergio = res.success
        return self

    def predict(self, Xraw):
        """-E de cada fila (más alto = más probable). Lo usa modelo_v2.ranking."""
        return self._transformar(Xraw) @ self.beta

    def tabla_beta(self):
        return (pd.DataFrame({"familia": [FAMILIA_DE[v] for v in self.variables],
                              "pista": self.variables, "beta": self.beta})
                .assign(abs_beta=lambda x: x.beta.abs())
                .sort_values("abs_beta", ascending=False).drop(columns="abs_beta"))


# =====================================================================================
# 4. MEDIR
# =====================================================================================
def rr_rapido(d, puntaje):
    """MRR filtrado por licitación (igual que evaluar.py) calculado directo sobre el pool.
    Se usa para la selección porque es mucho más rápido que armar el top-100 completo.
    Los ganadores fuera del pool valen 0 (en el ranking final quedan debajo del puesto 100)."""
    p = pd.DataFrame({"q": d.pool.q.values, "y": d.pool.ganador.values, "s": puntaje})
    p = p.sort_values(["q", "s"], ascending=[True, False], kind="stable")
    y = p.y.values
    q = p.q.values
    nuevo = np.r_[True, q[1:] != q[:-1]]
    # cuántas no-ganadoras y cuántas ganadoras tiene arriba cada fila, dentro de su licitación
    no_gan_arriba = np.cumsum(1 - y) - (1 - y)
    gan_arriba = np.cumsum(y) - y
    base_ng = np.maximum.accumulate(np.where(nuevo, no_gan_arriba, 0))
    base_g = np.maximum.accumulate(np.where(nuevo, gan_arriba, 0))
    ng, gg = no_gan_arriba - base_ng, gan_arriba - base_g
    puesto = 1 + ng + gg                               # puesto en la lista
    filtrado = 1 + ng                                  # sin contar a los otros ganadores
    rr = np.where((y == 1) & (puesto <= 100), 1.0 / filtrado, 0.0)
    suma = np.bincount(q, weights=rr, minlength=len(d.q))
    return np.divide(suma, d.n_gan, out=np.zeros(len(d.q)), where=d.n_gan > 0)


def ic_bootstrap(diferencias, n=2000, semilla=0):
    """IC95 de la media de las diferencias, remuestreando licitaciones (como evaluar.py)."""
    i = np.random.default_rng(semilla).integers(0, len(diferencias), size=(n, len(diferencias)))
    medias = diferencias[i].mean(axis=1)
    return float(np.percentile(medias, 2.5)), float(np.percentile(medias, 97.5))


def entrenar_y_medir_rapido(sems, entrenar_con, validar, familias, l2):
    mod = Boltzmann(variables_de(familias), l2=l2).fit([sems[s].pool for s in entrenar_con])
    dv = sems[validar]
    return rr_rapido(dv, mod.predict(dv.pool[mod.variables].values)), mod


# =====================================================================================
# 5. SELECCIÓN DE FAMILIAS (sólo con validación: entrenar 2022S2 + 2023S1, medir 2023S2)
#
#    Eliminación hacia atrás:
#      1) se ajusta el modelo con todas las familias activas;
#      2) se prueba sacar cada familia, de a una, y se mide cuánto baja el MRR;
#      3) se saca la que menos aporta, si su aporte es menor que el umbral;
#      4) se repite hasta que todas las que quedan aportan más que el umbral.
#    El paso 2 de la primera vuelta es la tabla de ablación completa.
# =====================================================================================
def seleccionar(sems, V, umbral, l2):
    entrenar_con, validar = ["2022S2", "2023S1"], "2023S2"
    activas = list(FAMILIAS)
    rr_actual, _ = entrenar_y_medir_rapido(sems, entrenar_con, validar, activas, l2)
    log(f"validación con todas las familias: MRR {rr_actual.mean():.4f}")
    pasos, ablacion = [], None
    while len(activas) > 1:
        filas = []
        for f in activas:
            rr_sin, _ = entrenar_y_medir_rapido(sems, entrenar_con, validar,
                                               [x for x in activas if x != f], l2)
            dif = rr_actual - rr_sin              # > 0: la familia aporta (sacarla baja el MRR)
            lo, hi = ic_bootstrap(dif)
            filas.append({"familia": f, "MRR con": rr_actual.mean(), "MRR sin": rr_sin.mean(),
                          "aporte": dif.mean(), "aporte IC95 bajo": lo, "aporte IC95 alto": hi,
                          "_rr": rr_sin})
            log(f"   sin {f:<18} MRR {rr_sin.mean():.4f}   aporte {dif.mean():+.4f} [{lo:+.4f}; {hi:+.4f}]")
        tabla = pd.DataFrame(filas).sort_values("aporte")
        if ablacion is None:                      # primera vuelta = ablación con todo
            ablacion = tabla.drop(columns="_rr").sort_values("aporte", ascending=False)
            ablacion.to_csv(f"ablacion_{V}.csv", index=False)
        peor = tabla.iloc[0]
        if peor.aporte >= umbral:
            log(f"todas las familias que quedan aportan al menos {umbral}: fin de la selección")
            break
        activas.remove(peor.familia)
        rr_actual = peor._rr
        pasos.append({"paso": len(pasos) + 1, "se saca": peor.familia, "aporte": peor.aporte,
                      "MRR validación después": rr_actual.mean(), "quedan": ", ".join(activas)})
        log(f"-> se saca {peor.familia} (aporte {peor.aporte:+.4f}); MRR {rr_actual.mean():.4f}")
    pd.DataFrame(pasos).to_csv(f"seleccion_pasos_{V}.csv", index=False)
    return activas, ablacion


# =====================================================================================
# 6. PRUEBA FINAL (2024S1, mirada una sola vez)
# =====================================================================================
def por_licitacion(rk, d):
    return evaluar.por_licitacion(rk, d.gan).MRR.reindex(sorted(d.q.id)).fillna(0).values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vertical", default="j")
    ap.add_argument("--K", type=int, default=300)
    ap.add_argument("--umbral", type=float, default=0.002)
    ap.add_argument("--l2", type=float, default=0.01)
    ap.add_argument("--sin-lambdamart", action="store_true")
    a = ap.parse_args()
    V = a.vertical

    # ---- datos, pool y pistas de modelo_v2 (con física), más las pistas nuevas ----
    D = m.Datos(V)
    sems = {}
    for s in SEMESTRES:
        sems[s] = m.preparar(D, s, K=a.K, fisica=True)
        agregar_pistas(D, sems[s])
    pd.concat([sems[s].temas for s in SEMESTRES]).to_csv(f"temas_{V}.csv", index=False)

    # ---- selección con validación ----
    log("SELECCIÓN DE FAMILIAS (entrenar 2022S2 + 2023S1, validar 2023S2)")
    elegidas, ablacion = seleccionar(sems, V, a.umbral, a.l2)
    log(f"familias elegidas: {', '.join(elegidas)}")

    # ---- prueba final: entrenar 2022S2-2023S2, medir 2024S1 ----
    log("PRUEBA FINAL (entrenar 2022S2 + 2023S1 + 2023S2, medir 2024S1)")
    tr, te = ["2022S2", "2023S1", "2023S2"], "2024S1"
    dt = sems[te]
    modelos, rks = {}, {}

    completo = Boltzmann(variables_de(FAMILIAS), l2=a.l2).fit([sems[s].pool for s in tr])
    elegido = Boltzmann(variables_de(elegidas), l2=a.l2).fit([sems[s].pool for s in tr])
    modelos["Boltzmann completo"] = (completo, completo.variables)
    modelos["Boltzmann seleccionado"] = (elegido, elegido.variables)

    if not a.sin_lambdamart:
        try:
            NF = m.VARIABLES_BASE + m.VARIABLES_NUEVAS + m.VARIABLES_FISICA
            modelos["LambdaMART v2 + física"] = (m.entrenar(sems, tr, a.K, NF), NF)
            modelos["LambdaMART con las pistas elegidas"] = (
                m.entrenar(sems, tr, a.K, elegido.variables), elegido.variables)
        except ImportError:
            log("lightgbm no está instalado: se saltea la comparación con LambdaMART")

    filas = []
    for nombre, (mod, var) in modelos.items():
        rks[nombre] = m.ranking(mod, dt, a.K, var)
        filas.append({"modelo": nombre, "n pistas": len(var), **m.medir(D, dt, rks[nombre])})
    res = pd.DataFrame(filas)
    res.to_csv(f"resultados_boltzmann_{V}.csv", index=False)

    # ---- diferencias entre modelos, con IC95 pareado ----
    rr = {k: por_licitacion(rk, dt) for k, rk in rks.items()}
    comps = []
    pares = [("Boltzmann seleccionado", "Boltzmann completo"),
             ("Boltzmann seleccionado", "LambdaMART v2 + física"),
             ("Boltzmann seleccionado", "LambdaMART con las pistas elegidas")]
    for a_, b_ in pares:
        if a_ in rr and b_ in rr:
            dif = rr[a_] - rr[b_]
            lo, hi = ic_bootstrap(dif)
            comps.append({"comparación": f"{a_} − {b_}", "Δ MRR": dif.mean(),
                          "IC95 bajo": lo, "IC95 alto": hi,
                          "¿distinto de 0?": "sí" if (lo > 0 or hi < 0) else "no"})
    comps = pd.DataFrame(comps)
    comps.to_csv(f"comparaciones_{V}.csv", index=False)

    elegido.tabla_beta().to_csv(f"beta_seleccionado_{V}.csv", index=False)
    completo.tabla_beta().to_csv(f"beta_completo_{V}.csv", index=False)
    rks["Boltzmann seleccionado"][["id", "parent_uei", "puesto"]].to_csv(f"ranking_boltzmann_{V}.csv", index=False)

    # ---- resumen en pantalla ----
    with pd.option_context("display.width", 200, "display.max_columns", 20,
                           "display.float_format", "{:.4f}".format):
        print("\n=== ABLACIÓN en validación (2023S2): cuánto baja el MRR al sacar cada familia ===")
        print(ablacion.to_string(index=False))
        print(f"\n=== Familias elegidas ({len(elegidas)}): {', '.join(elegidas)}")
        print(f"    Familias sacadas: {', '.join(f for f in FAMILIAS if f not in elegidas)}")
        print("\n=== PRUEBA 2024S1 ===")
        print(res.to_string(index=False))
        if len(comps):
            print("\n=== Diferencias (IC95 pareado por licitación) ===")
            print(comps.to_string(index=False))
        print("\n=== β del Boltzmann seleccionado (pistas estandarizadas) ===")
        print(elegido.tabla_beta().to_string(index=False))
        if not (completo.convergio and elegido.convergio):
            print("\nAviso: algún ajuste de Boltzmann no terminó de converger (ver --l2).")


if __name__ == "__main__":
    main()
