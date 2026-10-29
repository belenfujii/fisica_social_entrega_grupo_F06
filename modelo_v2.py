"""Datos, pool de candidatas y pistas de memoria y texto (similitud con cada contrato previo,
sólo título, recencia). Lo usan modelo_final.py y los experimentos.

(El script de este archivo entrena además un LambdaMART de comparación.)

Uso como módulo:
    import modelo_v2 as m
    D = m.Datos("j")
    sems = {s: m.preparar(D, s) for s in ["2022S2", "2023S1", "2023S2", "2024S1"]}
    res = m.experimento(D, sems, entrenar=["2022S2", "2023S1", "2023S2"], probar="2024S1")

Uso como script (entrena con 2022S2-2023S2 y mide en 2024S1):
    uv run modelo_v2.py --vertical j     (necesita lightgbm: uv sync --group dev)
"""
from __future__ import annotations

import argparse
import hashlib
import re
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

import b0
import evaluar

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:6.0f} s] {msg}", flush=True)


# ---------- Texto ----------
# Muchas descripciones del historial empiezan con códigos internos tipo "IGF::OT::IGF".
_BASURA = re.compile(r"IGF::\w+::IGF|\bIGF\b|::", re.I)


def limpiar(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.replace(_BASURA, " ", regex=True)


def semestre_inicio(sem):
    anio, mitad = int(sem[:4]), sem[-1]
    inicio = pd.Timestamp(f"{anio}-01-01" if mitad == "1" else f"{anio}-07-01")
    previo = f"{anio - 1}S2" if mitad == "1" else f"{anio}S1"
    return inicio, previo


# ---------- B0 con niveles separados y jerarquía de oficinas ----------
NIVELES_EXT = [("office",), ("psc", "sub"), ("sub",), ("psc", "dep"), ("naics", "office")]
NOMBRES_NIVELES = (["psc+office", "psc", "naics", "global", "psc+sa"]
                   + ["office", "psc+sub", "sub", "psc+dep", "naics+office"])
VACIOS = {"?", "NAN", "NONE", "", "<NA>"}


class B0Niveles(b0.B0):
    def __init__(self, historial, oficinas):
        super().__init__(historial)
        o = oficinas.assign(codigo=oficinas.codigo.astype(str).str.upper()).drop_duplicates("codigo")
        self.sub_de = o.set_index("codigo").subagencia.astype(str).str.upper().to_dict()
        self.dep_de = o.set_index("codigo").departamento.astype(str).str.upper().to_dict()
        self.h["sub"] = self.h.office.map(self.sub_de).fillna("?")
        self.h["dep"] = self.h.office.map(self.dep_de).fillna("?")
        self.tablas_ext = [self._indexar(cols) for cols in NIVELES_EXT]

    def niveles(self, lic):
        ts = np.datetime64(pd.Timestamp(lic.t), "D").astype(np.int64)
        office = str(lic.office).upper()
        ctx = {"psc": str(lic.psc).upper(), "office": office, "naics": str(lic.naics),
               "sa": b0.set_aside(lic.set_aside),
               "sub": self.sub_de.get(office, "?"), "dep": self.dep_de.get(office, "?")}
        out = np.zeros((len(NOMBRES_NIVELES), len(self.empresas)), dtype=np.float32)
        p = np.zeros(len(self.empresas))
        todos = list(zip(b0.NIVELES, self.tablas)) + list(zip(NIVELES_EXT, self.tablas_ext))
        for j, (cols, tabla) in enumerate(todos):
            if j >= len(b0.NIVELES) and any(ctx[c] in VACIOS for c in cols):
                continue
            e = tabla.get(tuple(ctx[c] for c in cols))
            if e is None:
                continue
            t_arr, fi, w = e
            n = np.searchsorted(t_arr, ts, side="left")       # sólo contratos anteriores a t
            if n == 0:
                continue
            v = np.bincount(fi[:n], weights=w[:n], minlength=len(self.empresas))
            out[j] = v / v.sum()
            if j < len(b0.NIVELES):
                p += b0.PESOS[j] * v / v.sum()
        return out, p



# ---------- Señales de física estadística (redes oficina–empresa y PSC–empresa) ----------
def _pot(k, e):
    out = np.zeros_like(k, dtype=np.float64)
    m = k > 0
    out[m] = k[m] ** e
    return out


def red_bipartita(prior, columna, t0):
    """Matriz (códigos x empresas) con los contratos anteriores a t0, pesados como en el B0."""
    from scipy import sparse
    h = prior.h
    t0d = np.datetime64(pd.Timestamp(t0), "D").astype(np.int64)
    col = h[columna].astype(str)
    m = (h.ts.values < t0d) & ~col.str.upper().isin(VACIOS).values
    codigos, filas = np.unique(col.values[m], return_inverse=True)
    w = np.exp((h.ts.values[m] - t0d) / (b0.TAU_ANIOS * 365.25))
    A = sparse.csr_matrix((w, (filas, h.fi.values[m])), shape=(len(codigos), len(prior.empresas)))
    A.sum_duplicates()
    return A, {c: i for i, c in enumerate(codigos)}


def difusion(A, filas_q, col_hist, lam):
    """Difusión híbrida (Zhou et al. 2010): código -> empresas -> otros códigos -> empresas.
    lam = 1 es difusión de masa (ProbS), lam = 0 conducción de calor (HeatS)."""
    from scipy import sparse
    k_o = np.asarray(A.sum(axis=1)).ravel()
    k_c = np.asarray(A.sum(axis=0)).ravel()
    ok = filas_q >= 0
    E = sparse.csr_matrix((np.ones(ok.sum()), (np.flatnonzero(ok), filas_q[ok])), shape=(len(filas_q), A.shape[0]))
    G = (E @ A) @ sparse.diags(_pot(k_c, -lam))
    H = ((G @ A.T) @ sparse.diags(_pot(k_o, -1.0))).toarray().astype(np.float32)
    validas = col_hist >= 0
    cols = np.where(validas, col_hist, 0)
    Ac = A.tocsc()[:, cols] @ sparse.diags(validas * _pot(k_c[cols], -(1 - lam)))
    return np.asarray((Ac.T.astype(np.float32) @ H.T).T, dtype=np.float32)


def modelo_nulo(A, iteraciones=3000):
    """Bipartite Configuration Model: p = x_o y_c / (1 + x_o y_c), forma de Fermi-Dirac,
    con los grados esperados iguales a los observados."""
    B = (A > 0).astype(np.float64)
    k_o = np.asarray(B.sum(axis=1)).ravel()
    k_c = np.asarray(B.sum(axis=0)).ravel()
    du, iu, mu = np.unique(k_o, return_inverse=True, return_counts=True)
    dv, iv, nv = np.unique(k_c, return_inverse=True, return_counts=True)
    L = max(k_o.sum(), 1.0)
    x, y = du / np.sqrt(L), dv / np.sqrt(L)
    for _ in range(iteraciones):
        x = 0.5 * x + 0.5 * du / np.maximum((nv * y / (1 + np.outer(x, y))).sum(axis=1), 1e-300)
        y = 0.5 * y + 0.5 * dv / np.maximum((mu[:, None] * x[:, None] / (1 + np.outer(x, y))).sum(axis=0), 1e-300)
    return x[iu], y[iv], B.tocsc()


def entropia(p):
    s = p.sum()
    if s <= 0:
        return 0.0
    q = p[p > 0] / s
    return float(-(q * np.log(q)).sum())


# ---------- Datos de una vertical ----------
class Datos:
    def __init__(self, vertical, carpeta="data"):
        self.vertical = vertical
        self.lic = pd.read_parquet(f"{carpeta}/{vertical}/licitaciones.parquet")
        self.gan = pd.read_parquet(f"{carpeta}/{vertical}/ganadores.parquet")
        self.emp = pd.read_parquet(f"{carpeta}/{vertical}/empresas.parquet")
        self.hist = pd.read_parquet(f"{carpeta}/{vertical}/historial.parquet")
        psc = pd.read_parquet(f"{carpeta}/psc.parquet")
        naics = pd.read_parquet(f"{carpeta}/naics.parquet")
        self.ofic = pd.read_parquet(f"{carpeta}/office.parquet")
        self.lic["t"] = pd.to_datetime(self.lic.t)
        self.lic["semestre"] = self.lic.t.map(b0.semestre)
        self.hist["fecha"] = pd.to_datetime(self.hist.fecha)
        self.hist["desc_limpia"] = limpiar(self.hist.descripcion)
        self.emp["texto_limpio"] = limpiar(self.emp.texto_awards)
        dpsc = psc.assign(codigo=psc.codigo.str.upper()).drop_duplicates("codigo").set_index("codigo").descripcion
        dnaics = naics.drop_duplicates("codigo").set_index("codigo").descripcion
        self.lic["texto"] = (self.lic.title.fillna("") + " " + self.lic.description.fillna("") + " "
                             + self.lic.psc.astype(str).str.upper().map(dpsc).fillna("") + " "
                             + self.lic.naics.astype(str).map(dnaics).fillna(""))
        self.lic["titulo"] = (self.lic.title.fillna("") + " "
                              + self.lic.psc.astype(str).str.upper().map(dpsc).fillna(""))
        log(f"{vertical}: indexando el historial...")
        self.prior = B0Niveles(self.hist, self.ofic)


def _azar(id_, n):
    semilla = int(hashlib.sha256(str(id_).encode()).hexdigest()[:16], 16)
    return np.random.default_rng(semilla).random(n)


def _puestos(S):
    R = np.empty(S.shape, dtype=np.float32)
    for i in range(S.shape[0]):
        orden = np.argsort(-S[i], kind="stable")
        R[i, orden] = np.arange(1, S.shape[1] + 1)
    return R


def _sim_historial(tfidf, Q, cand, hist, inicio, bloque):
    """Para cada (licitación, candidata): la mayor similitud entre la licitación y alguno de los
    contratos que la candidata ganó antes del inicio del semestre."""
    pos = {u: i for i, u in enumerate(cand.parent_uei)}
    h = hist[(hist.fecha < inicio) & hist.parent_uei.isin(pos)]
    h = h.assign(c=h.parent_uei.map(pos)).drop_duplicates(["c", "desc_limpia"]).sort_values("c")
    H = tfidf.transform(h.desc_limpia.values).T.tocsc()
    c = h.c.values
    cortes = np.flatnonzero(np.r_[True, c[1:] != c[:-1]])
    empresas = c[cortes]
    out = np.zeros((Q.shape[0], len(cand)), dtype=np.float32)
    for a in range(0, Q.shape[0], bloque):
        M = (Q[a:a + bloque] @ H).toarray()
        out[a:a + bloque, empresas] = np.maximum.reduceat(M, cortes, axis=1)
    return out


def preparar(D: Datos, sem: str, K: int = 1000, bloque: int = 48, fisica: bool = False):
    """Calcula todas las señales de un semestre y arma el pool (las K mejores candidatas según
    cada fuente). Sólo usa lo anterior al inicio del semestre, y para el B0 lo anterior a t."""
    inicio, previo = semestre_inicio(sem)
    lic, emp = D.lic, D.emp
    q = lic[lic.semestre == sem].reset_index(drop=True)
    cand = emp[emp.semestre == sem].reset_index(drop=True)
    g = D.gan[D.gan.id.isin(q.id)]
    pos = {u: i for i, u in enumerate(cand.parent_uei)}
    por_id = g.groupby("id").parent_uei.apply(list).to_dict()
    gan_idx = [np.array([pos[u] for u in por_id.get(i, []) if u in pos], dtype=int) for i in q.id]

    # Vocabulario con lo anterior al semestre
    corpus = pd.concat([lic[lic.t < inicio].texto, emp[emp.semestre == previo].texto_limpio])
    tfidf = TfidfVectorizer(stop_words="english", sublinear_tf=True, ngram_range=(1, 2),
                            min_df=3, max_df=0.5, dtype=np.float32).fit(corpus)
    Qt = tfidf.transform(q.texto)
    Dt = tfidf.transform(cand.texto_limpio)
    S = {"tfidf": (Qt @ Dt.T).toarray().astype(np.float32),
         "titulo": (tfidf.transform(q.titulo) @ Dt.T).toarray().astype(np.float32)}
    log(f"{sem}: similitud con cada contrato previo...")
    S["hist"] = _sim_historial(tfidf, Qt, cand, D.hist, inicio, bloque)
    R = {k: _puestos(v) for k, v in S.items()}

    col_hist = np.array([D.prior.idx.get(u, -1) for u in cand.parent_uei])
    if fisica:
        log(f"{sem}: difusión en redes y modelo nulo...")
        A_of, idx_of = red_bipartita(D.prior, "office", inicio)
        A_psc, idx_psc = red_bipartita(D.prior, "psc", inicio)
        fila_of = np.array([idx_of.get(str(o), -1) for o in q.office.astype(str).str.upper()])
        fila_psc = np.array([idx_psc.get(str(c), -1) for c in q.psc.astype(str).str.upper()])
        DIF = {"dif_masa_oficina": difusion(A_of, fila_of, col_hist, 1.0),
               "dif_calor_oficina": difusion(A_of, fila_of, col_hist, 0.0),
               "dif_mixta_oficina": difusion(A_of, fila_of, col_hist, 0.5),
               "dif_masa_psc": difusion(A_psc, fila_psc, col_hist, 1.0)}
        R_dif = _puestos(DIF["dif_masa_oficina"])
        R_dif[DIF["dif_masa_oficina"] <= 0] = np.inf
        x_of, y_emp, B_of = modelo_nulo(A_of)
        y_cand = np.where(col_hist >= 0, y_emp[np.maximum(col_hist, 0)], 0.0)

    # Recencia y volumen de cada candidata (hasta el inicio del semestre)
    h = D.hist[(D.hist.fecha < inicio) & D.hist.parent_uei.isin(pos)]
    ultimo = h.groupby("parent_uei").fecha.max().reindex(cand.parent_uei)
    dias_ultimo = ((inicio - ultimo).dt.days.fillna(9999).values).astype(np.float32)
    rec = h[h.fecha >= inicio - pd.Timedelta(days=730)].groupby("parent_uei").size()
    n_rec = rec.reindex(cand.parent_uei).fillna(0).values.astype(np.float32)
    n_awards = cand.n_awards.fillna(0).values.astype(np.float32)

    log(f"{sem}: B0 por niveles para {len(q)} licitaciones...")
    # Pista extra opcional (por ejemplo pista_nombre.PistaNombre): una función que, para cada
    # licitación, devuelve (puestos por candidata, {columna: valor por candidata}). Sus candidatas
    # también entran al pool. Si D no la tiene, todo queda igual que antes.
    extra = getattr(D, "pista_extra", None)
    P = np.zeros((len(q), len(cand)))
    filas = []
    for i, f in enumerate(q.itertuples()):
        L, p = D.prior.niveles(f)
        L, p = L[:, col_hist], p[col_hist]
        L[:, col_hist < 0], p[col_hist < 0] = 0, 0
        P[i] = p
        rb = np.full(len(cand), np.inf, dtype=np.float32)
        orden = np.argsort(-p, kind="stable")
        rb[orden] = np.arange(1, len(cand) + 1)
        rb[p <= 0] = np.inf
        fuentes = {"b0": rb, **{k: R[k][i] for k in R}}
        if fisica:
            fuentes["dif"] = R_dif[i]
        ex = extra(f, cand) if extra is not None else None
        if ex is not None:
            fuentes["extra"] = ex[0]
        mejor = np.minimum.reduce(list(fuentes.values()))
        idx = np.flatnonzero(mejor <= K)
        dias_t = dias_ultimo[idx] + (f.t - inicio).days
        fila = {"q": np.full(len(idx), i, dtype=np.int32), "c": idx.astype(np.int32),
                "p_b0": p[idx].astype(np.float32), "p_b0_max_lic": np.full(len(idx), p.max(), np.float32),
                "log_n_awards": np.log1p(n_awards[idx]), "log_n_rec": np.log1p(n_rec[idx]),
                "log_dias_ultimo": np.log1p(dias_t), "puesto_min": mejor[idx]}
        for k, v in fuentes.items():
            fila[f"log_puesto_{k}"] = np.log(np.minimum(v[idx], 1e6))
        for k, v in S.items():
            mx = v[i].max()
            fila[f"sim_{k}"] = v[i, idx]
            fila[f"sim_{k}_rel"] = v[i, idx] / mx if mx > 0 else np.zeros(len(idx), np.float32)
        for j, nom in enumerate(NOMBRES_NIVELES):
            fila[f"nivel_{nom}"] = L[j, idx]
        if fisica:
            for k, M in DIF.items():
                fila[k] = M[i, idx]
            o = fila_of[i]
            if o >= 0:
                xy = x_of[o] * y_cand[idx]
                p_nulo = xy / (1 + xy)
                a = np.asarray(B_of[o, np.maximum(col_hist[idx], 0)].todense()).ravel() * (col_hist[idx] >= 0)
            else:
                p_nulo, a = np.zeros(len(idx)), np.zeros(len(idx))
            fila["p_nulo_fd"] = p_nulo.astype(np.float32)
            fila["exceso_fd"] = (a - p_nulo).astype(np.float32)
            fila["entropia_b0"] = np.full(len(idx), entropia(p), np.float32)
        if ex is not None:
            for k, v in ex[1].items():
                fila[k] = v[idx].astype(np.float32)
        fila["ganador"] = np.isin(idx, gan_idx[i]).astype(np.int8)
        filas.append(pd.DataFrame(fila))
    pool = pd.concat(filas, ignore_index=True)

    rk_b0 = _top100(P, q, cand)
    x0 = evaluar.por_licitacion(rk_b0, g)
    # se guarda sólo lo necesario: las matrices completas de similitud ocupan mucha memoria
    d = SimpleNamespace(sem=sem, q=q, cand=cand, gan=g, P=P.astype(np.float32), pool=pool,
                        gan_idx=gan_idx, rk_b0=rk_b0)
    del S, R
    if fisica:
        del DIF, R_dif
    cub = pool.groupby("q").ganador.max().reindex(range(len(q)), fill_value=0).mean()
    log(f"{sem}: {len(q)} licitaciones, pool de {len(pool):,} filas, "
        f"contiene un ganador en el {cub:.0%} (MRR B0 {x0.MRR.mean():.3f})")
    return d


def _top100(scores, q, cand):
    filas = []
    for i, id_ in enumerate(q.id):
        orden = np.lexsort((_azar(id_, scores.shape[1]), -scores[i]))[:100]
        filas.append(pd.DataFrame({"id": id_, "parent_uei": cand.parent_uei.values[orden],
                                   "puesto": np.arange(1, len(orden) + 1)}))
    return pd.concat(filas, ignore_index=True)


def cobertura(d, K):
    """Fracción de licitaciones cuyo pool de tamaño K contiene algún ganador."""
    p = d.pool[d.pool.puesto_min <= K]
    return p.groupby("q").ganador.max().reindex(range(len(d.q)), fill_value=0).mean()


VARIABLES_BASE = (["p_b0", "p_b0_max_lic", "log_n_awards", "log_puesto_b0", "log_puesto_tfidf",
                   "sim_tfidf", "sim_tfidf_rel"] + [f"nivel_{n}" for n in NOMBRES_NIVELES])
VARIABLES_FISICA = ["dif_masa_oficina", "dif_calor_oficina", "dif_mixta_oficina", "dif_masa_psc",
                    "p_nulo_fd", "exceso_fd", "entropia_b0", "log_puesto_dif"]
VARIABLES_NUEVAS = ["sim_titulo", "sim_titulo_rel", "log_puesto_titulo", "sim_hist", "sim_hist_rel",
                    "log_puesto_hist", "log_n_rec", "log_dias_ultimo"]


def entrenar(sems, nombres, K, variables, n_estimators=600, learning_rate=0.03, num_leaves=31):
    import lightgbm as lgb
    partes = []
    for k, s in enumerate(nombres):
        p = sems[s].pool
        p = p[p.puesto_min <= K]
        # sólo licitaciones con algún ganador en el pool: las otras no enseñan nada
        ok = p.groupby("q").ganador.transform("max") > 0
        partes.append(p[ok].assign(g=p.q[ok] + 100000 * k))
    tr = pd.concat(partes, ignore_index=True).sort_values("g", kind="stable")
    modelo = lgb.LGBMRanker(objective="lambdarank", n_estimators=n_estimators,
                            learning_rate=learning_rate, num_leaves=num_leaves,
                            min_child_samples=50, subsample=0.8, subsample_freq=1,
                            colsample_bytree=0.8, verbose=-1, random_state=0)
    modelo.fit(tr[variables].values, tr.ganador.values, group=tr.groupby("g", sort=False).size().values)
    return modelo


def ranking(modelo, d, K, variables):
    p = d.pool[d.pool.puesto_min <= K]
    s = modelo.predict(p[variables].values)
    M = np.full(d.P.shape, -1e9, dtype=np.float64)
    M[p.q.values, p.c.values] = s
    # las que no entran en el pool quedan ordenadas por el B0, debajo de todas las del pool
    M = np.where(M > -1e9, M, -1e9 + d.P)
    return _top100(M, d.q, d.cand)


def medir(D, d, rk):
    r = evaluar.evaluar(rk, D.lic, D.emp, d.gan, rk_b0=d.rk_b0).set_index("conjunto")
    return {"MRR": r.loc["todas", "MRR"], "MRR S_inc": r.loc["S_inc (B0 top-10)", "MRR"],
            "MRR S_new": r.loc["S_new (el resto)", "MRR"], "B0 MRR": r.loc["todas", "B0 MRR"],
            "Δ MRR": r.loc["todas", "Δ MRR"], "Δ IC95": r.loc["todas", "Δ MRR IC95"],
            "Hit@5": r.loc["todas", "Hit@5"]}


def experimento(D, sems, entrenar_con, probar, K=1000, variables=None, **kw):
    variables = variables or VARIABLES_BASE + VARIABLES_NUEVAS
    modelo = entrenar(sems, entrenar_con, K, variables, **kw)
    res = medir(D, sems[probar], ranking(modelo, sems[probar], K, variables))
    return res, modelo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--vertical", default="j")
    ap.add_argument("--K", type=int, default=1000)
    a = ap.parse_args()
    D = Datos(a.vertical)
    sems = {s: preparar(D, s, K=a.K) for s in ["2022S2", "2023S1", "2023S2", "2024S1"]}
    res, _ = experimento(D, sems, ["2022S2", "2023S1", "2023S2"], "2024S1", K=a.K)
    print(pd.Series(res))
