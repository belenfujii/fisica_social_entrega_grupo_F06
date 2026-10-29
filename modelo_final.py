# =====================================================================================
# modelo_final.py — Investigathon 2026, Y-Hat: ¿quién gana la licitación?
#
# EL MODELO: una distribución de Boltzmann sobre las empresas candidatas de cada licitación
#
#     P(gana e | licitación q) = exp(-E(q,e)) / Z(q),     Z(q) = Σ_e' exp(-E(q,e'))
#     -E(q,e) = Σ_k β_k f_k(q,e)
#
# Las f_k son "pistas" (memoria del B0, parecido de texto TF-IDF/NMF, nombre en el aviso,
# red con modelo nulo de Fermi-Dirac...) y los β se ajustan por máxima verosimilitud con Z
# exacta sobre las candidatas (logit condicional; McFadden 1974).
#
# DOS MODOS
#   --modo validar  : entrena con 2022S2–2023S2 y mide en 2024S1 (como si fuera desconocido).
#                     Arma la "ESCALERA DE ENERGÍA" (se agrega un término por vez y se mide
#                     cuánto suma), los MODELOS NULOS de cada término, la información en nats,
#                     S_inc / S_new con IC95 y la tabla de β. Es lo que va al informe y al póster.
#   --modo entregar : entrena UN modelo global (el del informe: Boltzmann con las 44 pistas y un
#                     tronco común elegido por lasso, ver tronco.py) con todos los semestres con
#                     ganadores de TODAS las verticales, y rankea las licitaciones nuevas. Escribe
#                     el CSV de la entrega. No usa la vertical: sirve igual para una vertical nueva.
#                     Con --l2 usa el Boltzmann con L2 (comparación); con --por-vertical entrena
#                     sólo con la vertical pedida.
#
# USO
#   uv run modelo_final.py --vertical j --modo validar
#   uv run modelo_final.py --vertical todas --modo entregar --nuevas nuevas/
#          (espera nuevas/<vertical>/licitaciones.parquet y empresas.parquet; escribe <vertical>.csv)
#   uv run modelo_final.py --vertical todas --modo entregar --simular 2024S1   (ensayo con 2024S1)
#
# Necesita en la carpeta: modelo_v2.py (con el gancho pista_extra), boltzmann_seleccion.py,
# pista_nombre.py, tronco.py, b0.py, evaluar.py, y la carpeta data/ del repo.
# =====================================================================================
import argparse
import os

import numpy as np
import pandas as pd

import b0
import boltzmann_seleccion as bs
import modelo_v2 as m
from pista_nombre import VARIABLES_NOMBRE, PistaNombre
from tronco import Tronco

log = m.log
VENTANAS = [(["2022S2"], "2023S1"), (["2022S2", "2023S1"], "2023S2")]
L2S = [0.001, 0.01, 0.1]

# ---- LA ESCALERA DE ENERGÍA: cada término es una idea física, con pocas pistas ----
TERMINOS = {
    "memoria (B0)": ["p_b0"],
    "texto (TF-IDF y temas NMF)": ["log_puesto_tfidf", "sim_hist", "js_temas"],
    "nombre (sorpresa de Shannon)": ["nombre_sorpresa"],
    "red (exceso sobre el nulo de Fermi-Dirac)": ["exceso_fd"],
}


# =====================================================================================
# Datos
# =====================================================================================
def cargar(V, licitaciones=None, candidatas=None, historial=None, sem_test=None):
    """Lee los datos de la vertical y, si se pasan, les agrega las licitaciones y candidatas
    nuevas (el conjunto de prueba de los organizadores) ANTES de procesarlos, así reciben
    exactamente el mismo tratamiento que el resto."""
    original = pd.read_parquet

    def leer(ruta, *a, **k):
        df = original(ruta, *a, **k)
        r = str(ruta).replace("\\", "/")
        if licitaciones and r.endswith(f"{V}/licitaciones.parquet"):
            df = pd.concat([df, original(licitaciones)], ignore_index=True).drop_duplicates("id")
        if candidatas and r.endswith(f"{V}/empresas.parquet"):
            nuevas = original(candidatas)
            if "semestre" not in nuevas.columns:     # test: una sola foto de empresas, sin semestre
                nuevas = nuevas.assign(semestre=sem_test)
            df = pd.concat([df, nuevas], ignore_index=True).drop_duplicates(["parent_uei", "semestre"])
        if historial and r.endswith(f"{V}/historial.parquet"):       # por si entregan un historial nuevo
            df = pd.concat([df, original(historial)], ignore_index=True).drop_duplicates()
        return df

    pd.read_parquet = leer
    try:
        D = m.Datos(V)
    finally:
        pd.read_parquet = original
    if sem_test and licitaciones:
        # test: toda la información disponible es la del corte (inicio de sem_test). Todas las
        # licitaciones del test se tratan como de ese semestre: pistas con lo anterior al corte y
        # candidatas = la foto de empresas. El B0 sigue usando la fecha real de cada licitación.
        nuevas_ids = set(original(licitaciones).id)
        D.lic["semestre"] = D.lic.semestre.mask(D.lic.id.isin(nuevas_ids), sem_test)
    D.pista_extra = PistaNombre(D)
    return D


def preparar(D, semestres, K):
    sems = {}
    for s in semestres:
        d = m.preparar(D, s, K=K, fisica=True)
        bs.agregar_pistas(D, d)
        sems[s] = d
    bs.FAMILIAS["nombre"] = VARIABLES_NOMBRE
    bs.FAMILIA_DE.update({v: "nombre" for v in VARIABLES_NOMBRE})
    return sems


def todas_las_pistas():
    return bs.variables_de(bs.FAMILIAS)


# =====================================================================================
# Herramientas
# =====================================================================================
def ajustar(sems, tr, vv, l2):
    return bs.Boltzmann(vv, l2=l2).fit([sems[s].pool for s in tr])


def elegir_l2(sems, vv):
    """L2 con las dos ventanas de validación (nunca mira 2024S1)."""
    def media(l2):
        return np.mean([bs.rr_rapido(sems[va], ajustar(sems, tr, vv, l2).predict(sems[va].pool[vv].values)).mean()
                        for tr, va in VENTANAS])
    return max(L2S, key=media)


def log_p_ganador(mod, d, vv):
    """ln P(ganador) de cada licitación con ganador en el pool (promedio sobre sus ganadores)."""
    p = d.pool
    s = mod.predict(p[vv].values)
    q = p.q.values
    mx = pd.Series(s).groupby(q).transform("max").values
    lz = np.log(pd.Series(np.exp(s - mx)).groupby(q).transform("sum").values) + mx
    lp = s - lz
    g = p.ganador.values == 1
    return pd.Series(lp[g]).groupby(q[g]).mean()


def barajar(pools, cols, semilla):
    """MODELO NULO de un término: dentro de cada licitación, sus pistas se reasignan al azar
    entre las candidatas. La distribución de valores queda igual, pero se rompe la relación
    entre la empresa y su valor. Si el término aporta algo real, su ganancia desaparece."""
    rng = np.random.default_rng(semilla)
    out = []
    for p in pools:
        p = p.copy()
        idx = np.arange(len(p))
        nuevo = np.empty(len(p), dtype=int)
        # las filas están ordenadas por q: se permuta cada bloque
        inicio = np.flatnonzero(np.r_[True, p.q.values[1:] != p.q.values[:-1]])
        fin = np.r_[inicio[1:], len(p)]
        for a, b in zip(inicio, fin):
            nuevo[a:b] = idx[a:b][rng.permutation(b - a)]
        p[cols] = p[cols].values[nuevo]
        out.append(p)
    return out


def barajar_ganadores(pools, semilla):
    """MODELO NULO global: el 'ganador' de cada licitación de entrenamiento pasa a ser una
    candidata al azar de su pool. Con esto el modelo no puede aprender nada real."""
    rng = np.random.default_rng(semilla)
    out = []
    for p in pools:
        p = p[p.groupby("q").ganador.transform("max").values > 0].copy()
        r = pd.Series(rng.random(len(p))).groupby(p.q.values).rank(method="first").values
        p["ganador"] = (r == 1).astype(np.int8)
        out.append(p)
    return out


class Fijo:
    """Para pasar a modelo_v2.ranking un modelo ya entrenado sobre pools modificados."""
    def __init__(self, b):
        self.b = b

    def predict(self, X):
        return self.b.predict(X)


# =====================================================================================
# Modo validar: escalera, nulos, nats, β
# =====================================================================================
def validar(V, K):
    D = cargar(V)
    sems = preparar(D, bs.SEMESTRES, K)
    tr, te = ["2022S2", "2023S1", "2023S2"], "2024S1"
    dt = sems[te]

    filas, rr, lp = [], {}, {}
    escalones, acumuladas = [], []
    for nombre, cols in TERMINOS.items():
        acumuladas = acumuladas + cols
        escalones.append((f"+ {nombre}" if escalones else nombre, list(acumuladas)))
    escalones.append((f"Boltzmann completo ({len(todas_las_pistas())} pistas)", todas_las_pistas()))

    mods = {}
    for nombre, vv in escalones:
        l2 = elegir_l2(sems, vv)
        mod = ajustar(sems, tr, vv, l2)
        mods[nombre] = (mod, vv)
        rk = m.ranking(mod, dt, K, vv)
        rr[nombre] = bs.por_licitacion(rk, dt)
        lp[nombre] = log_p_ganador(mod, dt, vv)
        filas.append({"modelo": nombre, "pistas": len(vv), "L2": l2, **m.medir(D, dt, rk)})
        log(f"   {nombre}: MRR {filas[-1]['MRR']:.4f}")
    T = pd.DataFrame(filas)
    base = escalones[0][0]
    comunes = lp[base].index
    T["información sobre el B0 (nats por licitación)"] = [(lp[n].reindex(comunes) - lp[base]).mean() for n, _ in escalones]
    prev = None
    sube = []
    for n, _ in escalones:
        if prev is None:
            sube.append("")
        else:
            lo, hi = bs.ic_bootstrap(rr[n] - rr[prev])
            sube.append(f"{(rr[n] - rr[prev]).mean():+.4f} ({lo:+.4f}, {hi:+.4f})")
        prev = n
    T["sube respecto del escalón anterior (IC95)"] = sube

    # ---- modelos nulos: cada término de la escalera, barajado ----
    final = escalones[len(TERMINOS) - 1]                  # el escalón con los 4 términos
    mod_f, vv_f = mods[final[0]]
    nulos = []
    for nombre, cols in list(TERMINOS.items())[1:]:
        pools = barajar([sems[s].pool for s in tr], cols, 1)
        b = bs.Boltzmann(vv_f, l2=mod_f.l2).fit(pools)
        dt_b = dt.pool
        dt.pool = barajar([dt_b], cols, 2)[0]
        rk = m.ranking(Fijo(b), dt, K, vv_f)
        dt.pool = dt_b
        r = bs.por_licitacion(rk, dt)
        lo, hi = bs.ic_bootstrap(rr[final[0]] - r)
        nulos.append({"modelo nulo": f"término '{nombre}' barajado entre candidatas",
                      "MRR": r.mean(), "real − nulo (IC95)": f"{(rr[final[0]] - r).mean():+.4f} ({lo:+.4f}, {hi:+.4f})"})
    b = bs.Boltzmann(vv_f, l2=mod_f.l2).fit(barajar_ganadores([sems[s].pool for s in tr], 3))
    r = bs.por_licitacion(m.ranking(Fijo(b), dt, K, vv_f), dt)
    lo, hi = bs.ic_bootstrap(rr[final[0]] - r)
    nulos.append({"modelo nulo": "ganadores de entrenamiento barajados", "MRR": r.mean(),
                  "real − nulo (IC95)": f"{(rr[final[0]] - r).mean():+.4f} ({lo:+.4f}, {hi:+.4f})"})
    N = pd.DataFrame(nulos)

    beta = mod_f.tabla_beta()
    beta["término"] = beta.pista.map({c: n for n, cs in TERMINOS.items() for c in cs})
    T.to_csv(f"final_escalera_{V}.csv", index=False)
    N.to_csv(f"final_nulos_{V}.csv", index=False)
    beta[["término", "pista", "beta"]].to_csv(f"final_beta_{V}.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.max_colwidth", 60,
                           "display.float_format", "{:.4f}".format):
        print(f"\n=== ESCALERA DE ENERGÍA, vertical {V} (entrenado 2022S2–2023S2, medido en 2024S1) ===")
        print(T[["modelo", "pistas", "MRR", "MRR S_inc", "MRR S_new", "B0 MRR", "Δ MRR", "Δ IC95",
                 "sube respecto del escalón anterior (IC95)", "información sobre el B0 (nats por licitación)"]]
              .to_string(index=False))
        print(f"\n=== MODELOS NULOS (sobre el escalón '{final[0]}') ===")
        print(N.to_string(index=False))
        print(f"\n=== β del escalón '{final[0]}' (pistas estandarizadas) ===")
        print(beta[["término", "pista", "beta"]].to_string(index=False))


# =====================================================================================
# Modo entregar: entrenar con todo y rankear las licitaciones nuevas
# =====================================================================================
def entregar(V, K, licitaciones, candidatas, salida, simular=None, historial=None):
    D = cargar(V, licitaciones, candidatas, historial)
    # las licitaciones a rankear son EXACTAMENTE las del archivo nuevo (no se deduce por fecha:
    # podrían caer en un semestre que ya tiene licitaciones con ganador)
    if simular:                                     # prueba del modo: hace de cuenta que 'simular' es nuevo
        ids = set(D.lic.id[D.lic.semestre == simular])
    else:
        ids = set(pd.read_parquet(licitaciones).id)
    D.gan = D.gan[~D.gan.id.isin(ids)]              # por las dudas: nunca usar un ganador de lo que se rankea
    nuevos = sorted(set(D.lic.semestre[D.lic.id.isin(ids)]))
    con_gan = set(D.lic.semestre[D.lic.id.isin(D.gan.id)])
    entrenamiento = [s for s in bs.SEMESTRES if s in con_gan]
    if not ids:
        raise SystemExit("No hay licitaciones nuevas: revisar --licitaciones")
    log(f"{len(ids)} licitaciones a rankear (semestres {nuevos}); entrenamiento: {entrenamiento}")
    sems = preparar(D, sorted(set(entrenamiento) | set(nuevos)), K)
    vv = todas_las_pistas()
    l2 = elegir_l2(sems, vv)
    mod = ajustar(sems, entrenamiento, vv, l2)   # Boltzmann.fit sólo usa licitaciones con ganador
    rk = pd.concat([m.ranking(mod, sems[s], K, vv) for s in nuevos], ignore_index=True)
    rk = rk[rk.id.isin(ids)]
    faltan = ids - set(rk.id)
    if faltan:
        log(f"AVISO: {len(faltan)} licitaciones sin ranking (¿sin candidatas en su semestre?)")
    rk[["id", "parent_uei", "puesto"]].to_csv(salida, index=False)
    log(f"ranking de {rk.id.nunique()} licitaciones -> {salida} (L2 = {l2}, {len(vv)} pistas)")
    return rk



# =====================================================================================
# Modo entregar GLOBAL (recomendado): UN solo modelo para todas las verticales
# =====================================================================================
# Se busca una metodología que ande igual de bien en cualquier vertical, incluso en una
# que no está en los datos de entrenamiento. Se entrena un único juego de β con las licitaciones
# con ganador de TODAS las verticales disponibles y se aplica a la vertical pedida. En 2024S1,
# este modelo global rinde en promedio lo mismo que entrenar una vertical por vez
# (exp_universal.py) y no depende de conocer la vertical.
VERTICALES_ENTRENAMIENTO = ["d", "it", "j", "r", "rd"]
USAR_L2 = False          # False: tronco lasso (modelo del informe). True: L2 elegido en validación.


def por_vertical(entrenamiento):
    """{(vertical, semestre): sem} -> {vertical: [pool, ...]} para el tronco lasso."""
    out = {}
    for (W, _), x in sorted(entrenamiento.items()):
        out.setdefault(W, []).append(x.pool)
    return out


class _Sem:
    """Lo mínimo de un semestre para entrenar y validar (así no se guardan cinco verticales enteras)."""
    def __init__(self, d, vv):
        cols = ["q", "c", "ganador", "puesto_min"] + list(vv)
        self.pool = d.pool[cols].copy()
        for c in vv:
            self.pool[c] = self.pool[c].astype(np.float32)
        self.n_gan, self.q = d.n_gan, d.q[["id"]].copy()


def entregar_global(V, K, licitaciones, candidatas, salida, simular=None, historial=None,
                    verticales=VERTICALES_ENTRENAMIENTO):
    def con_ganador(D, excluir=()):
        sg = set(D.lic.semestre[D.lic.id.isin(D.gan.id) & ~D.lic.id.isin(excluir)])
        return [s for s in bs.SEMESTRES if s in sg and (not simular or s < simular)]

    entrenamiento = {}                               # (vertical, semestre) -> semestre preparado
    vv = None
    for W in verticales:                             # 1) las otras verticales: sólo para entrenar
        if W == V:
            continue
        D = cargar(W)
        S = preparar(D, con_ganador(D), K)
        vv = todas_las_pistas()
        for s, d in S.items():
            entrenamiento[(W, s)] = _Sem(d, vv)
        del D, S
        log(f"{W}: lista para entrenar")
    # 2) la vertical pedida: sus semestres con ganador (si tiene) + los que hay que rankear
    D = cargar(V, licitaciones, candidatas, historial)
    ids = set(D.lic.id[D.lic.semestre == simular]) if simular else set(pd.read_parquet(licitaciones).id)
    if not ids:
        raise SystemExit("No hay licitaciones nuevas: revisar --licitaciones")
    D.gan = D.gan[~D.gan.id.isin(ids)]
    nuevos = sorted(set(D.lic.semestre[D.lic.id.isin(ids)]))
    propios = con_ganador(D, ids)
    S = preparar(D, sorted(set(propios) | set(nuevos)), K)
    vv = todas_las_pistas()
    for s in propios:
        entrenamiento[(V, s)] = S[s]
    verts = sorted({W for W, _ in entrenamiento})
    log(f"{len(ids)} licitaciones de {V} a rankear (semestres {nuevos}); entrenamiento global con {verts}")

    def media(l2):                                   # L2 con las dos ventanas, sobre todas las verticales
        vals = []
        for tr, va in VENTANAS:
            claves = [(W, s) for W in verts for s in tr if (W, s) in entrenamiento]
            mod = bs.Boltzmann(vv, l2=l2).fit([entrenamiento[k].pool for k in claves])
            vals += [bs.rr_rapido(entrenamiento[(W, va)], mod.predict(entrenamiento[(W, va)].pool[vv].values)).mean()
                     for W in verts if (W, va) in entrenamiento]
        return np.mean(vals)
    if USAR_L2:
        l2 = max(L2S, key=media)
        mod = bs.Boltzmann(vv, l2=l2).fit([x.pool for x in entrenamiento.values()])
        desc = f"L2 = {l2}"
    else:
        mod = Tronco(vv).fit(por_vertical(entrenamiento))
        desc = f"tronco lasso, {int((mod.beta != 0).sum())} pistas activas"
    rk = pd.concat([m.ranking(mod, S[s], K, vv) for s in nuevos], ignore_index=True)
    rk = rk[rk.id.isin(ids)]
    faltan = ids - set(rk.id)
    if faltan:
        log(f"AVISO: {len(faltan)} licitaciones sin ranking (¿sin candidatas en su semestre?)")
    rk[["id", "parent_uei", "puesto"]].to_csv(salida, index=False)
    mod.tabla_beta().to_csv(salida.replace(".csv", "_beta.csv"), index=False)
    log(f"ranking de {rk.id.nunique()} licitaciones -> {salida} (modelo global, {desc}, {len(vv)} pistas)")
    return rk



TAM_BLOQUE = 400     # licitaciones nuevas por bloque (para no agotar la memoria con 30 000+ candidatas)


def _archivos(carpeta, W):
    import os
    arch = {n: os.path.join(carpeta, W, f"{n}.parquet") if carpeta else None
            for n in ("licitaciones", "empresas", "historial")}
    return {n: (r if r and os.path.exists(r) else None) for n, r in arch.items()}


def _cargar_nuevas(W, carpeta, simular):
    """Carga la vertical con las licitaciones nuevas agregadas. Devuelve (D, ids nuevos)."""
    arch = _archivos(carpeta, W)
    sem_test = None
    if arch["licitaciones"] and arch["empresas"] and "semestre" not in pd.read_parquet(arch["empresas"]).columns:
        sem_test = b0.semestre(pd.to_datetime(pd.read_parquet(arch["licitaciones"]).t).min())
        log(f"{W}: formato test (empresas sin semestre): corte de información al inicio de {sem_test}")
    D = cargar(W, arch["licitaciones"], arch["empresas"], arch["historial"], sem_test)
    if simular:
        ids = set(D.lic.id[D.lic.semestre == simular])
    else:
        ids = set(pd.read_parquet(arch["licitaciones"]).id) if arch["licitaciones"] else set()
    D.gan = D.gan[~D.gan.id.isin(ids)]
    return D, ids


def entregar_todas(K, carpeta, simular=None, verticales=VERTICALES_ENTRENAMIENTO, salida_dir="."):
    """Las cinco verticales: (1) se entrena UN modelo global con los semestres con ganadores de todas
    las verticales; (2) se rankean las licitaciones nuevas de cada vertical, por bloques, y se escribe
    <vertical>.csv. Espera <carpeta>/<vertical>/licitaciones.parquet y empresas.parquet (formato del
    test: empresas sin columna semestre)."""
    import os
    # ---- 1. entrenamiento ----
    entrenamiento, vv = {}, None
    for W in verticales:
        D, ids = _cargar_nuevas(W, carpeta, simular)
        sg = set(D.lic.semestre[D.lic.id.isin(D.gan.id) & ~D.lic.id.isin(ids)])
        propios = [s for s in bs.SEMESTRES if s in sg and (not simular or s < simular)]
        S = preparar(D, propios, K)
        vv = todas_las_pistas()
        for s in propios:
            entrenamiento[(W, s)] = _Sem(S[s], vv)
        del D, S
        log(f"{W}: {len(propios)} semestres para entrenar")
    verts = sorted({W for W, _ in entrenamiento})

    def media(l2):
        vals = []
        for tr, va in VENTANAS:
            claves = [(W, s) for W in verts for s in tr if (W, s) in entrenamiento]
            mod = bs.Boltzmann(vv, l2=l2).fit([entrenamiento[k].pool for k in claves])
            vals += [bs.rr_rapido(entrenamiento[(W, va)], mod.predict(entrenamiento[(W, va)].pool[vv].values)).mean()
                     for W in verts if (W, va) in entrenamiento]
        return np.mean(vals)
    if USAR_L2:
        l2 = max(L2S, key=media)
        mod = bs.Boltzmann(vv, l2=l2).fit([x.pool for x in entrenamiento.values()])
        desc = f"L2 = {l2}"
    else:
        mod = Tronco(vv).fit(por_vertical(entrenamiento))
        desc = f"tronco lasso, {int((mod.beta != 0).sum())} pistas activas"
    del entrenamiento
    mod.tabla_beta().to_csv(os.path.join(salida_dir, "entrega_beta_global.csv"), index=False)
    log(f"modelo global: {desc}, {len(vv)} pistas, entrenado con {verts}")

    # ---- 2. ranking de las licitaciones nuevas, por bloques ----
    for W in verticales:
        D, ids = _cargar_nuevas(W, carpeta, simular)
        if not ids:
            continue
        nuevas = D.lic[D.lic.id.isin(ids)].sort_values(["semestre", "t", "id"])
        sem_orig = D.lic.semestre.copy()
        partes = []
        for sem, grupo in nuevas.groupby("semestre"):
            lista = list(grupo.id)
            for a in range(0, len(lista), TAM_BLOQUE):
                bloque = set(lista[a:a + TAM_BLOQUE])
                D.lic["semestre"] = sem_orig
                D.lic["semestre"] = D.lic.semestre.mask(D.lic.id.isin(ids - bloque), "pendiente")
                d = preparar(D, [sem], K)[sem]
                partes.append(m.ranking(mod, d, K, vv))
                del d
                log(f"{W} {sem}: rankeadas {min(a + TAM_BLOQUE, len(lista))}/{len(lista)}")
        rk = pd.concat(partes, ignore_index=True)
        rk = rk[rk.id.isin(ids)]
        if ids - set(rk.id):
            log(f"AVISO {W}: {len(ids - set(rk.id))} licitaciones sin ranking")
        ruta = os.path.join(salida_dir, f"{W}.csv")
        rk[["id", "parent_uei", "puesto"]].to_csv(ruta, index=False)
        log(f"{W}: ranking de {rk.id.nunique()} licitaciones -> {ruta}")
        del D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vertical", default="j")
    ap.add_argument("--modo", choices=["validar", "entregar"], default="validar")
    ap.add_argument("--K", type=int, default=300)
    ap.add_argument("--licitaciones", help="parquet con las licitaciones nuevas (modo entregar)")
    ap.add_argument("--candidatas", help="parquet con las empresas candidatas nuevas (modo entregar)")
    ap.add_argument("--salida", help="CSV de salida (modo entregar)")
    ap.add_argument("--historial", help="(opcional) parquet con historial nuevo, si lo entregan")
    ap.add_argument("--simular", help="(prueba) tratar este semestre como nuevo, p. ej. 2024S1")
    ap.add_argument("--salida-dir", default=".", help="carpeta donde se escriben <vertical>.csv (con --vertical todas)")
    ap.add_argument("--nuevas", help="carpeta con <vertical>/licitaciones.parquet y empresas.parquet (con --vertical todas)")
    ap.add_argument("--l2", action="store_true",
                    help="usar el Boltzmann con L2 en lugar del tronco lasso (comparación)")
    ap.add_argument("--por-vertical", action="store_true",
                    help="entrenar sólo con la vertical pedida (por defecto: modelo global con todas)")
    a = ap.parse_args()
    global USAR_L2
    USAR_L2 = a.l2
    if a.modo == "validar":
        validar(a.vertical, a.K)
    else:
        if a.vertical == "todas":
            os.makedirs(a.salida_dir, exist_ok=True)
            return entregar_todas(a.K, a.nuevas, a.simular, salida_dir=a.salida_dir)
        f = entregar if a.por_vertical else entregar_global
        f(a.vertical, a.K, a.licitaciones, a.candidatas, a.salida or f"entrega_{a.vertical}.csv", a.simular, a.historial)


if __name__ == "__main__":
    main()
