"""Evalúa un ranking: MRR (la métrica de referencia), Hit@5 y Hit@1, y cuánto mejora sobre el B0.

Además reporta, como diagnóstico, Hit@100, Recall@100 y el puesto mediano del mejor ganador.

El ranking es un CSV con una fila por (licitación, empresa) y hasta 100 empresas por
licitación (pueden ser menos; un ganador que no está en la lista cuenta 0):

    id,parent_uei,puesto
    FA865124R0012,ABC123DEF456,1
    FA865124R0012,XYZ987GHI654,2

Uso, sobre una validación propia (las licitaciones de `respuestas` definen qué se evalúa):

    uv run evaluar.py mi_ranking.csv --datos data/j --respuestas mi_val_j.parquet \\
                      --b0 b0_j.csv

`--datos` es la carpeta con `licitaciones.parquet` y `empresas.parquet` (para chequear
que cada empresa rankeada sea una candidata válida); `--respuestas`, un parquet con
`id, parent_uei` (por defecto, `<datos>/ganadores.parquet`); `--b0`, el ranking que
escribe `b0.py` sobre las mismas licitaciones, para medir la mejora.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

from b0 import semestre

K_MAX = 100
S_INC = 10           # el B0 pone a un ganador en su top-10: el historial ya alcanza


class RankingInvalido(Exception):
    pass


def validar(rk: pd.DataFrame, lic: pd.DataFrame, emp: pd.DataFrame, ids: set) -> None:
    faltan = {"id", "parent_uei", "puesto"} - set(rk.columns)
    if faltan:
        raise RankingInvalido(f"faltan columnas: {sorted(faltan)}")
    hay = set(rk.id)
    if ids - hay:
        raise RankingInvalido(f"faltan {len(ids - hay)} licitaciones, p. ej. {sorted(ids - hay)[:3]}")
    if hay - ids:
        raise RankingInvalido(f"sobran {len(hay - ids)} licitaciones, p. ej. {sorted(hay - ids)[:3]}")
    if rk.duplicated(["id", "parent_uei"]).any():
        raise RankingInvalido("una empresa aparece dos veces en la misma licitación")
    n = rk.groupby("id").puesto.agg(["min", "max", "size", "nunique"])
    malos = n[(n["min"] != 1) | (n["max"] != n["size"]) | (n["nunique"] != n["size"])]
    if len(malos):
        raise RankingInvalido(f"los puestos tienen que ser 1..K sin repetir: {list(malos.index[:3])}")
    if n["size"].max() > K_MAX:
        raise RankingInvalido(f"más de {K_MAX} empresas en {list(n.index[n['size'] > K_MAX][:3])}")
    # cada empresa tiene que ser candidata de su licitación (la regla de `b0.candidatas`)
    if "semestre" in emp:
        x = rk[["id", "parent_uei"]].assign(
            semestre=rk.id.map(lic.set_index("id").t).map(semestre))
        ok = x.merge(emp[["parent_uei", "semestre"]], how="left", indicator=True)
        ok = ok._merge.eq("both").values
    else:
        x, ok = rk, rk.parent_uei.isin(set(emp.parent_uei)).values
    if not ok.all():
        malo = x[~ok].iloc[0]
        raise RankingInvalido(f"{malo.id}: {int((~ok).sum())} empresas no son candidatas, "
                              f"p. ej. {malo.parent_uei}")


def por_licitacion(rk: pd.DataFrame, gan: pd.DataFrame) -> pd.DataFrame:
    """Una fila por licitación: el MRR filtrado, Hit@1, Hit@5 y el puesto del mejor ganador.

    **MRR filtrado.** Cada ganador se puntúa con 1/puesto, donde el puesto no cuenta a los
    otros ganadores que tiene arriba, y la licitación vale el promedio sobre sus ganadores.
    Con un solo ganador es el MRR de siempre. Con varios, premia encontrarlos a todos sin
    castigar que un ganador quede detrás de otro. Un ganador fuera del ranking vale 0.

    Ejemplo: ganadores en los puestos 1, 2 y 50 -> puestos filtrados 1, 1 y 48 ->
    (1 + 1 + 1/48) / 3 = 0,67.
    """
    m = gan.merge(rk[["id", "parent_uei", "puesto"]], on=["id", "parent_uei"], how="left")
    m["puesto"] = m.puesto.fillna(np.inf)
    m = m.sort_values(["id", "puesto"])
    # los ganadores de la misma licitación que tiene arriba (ya ordenados por puesto)
    m["filtrado"] = m.puesto - m.groupby("id").cumcount()
    m["rr"] = 1 / m.filtrado
    m["en_lista"] = np.isfinite(m.puesto).astype(float)
    g = m.groupby("id")
    mejor = g.puesto.min()
    return pd.DataFrame({"MRR": g.rr.mean(), "Hit@1": (mejor <= 1).astype(float),
                         "Hit@5": (mejor <= 5).astype(float), "mejor": mejor,
                         "recall": g.en_lista.mean()}).sort_index()


METRICAS = ["MRR", "Hit@5", "Hit@1"]


def diagnostico(x: pd.DataFrame) -> dict:
    """Para ver el progreso que el MRR casi no registra (no son métricas oficiales).

    `Hit@100`: en qué fracción de las licitaciones aparece algún ganador en la lista.
    `Recall@100`: qué fracción de los ganadores aparece en la lista, promediada por
    licitación. Con un solo ganador es igual a Hit@100; con varios, cuenta a cada uno.
    `puesto mediano`: el del mejor ganador, entre las licitaciones donde aparece. Subir un
    ganador del puesto 80 al 30 casi no mueve el MRR (+0,02), pero se ve acá.
    """
    en = np.isfinite(x.mejor)
    return {f"Hit@{K_MAX}": float(en.mean()), f"Recall@{K_MAX}": float(x.recall.mean()),
            "puesto mediano": float(x.mejor[en].median()) if en.any() else np.nan}


def bootstrap_delta(a: np.ndarray, b: np.ndarray, n: int = 2000) -> tuple[float, float]:
    """IC95% de la media de a - b, remuestreando licitaciones (pareado)."""
    d = a - b
    i = np.random.default_rng(0).integers(0, len(d), size=(n, len(d)))
    m = d[i].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def evaluar(rk, lic, emp, gan, rk_b0=None) -> pd.DataFrame:
    """Una fila por conjunto (todas; S_inc y S_new si hay B0) con las métricas y el Δ."""
    ids = set(gan.id)
    validar(rk, lic, emp, ids)
    x = por_licitacion(rk, gan)
    if rk_b0 is None:
        return pd.DataFrame([{"conjunto": "todas", "N": len(x),
                              **{k: x[k].mean() for k in METRICAS}, **diagnostico(x)}])
    validar(rk_b0, lic, emp, ids)
    x0 = por_licitacion(rk_b0, gan)
    filas = []
    for nombre, m in (("todas", np.ones(len(x), bool)),
                      ("S_inc (B0 top-10)", (x0.mejor <= S_INC).values),
                      ("S_new (el resto)", (x0.mejor > S_INC).values)):
        if not m.any():
            continue
        a, b = x[m], x0[m]
        f = {"conjunto": nombre, "N": int(m.sum())}
        for k in METRICAS:
            f[k] = a[k].mean()
        f["B0 MRR"] = b.MRR.mean()
        f["Δ MRR"] = a.MRR.mean() - b.MRR.mean()
        f["Δ MRR IC95"] = "({:+.3f}, {:+.3f})".format(*bootstrap_delta(a.MRR.values, b.MRR.values))
        f["B0 Hit@5"] = b["Hit@5"].mean()
        f["Δ Hit@5"] = a["Hit@5"].mean() - b["Hit@5"].mean()
        f.update(diagnostico(a))
        f.update({f"B0 {k}": v for k, v in diagnostico(b).items()})
        filas.append(f)
    return pd.DataFrame(filas)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("ranking")
    ap.add_argument("--datos", required=True)
    ap.add_argument("--respuestas")
    ap.add_argument("--b0")
    a = ap.parse_args()

    d = pathlib.Path(a.datos)
    lic = pd.read_parquet(d / "licitaciones.parquet")
    emp = pd.read_parquet(d / "empresas.parquet")
    gan = pd.read_parquet(a.respuestas or d / "ganadores.parquet")
    rk = pd.read_csv(a.ranking)
    rk_b0 = pd.read_csv(a.b0) if a.b0 else None
    # Con respuestas propias (una validación), se evalúa sólo ese subconjunto aunque el
    # ranking cubra más. Sin ellas tienen que estar todas, justas.
    if a.respuestas:
        rk = rk[rk.id.isin(set(gan.id))]
    if rk_b0 is not None:
        rk_b0 = rk_b0[rk_b0.id.isin(set(gan.id))]
    try:
        res = evaluar(rk, lic, emp, gan, rk_b0)
    except RankingInvalido as e:
        sys.exit(f"ranking inválido: {e}")
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print(res.to_string(index=False))


if __name__ == "__main__":
    main()
