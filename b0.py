"""B0: el prior de incumbencia. El baseline contra el que se mide todo.

Para una licitación publicada en `t`, rankea las empresas candidatas usando **sólo quién
ganó antes**, sin mirar ningún texto. Si un modelo con texto no le gana a esto, el texto
no aportó nada.

Cómo funciona. Para cada nivel —misma oficina y PSC, mismo PSC, mismo NAICS, todo, y mismo
PSC con el mismo set-aside— se cuentan los contratos que ganó cada empresa antes de `t`,
pesando los viejos menos (vida media de 2,5 años), y se normaliza a una distribución.
`p_b0` es la mezcla de esas distribuciones con pesos fijos. Una empresa que nunca ganó en
ninguno de los niveles que aplican queda con `p_b0 = 0`.

Es determinístico: los empates exactos (sobre todo las muchas empresas con `p_b0 = 0`) se
ordenan con una semilla que sale del `id` de la licitación, así que da lo mismo en
cualquier máquina, en cualquier orden y con cualquier subconjunto de licitaciones.

Uso, como script (escribe el top-K en el formato que lee `evaluar.py`):

    uv run b0.py data/j --salida b0_j.csv

Uso, como módulo (`p_b0` de todas las candidatas, para combinarlo con otro modelo):

    import b0
    prior = b0.B0(historial)
    p = prior.p_b0(licitacion)          # pd.Series indexada por parent_uei
"""
from __future__ import annotations

import argparse
import hashlib
import pathlib
import re

import numpy as np
import pandas as pd

TAU_ANIOS = 2.5
NIVELES = [("psc", "office"), ("psc",), ("naics",), (), ("psc", "sa")]
PESOS = [0.40, 0.24, 0.12, 0.04, 0.20]

# La licitación y el historial escriben el set-aside distinto ("Small Business Set Aside -
# Total" contra "SMALL BUSINESS SET ASIDE - TOTAL"); se llevan los dos al mismo puñado.
_SA = [(re.compile(r"8\s*\(?a\)?", re.I), "8A"),
       (re.compile(r"service.?disabled|sdvosb", re.I), "SDVOSB"),
       (re.compile(r"women|wosb|edwosb", re.I), "WOSB"),
       (re.compile(r"hubzone", re.I), "HUBZONE"),
       (re.compile(r"veteran", re.I), "VOSB"),
       (re.compile(r"small business", re.I), "SMALL")]


def set_aside(s) -> str:
    s = "" if s is None or s != s else str(s)
    if not s.strip() or re.search(r"no set aside", s, re.I):
        return "NONE"
    for rx, et in _SA:
        if rx.search(s):
            return et
    return "OTRO"


def semestre(t) -> str:
    """"2024S1" (enero–junio) o "2024S2" (julio–diciembre): la llave con `empresas`."""
    t = pd.Timestamp(t)
    return f"{t.year}S{1 if t.month <= 6 else 2}"


def candidatas(empresas: pd.DataFrame, t) -> np.ndarray:
    """Las empresas que se rankean para una licitación publicada en `t`.

    Con columna `semestre`, las de ese semestre; sin ella, todas.
    """
    if "semestre" in empresas:
        return empresas.parent_uei.values[empresas.semestre.values == semestre(t)]
    return empresas.parent_uei.values


class B0:
    """El prior, indexado una vez sobre el historial y consultado por licitación."""

    def __init__(self, historial: pd.DataFrame):
        h = historial.copy()
        h["psc"] = h.psc.astype(str).str.upper()
        h["office"] = h.office.astype(str).str.upper()
        h["naics"] = h.naics.astype(str)
        h["sa"] = h.set_aside.map(set_aside)
        self.empresas = np.array(sorted(h.parent_uei.unique()))
        self.idx = {u: i for i, u in enumerate(self.empresas)}
        h["fi"] = h.parent_uei.map(self.idx).astype(np.int64)
        h["ts"] = pd.to_datetime(h.fecha).values.astype("datetime64[D]").astype(np.int64)
        self.h = h.sort_values(["ts", "fi"], kind="stable").reset_index(drop=True)
        # exp((t_award - t) / tau): el factor exp(-t/tau) es común a todas las empresas y
        # se cancela al normalizar, así que alcanza con exp(t_award / tau). Se resta una
        # referencia para que no desborde.
        self.w = np.exp((self.h.ts.values - self.h.ts.values.max()) / (TAU_ANIOS * 365.25))
        self.tablas = [self._indexar(cols) for cols in NIVELES]

    def _indexar(self, cols: tuple) -> dict:
        # Las claves son SIEMPRE tuplas, también la del nivel global, `()`: con una
        # columna sola, pandas devuelve 1-tuplas y una consulta escalar no matchearía.
        if not cols:
            return {(): (self.h.ts.values, self.h.fi.values, self.w)}
        out = {}
        for k, g in self.h.groupby(list(cols), sort=False):
            i = g.index.values
            out[k if isinstance(k, tuple) else (k,)] = (
                self.h.ts.values[i], self.h.fi.values[i], self.w[i])
        return out

    def _p(self, lic) -> np.ndarray:
        ts = np.datetime64(pd.Timestamp(lic.t), "D").astype(np.int64)
        ctx = {"psc": str(lic.psc).upper(), "office": str(lic.office).upper(),
               "naics": str(lic.naics), "sa": set_aside(lic.set_aside)}
        p = np.zeros(len(self.empresas))
        for cols, peso, tabla in zip(NIVELES, PESOS, self.tablas):
            e = tabla.get(tuple(ctx[c] for c in cols))
            if e is None:
                continue
            t_arr, fi, w = e
            n = np.searchsorted(t_arr, ts, side="left")     # estricto: sólo antes de t
            if n == 0:
                continue
            v = np.bincount(fi[:n], weights=w[:n], minlength=len(self.empresas))
            p += peso * v / v.sum()
        return p

    def p_b0(self, lic, empresas: pd.DataFrame | None = None) -> pd.Series:
        """`p_b0` de cada candidata de la licitación (todas las del historial si no se
        pasa `empresas`). Las candidatas sin historial previo quedan en 0."""
        p = pd.Series(self._p(lic), index=self.empresas)
        if empresas is None:
            return p
        return p.reindex(candidatas(empresas, lic.t), fill_value=0.0)

    def ranking(self, lic, empresas: pd.DataFrame, top: int = 100) -> pd.DataFrame:
        """Top-`top` de la licitación, en el formato que lee `evaluar.py`."""
        p = self.p_b0(lic, empresas)
        semilla = int(hashlib.sha256(str(lic.id).encode()).hexdigest()[:16], 16)
        azar = np.random.default_rng(semilla).random(len(p))
        orden = np.lexsort((azar, -p.values))[:top]   # por p; entre empates exactos, azar
        return pd.DataFrame({"id": lic.id, "parent_uei": p.index.values[orden],
                             "puesto": np.arange(1, len(orden) + 1),
                             "p_b0": p.values[orden]})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("datos", help="carpeta con licitaciones.parquet y empresas.parquet")
    ap.add_argument("--historial", help="por defecto, <datos>/historial.parquet")
    ap.add_argument("--salida", required=True, help="CSV con el ranking")
    ap.add_argument("--top", type=int, default=100)
    a = ap.parse_args()

    d = pathlib.Path(a.datos)
    lic = pd.read_parquet(d / "licitaciones.parquet")
    emp = pd.read_parquet(d / "empresas.parquet")
    hist = pd.read_parquet(a.historial or d / "historial.parquet")
    prior = B0(hist)
    out = pd.concat([prior.ranking(f, emp, a.top) for f in lic.itertuples()],
                    ignore_index=True)
    out.to_csv(a.salida, index=False)
    print(f"{len(lic):,} licitaciones -> {a.salida}")


if __name__ == "__main__":
    main()
