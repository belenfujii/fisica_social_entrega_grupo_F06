# =====================================================================================
# pista_nombre.py
#
# PISTA: ¿el texto de la licitación NOMBRA a la empresa?
#
# Muchos avisos son de contratación directa ("intent to award a sole source contract to
# Gartner, Inc.") o piden una marca ("Xerox printer lease"). El aviso ya lo dice el día en que
# se publica, así que es información legítima: es "lo que agrega el texto" sobre el B0.
#
# El problema son los falsos positivos: empresas que se llaman "American", "Precision" o
# "Commercial Services" aparecen en muchísimos textos sin ser nombradas. Por eso, además de la
# mención, se mide la SORPRESA del nombre (información de Shannon):
#
#     sorpresa(nombre) = -log10( f + 1e-5 ) / 5        (entre 0 y 1)
#
# donde f es la fracción de descripciones del historial ANTERIOR a 2022 (antes de todas las
# licitaciones) en las que aparece ese nombre. Un nombre que nunca aparece ("Trofholz") tiene
# sorpresa 1; uno que aparece en el 1% de los textos ("American"), 0,4. Es la misma idea que el
# IDF de TF-IDF: una coincidencia rara pesa más que una común.
#
# Columnas que agrega al pool:
#   nombre_mencion   1 si el nombre de la candidata aparece en el título o la descripción
#   nombre_sorpresa  sorpresa del nombre mencionado (0 si no se menciona)
#   nombre_ntok      cantidad de palabras del nombre mencionado (0 si no se menciona)
# y las candidatas nombradas entran al pool aunque ninguna otra fuente las proponga.
# =====================================================================================
import collections
import re

import numpy as np
import pandas as pd

FORMAS = r"\b(inc|llc|l l c|corp|corporation|co|company|ltd|limited|lp|llp|the|of|and|dba|d b a|incorporated)\b"
VARIABLES_NOMBRE = ["nombre_mencion", "nombre_sorpresa", "nombre_ntok"]
CORTE_FRECUENCIA = pd.Timestamp("2022-01-01")


def normalizar(s):
    """Nombre sin puntuación ni formas jurídicas: 'Gartner, Inc.' -> 'gartner'."""
    s = re.sub(r"[^a-z0-9 ]", " ", str(s).lower())
    s = re.sub(FORMAS, " ", s)
    return re.sub(r"\s+", " ", s).strip()


def tokens(s):
    return re.sub(r"[^a-z0-9 ]", " ", str(s).lower()).split()


class PistaNombre:
    def __init__(self, D, largo_min=5, muestra=60_000):
        nombre = D.emp.drop_duplicates("parent_uei", keep="last").set_index("parent_uei").name.map(normalizar)
        self.por_nombre = {}
        for u, n in nombre.items():
            if len(n) >= largo_min:
                self.por_nombre.setdefault(n, set()).add(u)
        self.largo_max = max(len(n.split()) for n in self.por_nombre)
        # frecuencia de cada nombre en textos del pasado (antes de todas las licitaciones)
        h0 = D.hist.descripcion[D.hist.fecha < CORTE_FRECUENCIA]
        h0 = h0.sample(min(muestra, len(h0)), random_state=0)
        cuenta = collections.Counter()
        for t in h0:
            cuenta.update(self._ngramas(t) & self.por_nombre.keys())
        n = max(len(h0), 1)
        self.sorpresa = {k: float(min(1.0, -np.log10(cuenta[k] / n + 1e-5) / 5)) for k in self.por_nombre}
        self._cand_id, self._pos = None, None

    def _ngramas(self, texto):
        tk = tokens(texto)
        return {" ".join(tk[i:i + k]) for i in range(len(tk)) for k in range(1, min(self.largo_max, len(tk) - i) + 1)}

    def __call__(self, f, cand):
        if self._cand_id != id(cand):                      # índice uei -> posición (por semestre)
            self._cand_id, self._pos = id(cand), {u: i for i, u in enumerate(cand.parent_uei)}
        n = len(cand)
        mencion, sorpresa, ntok = np.zeros(n), np.zeros(n), np.zeros(n)
        for nom in self._ngramas(f"{f.title} {f.description}") & self.por_nombre.keys():
            for u in self.por_nombre[nom]:
                i = self._pos.get(u)
                if i is not None:
                    mencion[i] = 1
                    sorpresa[i] = max(sorpresa[i], self.sorpresa[nom])
                    ntok[i] = max(ntok[i], len(nom.split()))
        puesto = np.where(mencion > 0, 1.0, np.inf).astype(np.float32)
        return puesto, {"nombre_mencion": mencion, "nombre_sorpresa": sorpresa, "nombre_ntok": ntok}
