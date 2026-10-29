"""Chequea que predicciones/<vertical>.csv cumpla una predicción por
licitación del test, hasta 100 empresas, puestos 1..K sin saltos, sólo empresas de
test/<vertical>/empresas.parquet y sin repetidas.

Uso (después de modelo_final.py):  uv run chequear_predicciones.py
"""
import pandas as pd
ok = True
for V in ["d", "it", "j", "r", "rd"]:
    lic = pd.read_parquet(f"test/{V}/licitaciones.parquet"); emp = pd.read_parquet(f"test/{V}/empresas.parquet")
    rk = pd.read_csv(f"predicciones/{V}.csv")
    assert list(rk.columns) == ["id", "parent_uei", "puesto"], rk.columns
    faltan = set(lic.id) - set(rk.id); sobran = set(rk.id) - set(lic.id)
    fuera = ~rk.parent_uei.isin(set(emp.parent_uei))
    dup = rk.duplicated(["id", "parent_uei"]).sum()
    g = rk.groupby("id").puesto
    consecutivos = (g.apply(lambda s: sorted(s) == list(range(1, len(s) + 1)))).all()
    n = rk.groupby("id").size()
    print(f"{V}: {rk.id.nunique()}/{len(lic)} licitaciones, filas {len(rk)}, por licitación {n.min()}–{n.max()}, "
          f"faltan {len(faltan)}, sobran {len(sobran)}, empresas fuera {fuera.sum()}, duplicadas {dup}, puestos consecutivos {consecutivos}")
    ok &= not faltan and not sobran and fuera.sum() == 0 and dup == 0 and consecutivos and n.max() <= 100
print("TODO OK" if ok else "HAY PROBLEMAS")
