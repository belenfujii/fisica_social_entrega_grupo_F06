# Y-Hat: ¿quién gana la licitación? Un modelo de Boltzmann global

Investigathon 2026 · Track Física Social · Grupo F06

Para cada licitación, el modelo ordena a las empresas candidatas según su probabilidad de
ganarla, usando solo información anterior a la publicación. La elección del ganador se modela
como una distribución de Boltzmann sobre las candidatas,

    P(e | q) = exp(-E(q,e)) / Z(q),      -E(q,e) = Σ_k β_k f_k(q,e)

La energía tiene 44 pistas agrupadas en cuatro ideas: memoria (el B0), texto (TF-IDF y
variantes), nombre de la empresa en el aviso y red oficina–empresa. Los pesos son **un único
juego para todas las verticales**, elegido con un lasso que separa un tronco común de
correcciones por vertical (`tronco.py`). Para rankear se usa solo el tronco, así que el modelo
no necesita saber la vertical y sirve igual para una vertical nueva.

Resultados en 2024S1 (prueba): MRR medio 0,282 contra 0,120 del B0. El ganador queda primero
en el 22 % de las licitaciones (8 % con el B0). Detalle en `informe/`.

## Instalación

```
uv sync
```

Los datos van en `data/`, igual que en el repo original (`data/<vertical>/licitaciones.parquet`,
`historial.parquet`, `empresas.parquet`, `ganadores.parquet`, y `data/psc.parquet`,
`naics.parquet`, `office.parquet`).

## Reproducir las predicciones del test (lo que se evalúa)

1. Bajar los datos y el test en la raíz del repo (releases `datos-v1` y `test-v1` del repo de la cátedra):

```
curl -LO https://github.com/bravoivanagustin/ranking-licitaciones/releases/download/datos-v1/data.zip
curl -LO https://github.com/bravoivanagustin/ranking-licitaciones/releases/download/test-v1/test.zip
unzip data.zip
unzip test.zip
```

2. Correr el modelo (unos 15 minutos):

```
uv run modelo_final.py --vertical todas --modo entregar --nuevas test --salida-dir predicciones
```

Escribe `predicciones/d.csv`, `it.csv`, `j.csv`, `r.csv` y `rd.csv` (columnas `id, parent_uei, puesto`,
hasta 100 empresas por licitación) y `predicciones/entrega_beta_global.csv` con los pesos del modelo.
El modelo se entrena con todos los semestres con ganadores de las cinco verticales (2022S2 a 2024S1).
Como `test/<vertical>/empresas.parquet` es una sola foto sin columna `semestre`, todas las
licitaciones del test se rankean contra esa tabla y sus pistas usan solo información anterior al
corte (1/7/2024). El B0 usa la fecha real de cada licitación.

3. Chequear el formato:

```
uv run chequear_predicciones.py
```

Ensayo con datos con respuesta: tratando 2024S1 como si fuera el test (mismo formato), el
modelo da MRR d 0,345, it 0,245, j 0,285, r 0,343, rd 0,192, y `evaluar.py` acepta los archivos.

## Qué hay en cada archivo

| Archivo | Qué hace |
|---|---|
| `modelo_final.py` | Programa principal: validación (escalera, modelos nulos) y entrega. |
| `tronco.py` | Ajuste del Boltzmann con tronco común por lasso (el modelo del informe). |
| `boltzmann_seleccion.py` | Cálculo de las pistas (BM25, temas, red, correspondencias, entropía) y el Boltzmann base. |
| `modelo_v2.py` | Datos, candidatas preseleccionadas (pool) y pistas de memoria y texto. |
| `pista_nombre.py` | Mención y rareza del nombre de la empresa en el aviso. |
| `b0.py`, `evaluar.py` | Modelo de referencia y evaluación oficiales (sin cambios). |
| `experimentos/` | Experimentos del informe (ver abajo). |
| `resultados/` | Tablas que salen de los experimentos. |
| `informe/` | Informe en LaTeX. |

Experimentos (correr desde la raíz, por ejemplo `uv run experimentos/exp_lasso.py`):

| Script | Sección del informe |
|---|---|
| `exp_universal.py --preparar <v>` | Prepara las candidatas de cada vertical (correr primero, una vez por vertical). |
| `exp_lasso.py` | Camino del lasso, tronco común y correcciones por vertical. |
| `exp_ablacion.py` | Sacar una idea entera del modelo y correlación entre pistas. |
| `exp_empresas_falsas.py` | Modelo nulo de empresas falsas. |
| `exp_bayes.py` | Modelo bayesiano jerárquico (cuánto agrega la vertical). |
| `exp_caja_negra.py` | Comparación con LambdaMART (necesita lightgbm). |
| `modelo_final.py --vertical <v> --modo validar` | Escalera y modelos nulos de término y de ganadores. |

## Cuidados contra la fuga de información

Todas las pistas usan información anterior al semestre de la licitación (el B0, anterior a su
fecha). Los hiperparámetros se eligen con dos ventanas de validación (2022S2→2023S1 y
2022S2+2023S1→2023S2), y la estandarización se ajusta solo con entrenamiento. 2024S1 se usó
como prueba.
