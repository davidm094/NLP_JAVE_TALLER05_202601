# Taller 05 — RNN para clasificación de texto en español (NLP, Javeriana)

Clasificación de textos de salud mental (`statement_es_es`) con redes recurrentes (LSTM, GRU, SimpleRNN), en dos tareas:
**binaria** (Normal vs Depression) y **multiclase** (7 clases). Incluye un diseño experimental de variación sistemática de hiperparámetros.

- `DISENO_EXPERIMENTAL.md` — variables, factores, niveles y etapas.
- `src/datos.py` — carga y particiones (train / validación / test).
- `src/modelo.py` — vectorización y construcción de la RNN.
- `src/entrenar.py` — entrena una configuración y guarda métricas.
- `src/experimentos.py` — define y corre los experimentos.
- `results/` — métricas por corrida (`resultados_<tarea>.csv`) e historias de entrenamiento.

- `notebooks/Taller05_RNN.ipynb` — notebook ejecutado: datos, codificación, entrenamiento en vivo y análisis.
- `Reporte_Taller05_RNN.pdf` — reporte escrito.
- `figs/` — figuras del reporte.

## Resultados principales
| Tarea | Base (3 semillas) | Mejor combinación (3 semillas) |
|---|---|---|
| Binaria (Normal vs Depression) | accuracy test 0,948 ± 0,002 | 0,948 ± 0,003 (sin mejora; 2,6× más lenta) |
| Multiclase (7 clases, pesos de clase) | F1 macro test 0,616 ± 0,009 | 0,631 ± 0,012 (no significativa con 3 semillas) |

- La binaria es casi insensible a los hiperparámetros (todos los efectos en ±0,8 pp de F1).
- En multiclase pesan la bidireccionalidad (−12 pp sin ella), la celda con compuertas (SimpleRNN −9 pp) y el vocabulario (1.000 palabras −8 pp).
- GRU es igual o mejor que LSTM en ambas tareas y más rápida.

## Cómo correrlo
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
gunzip -k data/Combined_Data_es-ES.csv.gz
cd src
python3 experimentos.py reducido binaria multiclase      # diseño reducido completo
python3 correr.py multiclase cw+celda=gru                  # una corrida puntual (multiclase con pesos de clase)
python3 figuras.py && python3 reporte.py                  # figuras y reporte PDF
```
