# Taller 05 — RNN para clasificación de texto en español (NLP, Javeriana)

Clasificación de textos de salud mental (`statement_es_es`) con redes recurrentes (LSTM, GRU, SimpleRNN), en dos tareas:
**binaria** (Normal vs Depression) y **multiclase** (7 clases). Incluye un diseño experimental de variación sistemática de hiperparámetros.

- `DISENO_EXPERIMENTAL.md` — variables, factores, niveles y etapas.
- `src/datos.py` — carga y particiones (train / validación / test).
- `src/modelo.py` — vectorización y construcción de la RNN.
- `src/entrenar.py` — entrena una configuración y guarda métricas.
- `src/experimentos.py` — define y corre los experimentos.
- `results/` — métricas por corrida (`resultados_<tarea>.csv`) e historias de entrenamiento.

## Cómo correrlo
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
gunzip -k data/Combined_Data_es-ES.csv.gz
cd src
python3 experimentos.py reducido binaria multiclase      # diseño reducido completo
python3 experimentos.py solo multiclase celda=gru         # una corrida puntual
```
