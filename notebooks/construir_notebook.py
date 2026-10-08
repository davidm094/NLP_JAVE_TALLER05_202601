"""Genera notebooks/Taller05_RNN.ipynb (se ejecuta luego con nbconvert)."""
from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s))
code = lambda s: C.append(nbf.v4.new_code_cell(s))

md("""# Taller 05 — Clasificación de texto en español con RNN

**Curso:** Procesamiento de Lenguaje Natural — Pontificia Universidad Javeriana
**Repositorio:** https://github.com/davidm094/NLP_JAVE_TALLER05_202601

Este notebook adapta el ejemplo de clase (*Text classification with an RNN*, TensorFlow) al dataset de salud mental en español
(`statement_es_es`). Se resuelven dos tareas:

* **Binaria:** Normal vs Depression.
* **Multiclase:** 7 clases (Anxiety, Bipolar, Depression, Normal, Personality disorder, Stress, Suicidal).

El código reutilizable está en `src/` y el diseño experimental en `DISENO_EXPERIMENTAL.md`. Aquí se (1) explora el dataset,
(2) se entrena en vivo la RNN base, y (3) se analizan las corridas del barrido de hiperparámetros guardadas en `results/`.""")

code("""import sys, json
from pathlib import Path
RAIZ = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(RAIZ / "src"))

import numpy as np, pandas as pd, tensorflow as tf
import matplotlib.pyplot as plt
from IPython.display import Image, display
from datos import cargar, split_binario, split_multiclase, TEXTO
from modelo import vectorizar, construir, optimizador
from entrenar import BASE
print("TensorFlow", tf.__version__)""")

md("## 1. Datos\nSolo se usa el texto en español. Hay 53.043 filas; 45 tienen el texto vacío.")
code("""df = cargar()
print(df.shape)
print(df.status.value_counts())
largo = df[TEXTO].fillna("").str.split().str.len()
print("\\nPalabras por texto:")
print(largo.describe(percentiles=[.5, .9, .95, .99]).round(1))""")

md("""## 2. Particiones
* **Binaria:** solo filas Normal y Depression (31.755). Test = 20 % (`random_state=42`); del 80 % restante se separa 10 % para validación.
* **Multiclase:** 7 clases, split estratificado 70 / 10 / 20.

La **validación** se usa para early stopping y para elegir hiperparámetros; el **test** solo se reporta.""")
code("""for nombre, f in [("binaria", split_binario), ("multiclase", split_multiclase)]:
    tr, va, te, clases = f()
    print(f"{nombre:10s} train={len(tr):6d} val={len(va):5d} test={len(te):6d}  clases={clases}")""")

md("""## 3. Codificación del texto (`TextVectorization`)
Responde a las preguntas del notebook de clase:

* **¿Qué es esto?** `TextVectorization` aprende un vocabulario (las `max_tokens` palabras más frecuentes **del train**) y convierte cada texto en
  una secuencia de índices enteros. El índice 0 es relleno (*padding*) y el 1 es `[UNK]` (palabra fuera del vocabulario).
* **¿La matriz tiene número fijo de filas y columnas?** Las filas = número de textos del lote. Las columnas: en el notebook original **no** son
  fijas (se rellena hasta el texto más largo de cada lote); aquí se fijan con `output_sequence_length=max_len` (128 en la base), truncando los
  textos largos y rellenando los cortos con ceros.

La estandarización se ajustó para español: además de minúsculas y puntuación, se quitan `¿ ¡ « »`.""")
code("""tr, va, te, clases = split_binario()
textos = [d[TEXTO].fillna("").astype(str).tolist() for d in (tr, va, te)]
enc, (Xtr, Xva, Xte) = vectorizar(textos[0], textos, vocab=BASE["vocab"], max_len=BASE["max_len"])
vocab = np.array(enc.get_vocabulary())
print("Primeros 20 tokens:", vocab[:20])
print("Forma de la matriz de train:", Xtr.shape)
ej = textos[0][0]
print("\\nOriginal :", ej[:300])
print("Ida y vuelta:", " ".join(vocab[Xtr[0]][Xtr[0] > 0])[:300])""")

md("""## 4. Modelo base
Misma arquitectura del ejemplo de clase: `Embedding → Bidirectional(LSTM 64) → Dense(64, relu) → Dense(salida)`, con `mask_zero=True`
para que la RNN ignore el relleno. Diferencias justificadas: vocabulario de 10.000 (en vez de 1.000), Adam con lr 1e-3 (en vez de 1e-4) y
early stopping sobre la pérdida de validación.""")
code("""print({k: BASE[k] for k in ["celda", "bidireccional", "vocab", "emb_dim", "unidades", "capas", "dropout", "dense", "lr", "batch", "max_len"]})
tf.keras.utils.set_random_seed(42)
modelo = construir(BASE, len(vocab), n_clases=2)
modelo.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
               optimizer=optimizador("adam", BASE["lr"]), metrics=["accuracy"])
modelo.build((None, Xtr.shape[1]))
modelo.summary()""")

md("## 5. Entrenamiento en vivo (binaria, semilla 42)\nTarda ~2–3 minutos en CPU.")
code("""hist = modelo.fit(Xtr, tr.y.values, validation_data=(Xva, va.y.values), epochs=20, batch_size=BASE["batch"],
                  callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)],
                  verbose=2)
perdida, acc = modelo.evaluate(Xte, te.y.values, verbose=0)
print(f"Test accuracy: {acc:.4f}")
fig, axs = plt.subplots(1, 2, figsize=(10, 3))
for ax, m in zip(axs, ["accuracy", "loss"]):
    ax.plot(hist.history[m], "-o", label=m); ax.plot(hist.history["val_" + m], "-o", label="val_" + m)
    ax.set_xlabel("Época"); ax.legend()
plt.show()""")

md("""La pérdida de validación toca su mínimo en la época 1–2 y luego sube mientras la de entrenamiento sigue bajando: **sobreajuste temprano**.
`restore_best_weights=True` devuelve los pesos de la mejor época.

### Predicción sobre textos nuevos
Salida > 0 ⇒ Depression; < 0 ⇒ Normal (logit).""")
code("""nuevos = ["hoy fue un buen día, salí con mis amigos y me reí mucho",
          "no tengo ganas de nada, me siento vacío y cansado todo el tiempo",
          "tengo que entregar el informe mañana y todavía no termino"]
logits = modelo.predict(enc(tf.constant(nuevos)), verbose=0)[:, 0]
for t, z in zip(nuevos, logits):
    print(f"{z:+.2f}  {'Depression' if z >= 0 else 'Normal':10s}  {t}")""")

md("""## 6. Diseño experimental y resultados del barrido
Las corridas se hicieron con `src/experimentos.py` y `src/correr.py` (una configuración por corrida) y quedaron en `results/`.

* **Base binaria:** la del punto 4, 3 semillas.
* **Base multiclase:** la misma arquitectura **con pesos de clase** (reducen el ruido entre semillas ~3×), 3 semillas, paciencia 1.
* **Factores:** 9 variaciones, una a la vez. Un efecto es "real" si |Δ F1 val| > 2σ (σ = desviación estándar entre semillas de la base).""")
code("""import importlib, figuras
importlib.reload(figuras)
b, m = figuras.cargar()
t = figuras.tabla_efectos(b, m)
t[["tarea", "etiqueta", "val_f1", "delta_pp", "delta_sigma", "veredicto", "test_acc", "tiempo_s"]].round(3)""")
code("""figuras.fig_efectos_pp(t); figuras.fig_efectos_sigma(t)
display(Image(str(RAIZ / "figs" / "efectos_por_factor.png")))
display(Image(str(RAIZ / "figs" / "binaria_vs_multiclase_sigma.png")))""")

md("### Ruido entre semillas y efecto de los pesos de clase")
code("""figuras.fig_semillas(b, m); figuras.fig_recall_clases(m)
display(Image(str(RAIZ / "figs" / "variabilidad_semillas.png")))
display(Image(str(RAIZ / "figs" / "recall_por_clase.png")))""")

md("### Base vs mejor combinación (3 semillas cada una)")
code("""figuras.resumen_mejor(b, m).round(4)""")
code("""figuras.fig_confusion("binaria", "base_s42", "confusion_binaria.png")
figuras.fig_confusion("multiclase", "mejor_s42", "confusion_multiclase.png")
display(Image(str(RAIZ / "figs" / "confusion_binaria.png")))
display(Image(str(RAIZ / "figs" / "confusion_multiclase.png")))""")

md("""## 7. Conclusiones
Ver el reporte PDF para la discusión completa. En resumen:

1. **Binaria:** problema fácil y casi insensible a los hiperparámetros (todos los efectos en ±0,8 pp). La base ya está en el techo (~0,948 de accuracy en test); la mejor combinación no la supera.
2. **Multiclase:** mucho más difícil (F1 macro ~0,62) y sensible: bidireccionalidad, celda con compuertas (LSTM/GRU) y vocabulario suficiente son determinantes.
3. **Desbalance:** sin pesos de clase la clase minoritaria (Personality disorder, 2 %) se detecta o no según la semilla; los pesos de clase lo estabilizan a costa de ~3,5 pp de accuracy.
4. **GRU** es la única variación que mejora (o no empeora) en ambas tareas y entrena más rápido que LSTM.""")

nb["cells"] = C
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = Path(__file__).with_name("Taller05_RNN.ipynb")
nbf.write(nb, out)
print("ok", out)
