"""Corre UNA configuración en primer plano.  python correr.py <tarea> <nombre> [semilla]
Multiclase usa paciencia=1 (decisión 7-oct: el mejor modelo aparece en época 1-2)."""
import sys
from experimentos import REDUCIDO
from entrenar import entrenar
tarea, nombre = sys.argv[1], sys.argv[2]
semilla = int(sys.argv[3]) if len(sys.argv) > 3 else 42
cat = {n: (c, g) for n, c, g in REDUCIDO}
# Etapa 3: mejor combinación por tarea (regla: factores con Δ F1 val > 0 en esa tarea;
# en binaria los tres superan 2σ; en multiclase ninguno lo supera solo, combinación exploratoria).
MEJOR = {"binaria": {"celda": "gru", "capas": 2, "lr": 3e-4},
         "multiclase": {"celda": "gru", "max_len": 256, "dropout": 0.2, "class_weight": True}}
cat["mejor"] = (MEJOR[tarea], "mejor")
cat["class_weight=True"] = ({"class_weight": True}, "class_weight")
cw = nombre.startswith("cw+")          # base multiclase con pesos de clase (decisión 7-oct)
clave = nombre[3:] if cw else nombre
base_nombre = clave.split("_s")[0] if "_s" in clave and not clave.startswith("base") else clave
if base_nombre.startswith("base"):
    cambios, grupo = {}, "base"
else:
    cambios, grupo = cat[base_nombre]
if cw:
    cambios = {**cambios, "class_weight": True}
if tarea == "multiclase":
    cambios = {**cambios, "paciencia": 1}
entrenar(nombre, cambios, tarea=tarea, grupo=grupo, semilla=semilla)
