"""Barrido de hiperparámetros: un factor a la vez alrededor de BASE.

Uso:  python experimentos.py barrido     (réplica + base x3 semillas + factores)
      python experimentos.py final       (mejor combinación, binaria y 7 clases)
Los resultados se agregan a results/resultados_<tarea>.csv; si una corrida ya
existe se salta, así que el script se puede reanudar.
"""
import json
import sys

from entrenar import RES, entrenar

# Réplica del notebook de ejemplo (vocab 1000, Adam 1e-4, 10 épocas fijas).
# Única diferencia: se trunca a 512 tokens (el notebook no trunca) para que
# sea viable en CPU; 512 cubre ~99 % de los textos.
REPLICA = dict(vocab=1000, lr=1e-4, epocas=10, early_stopping=False, max_len=512)

FACTORES = {
    "celda": ["gru", "simplernn"],
    "bidireccional": [False],
    "vocab": [1000, 5000, 20000, 40000],
    "emb_dim": [16, 32, 128, 256],
    "unidades": [16, 32, 128, 256],
    "capas": [2, 3],
    "dropout": [0.2, 0.5],
    "dense": [0, 16, 128],
    "dense_act": ["tanh"],
    "rec_dropout": [0.2],
    "l2": [1e-4, 1e-3],
    "clipnorm": [1.0],
    "merge": ["sum"],
    "paciencia": [1, 5],
    "lr": [1e-4, 3e-4, 3e-3, 1e-2],
    "batch": [32, 128, 256],
    "max_len": [32, 64, 256, 512],
    "optimizador": [("rmsprop", 1e-3), ("sgd", 1e-2)],
    "early_stopping": [False],
}


def barrido(tarea="binaria"):
    for s in (42, 7, 123):
        entrenar(f"base_s{s}", {}, tarea=tarea, grupo="base", semilla=s)
    for f, valores in FACTORES.items():
        for v in valores:
            if f == "optimizador":
                cambios, etiqueta = {"optimizador": v[0], "lr": v[1]}, v[0]
            elif f == "early_stopping":
                cambios, etiqueta = {"early_stopping": False, "epocas": 20}, "sin_es_20ep"
            else:
                cambios, etiqueta = {f: v}, v
            entrenar(f"{f}={etiqueta}", cambios, tarea=tarea, grupo=f)
    if tarea == "multiclase":  # solo aplica con clases desbalanceadas
        entrenar("class_weight=True", {"class_weight": True}, tarea=tarea, grupo="class_weight")
    # La más lenta (512 tokens, ~25 min) va al final.
    entrenar("replica_notebook", REPLICA, tarea=tarea, grupo="replica")


MVP = [("gru", {"celda": "gru"}, "celda"), ("simplernn", {"celda": "simplernn"}, "celda"),
       ("bidireccional=False", {"bidireccional": False}, "bidireccional"),
       ("unidades=128", {"unidades": 128}, "unidades"),
       ("dropout=0.2", {"dropout": 0.2}, "dropout"),
       ("lr=0.003", {"lr": 3e-3}, "lr"), ("max_len=256", {"max_len": 256}, "max_len")]


def mvp():
    """7 corridas exploratorias (una por familia); coinciden con nombres del barrido completo."""
    for etiqueta, cambios, grupo in MVP:
        nombre = etiqueta if "=" in etiqueta else f"celda={etiqueta}"
        entrenar(nombre, cambios, grupo=grupo)


# Diseño reducido: un nivel alternativo por factor, los de mayor relevancia esperada.
REDUCIDO = [
    ("celda=gru", {"celda": "gru"}, "celda"),
    ("celda=simplernn", {"celda": "simplernn"}, "celda"),
    ("bidireccional=False", {"bidireccional": False}, "bidireccional"),
    ("unidades=128", {"unidades": 128}, "unidades"),
    ("capas=2", {"capas": 2}, "capas"),
    ("dropout=0.2", {"dropout": 0.2}, "dropout"),
    ("lr=0.0003", {"lr": 3e-4}, "lr"),
    ("vocab=1000", {"vocab": 1000}, "vocab"),
    ("max_len=256", {"max_len": 256}, "max_len"),
]


def reducido(tarea):
    for s in (42, 7, 123):
        entrenar(f"base_s{s}", {}, tarea=tarea, grupo="base", semilla=s)
    for nombre, cambios, grupo in REDUCIDO:
        entrenar(nombre, cambios, tarea=tarea, grupo=grupo)
    if tarea == "multiclase":
        entrenar("class_weight=True", {"class_weight": True}, tarea=tarea, grupo="class_weight")


def final():
    mejor = json.load(open(RES / "mejor_config.json"))
    for s in (42, 7, 123):
        entrenar(f"mejor_s{s}", mejor, grupo="mejor", semilla=s, guardar_modelo=(s == 42))
    entrenar("base", {}, tarea="multiclase", grupo="base")
    entrenar("mejor", mejor, tarea="multiclase", grupo="mejor", guardar_modelo=True)
    entrenar("mejor+class_weight", {**mejor, "class_weight": True}, tarea="multiclase",
             grupo="mejor", guardar_modelo=True)


if __name__ == "__main__":
    modo = sys.argv[1]
    if modo == "solo":  # python experimentos.py solo <tarea> <nombre> [<nombre> ...]
        tarea, pedidos = sys.argv[2], set(sys.argv[3:])
        catalogo = {f"base_s{s}": ({}, "base", s) for s in (42, 7, 123)}
        catalogo.update({n: (c, g, 42) for n, c, g in REDUCIDO})
        catalogo["class_weight=True"] = ({"class_weight": True}, "class_weight", 42)
        for n in sys.argv[3:]:
            c, g, s = catalogo[n]
            entrenar(n, c, tarea=tarea, grupo=g, semilla=s)
    elif modo == "reducido":
        for t in sys.argv[2:] or ["binaria", "multiclase"]:
            reducido(t)
    elif modo == "barrido":
        barrido(sys.argv[2] if len(sys.argv) > 2 else "binaria")
    else:
        {"mvp": mvp, "final": final}[modo]()
