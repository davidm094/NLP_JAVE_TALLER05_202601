"""Métricas de sobreajuste por corrida, a partir de results/histories.
  mejor_ep: época con la menor pérdida de validación
  brecha:   accuracy train - accuracy val en esa época (pp)
  subida:   cuánto sube la pérdida de val desde su mínimo hasta la última época (%)"""
import json, sys
import pandas as pd
from datos import RAIZ

def metricas(tarea, nombre):
    h = json.load(open(RAIZ / "results" / "histories" / f"{tarea}__{nombre}.json"))["history"]
    vl = h["val_loss"]; i = min(range(len(vl)), key=vl.__getitem__)
    return dict(mejor_ep=i + 1, epocas=len(vl), val_loss_min=vl[i],
                brecha_pp=(h["accuracy"][i] - h["val_accuracy"][i]) * 100,
                subida_pct=(vl[-1] / vl[i] - 1) * 100)

if __name__ == "__main__":
    tarea = sys.argv[1]
    d = pd.read_csv(RAIZ / "results" / f"resultados_{tarea}.csv")
    filas = [dict(nombre=n, val_f1=r.val_f1_macro, test_acc=r.test_accuracy, params=r.parametros, t=r.tiempo_s,
                  **metricas(tarea, n)) for n, r in zip(d.nombre, d.itertuples()) if n in sys.argv[2:]]
    print(pd.DataFrame(filas).round(3).to_string(index=False))
