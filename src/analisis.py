"""Tablas y figuras del análisis comparativo. Uso: python analisis.py [elegir]"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from datos import RAIZ
from entrenar import BASE, RES

FIG = RAIZ / "figs"
FIG.mkdir(exist_ok=True)
AZUL, NARANJA, AQUA, GRIS, TINTA = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984", "#2b2b2a"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#b5b4ae", "axes.labelcolor": TINTA,
                     "xtick.color": "#52514e", "ytick.color": "#52514e",
                     "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6,
                     "figure.dpi": 150, "savefig.bbox": "tight"})

ORDEN = ["celda", "bidireccional", "vocab", "emb_dim", "unidades", "capas", "dropout",
         "dense", "dense_act", "rec_dropout", "l2", "clipnorm", "merge", "lr", "batch", "max_len",
         "optimizador", "early_stopping", "paciencia"]
NOMBRE = {"celda": "Tipo de celda", "bidireccional": "Bidireccional", "vocab": "Tamaño del vocabulario",
          "emb_dim": "Dimensión del embedding", "unidades": "Unidades por capa RNN",
          "capas": "Capas RNN apiladas", "dropout": "Dropout", "dense": "Unidades capa densa", "dense_act": "Activación capa densa",
          "rec_dropout": "Dropout recurrente", "l2": "Regularización L2", "clipnorm": "Recorte de gradiente",
          "merge": "Combinación bidireccional", "paciencia": "Paciencia early stopping",
          "lr": "Learning rate", "batch": "Batch size", "max_len": "Longitud máx. (tokens)",
          "optimizador": "Optimizador", "early_stopping": "Early stopping"}


def cargar(tarea="binaria"):
    return pd.read_csv(RES / f"resultados_{tarea}.csv")


def ruido_base(df):
    b = df[df.grupo == "base"]
    return b, {m: (b[m].mean(), b[m].std(ddof=1)) for m in
               ["val_f1_macro", "test_accuracy", "test_f1_macro", "tiempo_s", "s_por_epoca"]}


def valor(fila, f):
    if f == "optimizador":
        return f"{fila.optimizador} (lr={fila.lr:g})"
    if f == "early_stopping":
        return "sí (pac. 3)" if fila.early_stopping else "no (20 ép.)"
    v = fila[f]
    return f"{v:g}" if isinstance(v, (float, np.floating)) else str(v)


def tabla_factores(df):
    b, r = ruido_base(df)
    filas = []
    base_row = b.iloc[0]
    for f in ORDEN:
        sub = df[df.grupo == f]
        filas.append(dict(factor=NOMBRE[f], valor=valor(base_row, f) + " (base)",
                          val_f1=r["val_f1_macro"][0], test_acc=r["test_accuracy"][0],
                          test_f1=r["test_f1_macro"][0], delta_val=0.0,
                          parametros=base_row.parametros, epocas=b.epocas_corridas.mean(),
                          tiempo_s=r["tiempo_s"][0], es_base=True))
        for _, x in sub.iterrows():
            filas.append(dict(factor=NOMBRE[f], valor=valor(x, f), val_f1=x.val_f1_macro,
                              test_acc=x.test_accuracy, test_f1=x.test_f1_macro,
                              delta_val=x.val_f1_macro - r["val_f1_macro"][0],
                              parametros=x.parametros, epocas=x.epocas_corridas,
                              tiempo_s=x.tiempo_s, es_base=False))
    t = pd.DataFrame(filas)
    t["significativo"] = (~t.es_base) & (t.delta_val.abs() > 2 * r["val_f1_macro"][1])
    return t


def elegir_mejor(df):
    """Por cada factor, se adopta el valor con mayor F1 macro de VALIDACIÓN solo si
    supera a la base por más de 2 desviaciones estándar entre semillas."""
    _, r = ruido_base(df)
    mu, sd = r["val_f1_macro"]
    mejor, decisiones = {}, []
    for f in ORDEN:
        sub = df[df.grupo == f]
        if sub.empty:
            continue
        x = sub.loc[sub.val_f1_macro.idxmax()]
        gana = x.val_f1_macro - mu > 2 * sd
        if gana and f not in ("early_stopping",):
            if f == "optimizador":
                mejor.update(optimizador=x.optimizador, lr=float(x.lr))
            else:
                v = x[f]
                mejor[f] = v.item() if hasattr(v, "item") else v
        decisiones.append(dict(factor=f, mejor_valor=valor(x, f), val_f1=x.val_f1_macro,
                               delta=x.val_f1_macro - mu, umbral=2 * sd, adoptado=bool(gana)))
    for k in ("bidireccional", "early_stopping", "class_weight"):
        if k in mejor:
            mejor[k] = bool(mejor[k])
    json.dump(mejor, open(RES / "mejor_config.json", "w"), indent=2)
    pd.DataFrame(decisiones).to_csv(RES / "decisiones_mejor_config.csv", index=False)
    return mejor, pd.DataFrame(decisiones)


def fig_factores(df):
    b, r = ruido_base(df)
    mu, sd = r["val_f1_macro"]
    fs = [f for f in ORDEN if (df.grupo == f).any()]
    n = len(fs)
    cols = 5
    fig, axs = plt.subplots(int(np.ceil(n / cols)), cols, figsize=(11, 2.3 * np.ceil(n / cols)),
                            sharey=True)
    axs = axs.ravel()
    for ax, f in zip(axs, fs):
        sub = pd.concat([b.iloc[[0]].assign(val_f1_macro=mu, test_f1_macro=r["test_f1_macro"][0]),
                         df[df.grupo == f]])
        etiquetas = [valor(x, f) for _, x in sub.iterrows()]
        if f in ("vocab", "emb_dim", "unidades", "capas", "dropout", "dense", "rec_dropout", "l2", "clipnorm", "paciencia", "lr", "batch", "max_len"):
            orden = np.argsort(sub[f].astype(float).values)
        else:
            orden = np.arange(len(sub))
        x = np.arange(len(sub))
        ax.axhspan(mu - 2 * sd, mu + 2 * sd, color="#e6e5e0", zorder=0, lw=0)
        ax.plot(x, sub.val_f1_macro.values[orden], "-o", color=AZUL, ms=4, lw=1.6, label="F1 macro validación")
        ax.plot(x, sub.test_f1_macro.values[orden], "-o", color=NARANJA, ms=4, lw=1.6, label="F1 macro test")
        base_pos = list(orden).index(0)
        ax.plot([base_pos], [mu], "s", color=TINTA, ms=5, zorder=5)
        ax.set_xticks(x, [etiquetas[i] for i in orden], rotation=30 if len(sub) > 3 else 0,
                      ha="right" if len(sub) > 3 else "center", fontsize=7.5)
        ax.set_title(NOMBRE[f], fontsize=9, color=TINTA, loc="left")
        ax.grid(axis="x", visible=False)
    for ax in axs[n:]:
        ax.axis("off")
    axs[0].set_ylabel("F1 macro")
    axs[cols].set_ylabel("F1 macro") if n > cols else None
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h + [plt.Rectangle((0, 0), 1, 1, color="#e6e5e0"),
                    plt.Line2D([], [], marker="s", ls="", color=TINTA)],
               l + ["base ± 2σ entre semillas", "configuración base"],
               loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(FIG / "factores_f1.png")
    plt.close(fig)


def fig_ranking(t):
    t = t[~t.es_base].sort_values("delta_val")
    fig, ax = plt.subplots(figsize=(7, 0.19 * len(t) + 1))
    col = [AZUL if s and d > 0 else NARANJA if s else "#c9c8c2" for s, d in zip(t.significativo, t.delta_val)]
    ax.barh(np.arange(len(t)), t.delta_val * 100, color=col, height=0.7)
    ax.set_yticks(np.arange(len(t)), [f"{a}: {b}" for a, b in zip(t.factor, t.valor)], fontsize=7)
    ax.axvline(0, color=TINTA, lw=0.8)
    ax.set_xlabel("Δ F1 macro de validación vs. base (puntos porcentuales)")
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in (AZUL, NARANJA, "#c9c8c2")],
              labels=["mejora > 2σ", "empeora > 2σ", "dentro del ruido"], frameon=False, fontsize=7, loc="lower right")
    fig.savefig(FIG / "ranking_delta.png")
    plt.close(fig)


def fig_costo(df):
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    d = df[df.grupo != "replica"]
    for ax, col, lab, logx in [(axs[0], "parametros", "Parámetros entrenables", True),
                               (axs[1], "tiempo_s", "Tiempo de entrenamiento (s)", True)]:
        ax.scatter(d[col], d.test_f1_macro, s=22, color=AZUL, edgecolor="white", lw=0.8, zorder=3)
        for _, x in d.iterrows():
            if x.test_f1_macro < d.test_f1_macro.quantile(0.15) or x[col] == d[col].max() or x.val_f1_macro == d.val_f1_macro.max():
                ax.annotate(x.nombre, (x[col], x.test_f1_macro), fontsize=6.5, color="#52514e",
                            xytext=(4, 2), textcoords="offset points")
        ax.set_xscale("log" if logx else "linear")
        ax.set_xlabel(lab)
    axs[0].set_ylabel("F1 macro test")
    fig.tight_layout()
    fig.savefig(FIG / "costo_vs_desempeno.png")
    plt.close(fig)


def fig_curvas(nombres, tarea="binaria", archivo="curvas.png"):
    fig, axs = plt.subplots(1, len(nombres), figsize=(3.4 * len(nombres), 2.8), sharey=True)
    for ax, (n, titulo) in zip(np.atleast_1d(axs), nombres):
        h = json.load(open(RES / "histories" / f"{tarea}__{n}.json"))["history"]
        e = np.arange(1, len(h["loss"]) + 1)
        ax.plot(e, h["loss"], "-o", ms=3, color=AZUL, label="pérdida train")
        ax.plot(e, h["val_loss"], "-o", ms=3, color=NARANJA, label="pérdida validación")
        ax.set_title(titulo, fontsize=9, loc="left")
        ax.set_xlabel("Época")
    np.atleast_1d(axs)[0].set_ylabel("Pérdida (entropía cruzada)")
    np.atleast_1d(axs)[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / archivo)
    plt.close(fig)


def fig_confusion(n, tarea, archivo):
    h = json.load(open(RES / "histories" / f"{tarea}__{n}.json"))
    cm = np.array(h["cm_test"]); cl = h["clases"]
    cmn = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(1.0 + 0.75 * len(cl), 0.8 + 0.7 * len(cl)))
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(cl)):
        for j in range(len(cl)):
            ax.text(j, i, f"{cmn[i, j]:.0%}\n({cm[i, j]})", ha="center", va="center", fontsize=6.5,
                    color="white" if cmn[i, j] > 0.55 else TINTA)
    ax.set_xticks(range(len(cl)), cl, rotation=35, ha="right", fontsize=7)
    ax.set_yticks(range(len(cl)), cl, fontsize=7)
    ax.set_xlabel("Predicho"); ax.set_ylabel("Real"); ax.grid(False)
    fig.savefig(FIG / archivo)
    plt.close(fig)


if __name__ == "__main__":
    df = cargar()
    t = tabla_factores(df)
    t.to_csv(RES / "tabla_factores_binaria.csv", index=False)
    print(t.to_string())
    fig_factores(df); fig_ranking(t); fig_costo(df)
    if len(sys.argv) > 1 and sys.argv[1] == "elegir":
        print(elegir_mejor(df))
