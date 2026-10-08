"""Figuras y tablas del reporte final a partir de results/*.csv y results/histories/*.json.
Uso: python figuras.py
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from datos import RAIZ

RES, FIG = RAIZ / "results", RAIZ / "figs"
FIG.mkdir(exist_ok=True)
AZUL, NARANJA, AQUA, GRIS, TINTA, TINTA2 = "#2a78d6", "#eb6834", "#1baf7a", "#c9c8c2", "#2b2b2a", "#52514e"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#b5b4ae", "axes.labelcolor": TINTA, "xtick.color": TINTA2,
                     "ytick.color": TINTA2, "axes.grid": True, "grid.color": "#e6e5e0",
                     "grid.linewidth": 0.6, "axes.axisbelow": True, "figure.dpi": 160,
                     "savefig.bbox": "tight", "legend.frameon": False})

FACTORES = ["celda=gru", "celda=simplernn", "bidireccional=False", "unidades=128", "capas=2",
            "dropout=0.2", "lr=0.0003", "vocab=1000", "max_len=256"]
ETIQ = {"celda=gru": "GRU (vs LSTM)", "celda=simplernn": "SimpleRNN (vs LSTM)",
        "bidireccional=False": "Unidireccional", "unidades=128": "128 unidades (vs 64)",
        "capas=2": "2 capas (vs 1)", "dropout=0.2": "Dropout 0,2 (vs 0)",
        "lr=0.0003": "lr 3e-4 (vs 1e-3)", "vocab=1000": "Vocabulario 1.000 (vs 10.000)",
        "max_len=256": "256 tokens (vs 128)"}
CLASES_ES = {"Anxiety": "Ansiedad", "Bipolar": "Bipolar", "Depression": "Depresión", "Normal": "Normal",
             "Personality disorder": "T. personalidad", "Stress": "Estrés", "Suicidal": "Suicida"}


def cargar():
    b = pd.read_csv(RES / "resultados_binaria.csv")
    m = pd.read_csv(RES / "resultados_multiclase.csv")
    return b, m


def bases(b, m):
    """Devuelve dict tarea -> (df_base, prefijo de nombres de factores)."""
    return {"binaria": (b[b.grupo == "base"], ""),
            "multiclase": (m[m.nombre.str.startswith("class_weight")], "cw+")}


def tabla_efectos(b, m):
    filas = []
    for tarea, d in (("binaria", b), ("multiclase", m)):
        base, pre = bases(b, m)[tarea]
        mu, sd = base.val_f1_macro.mean(), base.val_f1_macro.std(ddof=1)
        mu_t = base.test_accuracy.mean()
        for f in FACTORES:
            r = d[d.nombre == pre + f]
            if r.empty:
                continue
            x = r.iloc[0]
            filas.append(dict(tarea=tarea, factor=f, etiqueta=ETIQ[f], val_f1=x.val_f1_macro,
                              delta_pp=(x.val_f1_macro - mu) * 100, delta_sigma=(x.val_f1_macro - mu) / sd,
                              test_acc=x.test_accuracy, test_f1=x.test_f1_macro,
                              delta_test_acc_pp=(x.test_accuracy - mu_t) * 100,
                              tiempo_s=x.tiempo_s, parametros=x.parametros, epocas=x.epocas_corridas,
                              base_mu=mu, base_sd=sd))
    t = pd.DataFrame(filas)
    t["veredicto"] = np.where(t.delta_sigma > 2, "mejora", np.where(t.delta_sigma < -2, "empeora", "ruido"))
    t.to_csv(RES / "tabla_efectos.csv", index=False)
    return t


def fig_efectos_pp(t):
    """Dos paneles (cada tarea con su escala): Δ F1 val en pp con banda ±2σ."""
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.8))
    orden = [ETIQ[f] for f in FACTORES][::-1]
    for ax, tarea, titulo in [(axs[0], "binaria", "Binaria (Normal vs Depresión)"),
                              (axs[1], "multiclase", "Multiclase (7 clases, con pesos de clase)")]:
        s = t[t.tarea == tarea].set_index("etiqueta").reindex(orden).dropna(subset=["delta_pp"])
        sd = s.base_sd.iloc[0] * 100
        ax.axvspan(-2 * sd, 2 * sd, color="#ecebe6", lw=0, zorder=0)
        col = [AZUL if v == "mejora" else NARANJA if v == "empeora" else GRIS for v in s.veredicto]
        ax.barh(range(len(s)), s.delta_pp, color=col, height=0.62, zorder=2)
        ax.axvline(0, color=TINTA, lw=0.8, zorder=3)
        ax.set_yticks(range(len(s)), s.index, fontsize=8)
        ax.set_xlabel("Δ F1 macro de validación vs. base (puntos porcentuales)", fontsize=8)
        ax.set_title(titulo, fontsize=9.5, loc="left", color=TINTA)
        ax.grid(axis="y", visible=False)
        for i, (v, d) in enumerate(zip(s.delta_pp, s.delta_sigma)):
            ax.text(v + (0.03 if v >= 0 else -0.03) * max(abs(s.delta_pp)), i, f"{v:+.2f}",
                    va="center", ha="left" if v >= 0 else "right", fontsize=7, color=TINTA2)
        lim = max(abs(s.delta_pp)) * 1.35
        ax.set_xlim(-lim, lim)
    fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in (AZUL, NARANJA, GRIS, "#ecebe6")],
               labels=["mejora > 2σ", "empeora > 2σ", "dentro del ruido", "banda ±2σ (ruido entre semillas)"],
               loc="upper center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIG / "efectos_por_factor.png")
    plt.close(fig)


def fig_efectos_sigma(t):
    """Misma escala (σ de cada tarea) para comparar binaria vs multiclase."""
    orden = [ETIQ[f] for f in FACTORES]
    p = t.pivot(index="etiqueta", columns="tarea", values="delta_sigma").reindex(orden)
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    x = np.arange(len(p))
    w = 0.38
    ax.axhspan(-2, 2, color="#ecebe6", lw=0, zorder=0)
    ax.bar(x - w / 2, p["binaria"], w, color=AZUL, label="Binaria", zorder=2)
    ax.bar(x + w / 2, p["multiclase"], w, color=NARANJA, label="Multiclase", zorder=2)
    ax.axhline(0, color=TINTA, lw=0.8)
    ax.set_xticks(x, [o.split(" (")[0] for o in orden], rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Efecto en desviaciones estándar\nentre semillas (σ)", fontsize=8)
    ax.grid(axis="x", visible=False)
    ax.legend(fontsize=8, loc="lower left")
    ax.text(len(p) - 0.5, 2.2, "banda ±2σ = ruido", ha="right", fontsize=7, color=TINTA2)
    fig.savefig(FIG / "binaria_vs_multiclase_sigma.png")
    plt.close(fig)


def fig_semillas(b, m):
    """Variabilidad entre semillas: base, pesos de clase y mejor combinación."""
    fig, axs = plt.subplots(1, 2, figsize=(9, 2.9))
    grupos = {"binaria": [("Base BiLSTM", b[b.grupo == "base"]), ("Mejor combinación", b[b.grupo == "mejor"])],
              "multiclase": [("Base sin pesos", m[m.grupo == "base"]),
                             ("Base con pesos de clase", m[m.nombre.str.startswith("class_weight")]),
                             ("Mejor combinación", m[m.grupo == "mejor"])]}
    for ax, (tarea, gs) in zip(axs, grupos.items()):
        for i, (n, s) in enumerate(gs):
            if s.empty:
                continue
            ax.plot([s.val_f1_macro.mean()] * 2, [i - 0.25, i + 0.25], color=TINTA, lw=1.6)
            ax.scatter(s.val_f1_macro, [i] * len(s), s=36, color=AZUL, edgecolor="white", lw=1, zorder=3)
        ax.set_yticks(range(len(gs)), [g[0] for g in gs], fontsize=8)
        ax.set_ylim(-0.6, len(gs) - 0.4)
        ax.set_xlabel("F1 macro de validación (cada punto = una semilla; raya = media)", fontsize=7.5)
        ax.set_title(tarea.capitalize(), fontsize=9.5, loc="left")
        ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(FIG / "variabilidad_semillas.png")
    plt.close(fig)


def fig_recall_clases(m):
    cols = [c for c in m.columns if c.startswith("test_recall_") and "macro" not in c]
    clases = [c.replace("test_recall_", "").replace("_", " ") for c in cols]
    grupos = [("Base sin pesos (3 sem.)", m[m.grupo == "base"], GRIS),
              ("Con pesos de clase (3 sem.)", m[m.nombre.str.startswith("class_weight")], AZUL),
              ("Mejor combinación (3 sem.)", m[m.grupo == "mejor"], NARANJA)]
    grupos = [g for g in grupos if not g[1].empty]
    orden = np.argsort(-m[m.grupo == "base"][cols].mean().values)
    fig, ax = plt.subplots(figsize=(9, 3.3))
    x = np.arange(len(cols))
    w = 0.8 / len(grupos)
    for i, (n, s, c) in enumerate(grupos):
        mu, sd = s[cols].mean().values[orden], s[cols].std(ddof=1).values[orden]
        ax.bar(x + (i - (len(grupos) - 1) / 2) * w, mu, w * 0.92, yerr=sd, color=c, label=n,
               error_kw=dict(lw=0.8, capsize=2, ecolor=TINTA2), zorder=2)
    ax.set_xticks(x, [CLASES_ES.get(clases[o], clases[o]) for o in orden], fontsize=8)
    ax.set_ylabel("Recall en test", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="x", visible=False)
    ax.legend(fontsize=7.5, ncol=3, loc="upper right", bbox_to_anchor=(1, 1.13))
    fig.savefig(FIG / "recall_por_clase.png")
    plt.close(fig)


def fig_confusion(tarea, nombre, archivo):
    h = json.load(open(RES / "histories" / f"{tarea}__{nombre}.json"))
    cm = np.array(h["cm_test"])
    cl = [CLASES_ES.get(c, c) for c in h["clases"]]
    cmn = cm / cm.sum(1, keepdims=True)
    n = len(cl)
    fig, ax = plt.subplots(figsize=(1.2 + 0.72 * n, 1.0 + 0.62 * n))
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{cmn[i, j]:.0%}\n{cm[i, j]}", ha="center", va="center", fontsize=6.5,
                    color="white" if cmn[i, j] > 0.55 else TINTA)
    ax.set_xticks(range(n), cl, rotation=35, ha="right", fontsize=7.5)
    ax.set_yticks(range(n), cl, fontsize=7.5)
    ax.set_xlabel("Clase predicha", fontsize=8)
    ax.set_ylabel("Clase real", fontsize=8)
    ax.grid(False)
    fig.savefig(FIG / archivo)
    plt.close(fig)


def fig_curvas():
    casos = [("binaria", "base_s42", "Binaria — base (paciencia 3)"),
             ("multiclase", "class_weight=True", "Multiclase — base con pesos (paciencia 1)"),
             ("multiclase", "mejor_s42", "Multiclase — mejor combinación")]
    casos = [c for c in casos if (RES / "histories" / f"{c[0]}__{c[1]}.json").exists()]
    fig, axs = plt.subplots(1, len(casos), figsize=(3.4 * len(casos), 2.7))
    for ax, (t, n, tit) in zip(np.atleast_1d(axs), casos):
        h = json.load(open(RES / "histories" / f"{t}__{n}.json"))["history"]
        e = np.arange(1, len(h["loss"]) + 1)
        ax.plot(e, h["loss"], "-o", ms=3.5, lw=1.8, color=AZUL, label="train")
        ax.plot(e, h["val_loss"], "-o", ms=3.5, lw=1.8, color=NARANJA, label="validación")
        best = int(np.argmin(h["val_loss"])) + 1
        ax.axvline(best, color=GRIS, lw=1, ls="--")
        ax.text(best, ax.get_ylim()[1] * 0.97, " mejor época", fontsize=7, color=TINTA2, va="top")
        ax.set_title(tit, fontsize=8.5, loc="left")
        ax.set_xlabel("Época", fontsize=8)
        ax.set_xticks(e)
    np.atleast_1d(axs)[0].set_ylabel("Pérdida (entropía cruzada)", fontsize=8)
    np.atleast_1d(axs)[0].legend(fontsize=7.5)
    fig.tight_layout()
    fig.savefig(FIG / "curvas_aprendizaje.png")
    plt.close(fig)


def fig_costo(t):
    fig, ax = plt.subplots(figsize=(8, 3.2))
    for tarea, c, mk in (("binaria", AZUL, "o"), ("multiclase", NARANJA, "s")):
        s = t[t.tarea == tarea]
        ax.scatter(s.tiempo_s, s.delta_pp, color=c, marker=mk, s=34, edgecolor="white", lw=0.8,
                   label=tarea.capitalize(), zorder=3)
        for _, r in s.iterrows():
            if abs(r.delta_sigma) > 2:
                ax.annotate(r.etiqueta.split(" (")[0], (r.tiempo_s, r.delta_pp), fontsize=6.5,
                            color=TINTA2, xytext=(4, 2), textcoords="offset points")
    ax.axhline(0, color=TINTA, lw=0.8)
    ax.set_xlabel("Tiempo de entrenamiento (s)", fontsize=8)
    ax.set_ylabel("Δ F1 macro val (pp)", fontsize=8)
    ax.legend(fontsize=8)
    fig.savefig(FIG / "costo_vs_efecto.png")
    plt.close(fig)


def resumen_mejor(b, m):
    filas = []
    for tarea, d, base in (("binaria", b, b[b.grupo == "base"]),
                           ("multiclase", m, m[m.nombre.str.startswith("class_weight")])):
        for n, s in (("base", base), ("mejor", d[d.grupo == "mejor"])):
            if s.empty:
                continue
            filas.append(dict(tarea=tarea, modelo=n, n=len(s),
                              **{f"{k}_media": s[k].mean() for k in
                                 ["val_f1_macro", "test_accuracy", "test_f1_macro", "tiempo_s"]},
                              **{f"{k}_de": s[k].std(ddof=1) for k in
                                 ["val_f1_macro", "test_accuracy", "test_f1_macro"]}))
    r = pd.DataFrame(filas)
    r.to_csv(RES / "resumen_base_vs_mejor.csv", index=False)
    return r


if __name__ == "__main__":
    b, m = cargar()
    t = tabla_efectos(b, m)
    print(t[["tarea", "factor", "delta_pp", "delta_sigma", "veredicto", "test_acc", "tiempo_s"]].round(3).to_string())
    print(resumen_mejor(b, m).round(4).to_string())
    fig_efectos_pp(t)
    fig_efectos_sigma(t)
    fig_semillas(b, m)
    fig_recall_clases(m)
    fig_curvas()
    fig_costo(t)
    fig_confusion("binaria", "base_s42", "confusion_binaria.png")
    mejor_m = "mejor_s42" if (RES / "histories" / "multiclase__mejor_s42.json").exists() else "class_weight=True"
    fig_confusion("multiclase", mejor_m, "confusion_multiclase.png")
    print("figuras ok")
