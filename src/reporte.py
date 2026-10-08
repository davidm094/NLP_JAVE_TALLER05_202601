"""Reporte final (PDF). Todas las cifras se leen de results/. Uso: python reporte.py [salida.pdf]"""
import sys
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (CondPageBreak, Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer,
                                Table, TableStyle)
from PIL import Image as PILImage
from scipy import stats

import figuras as F
from datos import split_binario, split_multiclase

RAIZ = Path(__file__).resolve().parents[1]
SALIDA = Path(sys.argv[1]) if len(sys.argv) > 1 else RAIZ / "Reporte_Taller05_RNN.pdf"
REPO = "https://github.com/davidm094/NLP_JAVE_TALLER05_202601"
D = "/usr/share/fonts/truetype/dejavu/"
pdfmetrics.registerFont(TTFont("F", D + "DejaVuSans.ttf"))
pdfmetrics.registerFont(TTFont("FB", D + "DejaVuSans-Bold.ttf"))
pdfmetrics.registerFont(TTFont("FI", D + "DejaVuSans-Oblique.ttf"))

st = getSampleStyleSheet()
N = ParagraphStyle("N", parent=st["Normal"], fontName="F", fontSize=9.2, leading=13, spaceAfter=4)
H1 = ParagraphStyle("H1", parent=N, fontName="FB", fontSize=13, leading=17, spaceBefore=12, spaceAfter=5)
H2 = ParagraphStyle("H2", parent=N, fontName="FB", fontSize=10.5, leading=14, spaceBefore=8, spaceAfter=3)
T = ParagraphStyle("T", parent=N, fontName="FB", fontSize=18, leading=23, spaceAfter=4)
S = ParagraphStyle("S", parent=N, fontSize=8.3, leading=11, textColor=colors.HexColor("#52514e"))
C = ParagraphStyle("C", parent=N, fontSize=7.8, leading=9.6, spaceAfter=0)
CB = ParagraphStyle("CB", parent=C, fontName="FB")
B = ParagraphStyle("B", parent=N, leftIndent=10, bulletIndent=0)

p = lambda t, s=N: Paragraph(t, s)
f4 = lambda x: f"{x:.4f}".replace(".", ",")
f3 = lambda x: f"{x:.3f}".replace(".", ",")
pp = lambda x: f"{x:+.2f} pp".replace(".", ",")
sg = lambda x: f"{x:+.1f}σ".replace(".", ",")
fc = lambda x: f"{x:.2f}".replace(".", ",")
mil = lambda x: f"{int(x):,}".replace(",", ".")


def bullet(t):
    return Paragraph(t, B, bulletText="•")


def tabla(filas, anchos, cab=1, resaltar=None):
    data = [[p(str(c), CB if i < cab else C) for c in fila] for i, fila in enumerate(filas)]
    t = Table(data, colWidths=[a * cm for a in anchos], repeatRows=cab)
    estilo = [("BACKGROUND", (0, 0), (-1, cab - 1), colors.HexColor("#ecebe6")),
              ("LINEBELOW", (0, 0), (-1, -1), 0.3, colors.HexColor("#c9c8c2")),
              ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
              ("TOPPADDING", (0, 0), (-1, -1), 2.2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.2)]
    for r in resaltar or []:
        estilo.append(("BACKGROUND", (0, r), (-1, r), colors.HexColor("#eef4fc")))
    t.setStyle(TableStyle(estilo))
    return t


def fig(nombre, ancho_cm, pie):
    ruta = RAIZ / "figs" / nombre
    w, h = PILImage.open(ruta).size
    return KeepTogether([Image(str(ruta), width=ancho_cm * cm, height=ancho_cm * cm * h / w), p(pie, S)])


# ------------------------------------------------------------------ datos
b, m = F.cargar()
t = F.tabla_efectos(b, m)
res = F.resumen_mejor(b, m)
F.fig_efectos_pp(t); F.fig_efectos_sigma(t); F.fig_semillas(b, m); F.fig_recall_clases(m)
F.fig_curvas(); F.fig_costo(t)
F.fig_confusion("binaria", "base_s42", "confusion_binaria.png")
F.fig_confusion("multiclase", "mejor_s42", "confusion_multiclase.png")

trb, vab, teb, _ = split_binario()
trm, vam, tem, clases_m = split_multiclase()
bb = b[b.grupo == "base"]
mb0 = m[m.grupo == "base"]
mcw = m[m.nombre.str.startswith("class_weight")]
bm = b[b.grupo == "mejor"]
mm = m[m.grupo == "mejor"]
ms = lambda s, k: (s[k].mean(), s[k].std(ddof=1))
E = {(r.tarea, r.factor): r for r in t.itertuples()}
n_corridas = len(b) + len(m)


def ef(tarea, f):
    r = E[(tarea, f)]
    return f"{pp(r.delta_pp)} ({sg(r.delta_sigma)})"


cuerpo = []
add = cuerpo.append

# ------------------------------------------------------------------ portada / resumen
add(p("Clasificación de texto en español con redes neuronales recurrentes", T))
add(p("Taller 05 · Procesamiento de Lenguaje Natural · Pontificia Universidad Javeriana · octubre de 2026", S))
add(p(f"Código y resultados: <link href='{REPO}' color='#2a78d6'><u>{REPO}</u></link>", S))
add(Spacer(1, 6))
add(p("Resumen", H2))
add(p("Se entrenaron redes recurrentes (LSTM, GRU y SimpleRNN) sobre 53.043 textos de salud mental "
      "en español para dos tareas: <b>binaria</b> (Normal vs Depression) y <b>multiclase</b> (7 clases). Se identificaron 19 hiperparámetros, "
      f"se variaron 9 de forma sistemática (uno a la vez, alrededor de una configuración base) y se midió el ruido entre semillas para separar "
      f"efectos reales del azar. En total se hicieron {n_corridas} entrenamientos. "
      f"<b>La tarea binaria es fácil e insensible a los hiperparámetros</b>: la base alcanza {f3(ms(bb,'test_accuracy')[0])} de accuracy en test "
      f"y ningún cambio la mueve más de 0,8 pp. <b>La multiclase es difícil y sensible</b>: quitar la bidireccionalidad, usar SimpleRNN o "
      f"reducir el vocabulario cuestan entre 8 y 12 pp de F1 macro. El desbalance de clases hace que, sin pesos de clase, el resultado dependa "
      f"de la semilla; con pesos de clase el ruido baja ~{fc(ms(mb0,'val_f1_macro')[1]/ms(mcw,'val_f1_macro')[1])[:-1]}× y la clase minoritaria "
      f"pasa de detectarse {f3(mb0['test_recall_Personality_disorder'].mean())} a {f3(mcw['test_recall_Personality_disorder'].mean())} (recall medio)."))

# ------------------------------------------------------------------ 1 datos
add(CondPageBreak(4 * cm)); add(p("1. Datos y particiones", H1))
add(p("Se usa únicamente la columna <b>statement_es_es</b> (texto en español) y la etiqueta <b>status</b>. El dataset tiene 53.043 textos; "
      "45 tienen el texto vacío. La longitud es muy desigual (mediana 52 palabras, percentil 95 de 296, máximo 5.087)."))
vc = pd.concat([trm, vam, tem]).status.value_counts()
add(tabla([["Clase", "Textos (sin los 45 vacíos)", "%"]] + [[c, mil(n), f"{100*n/vc.sum():.1f}".replace(".", ",")] for c, n in vc.items()],
          [5, 4, 1.6]))
add(Spacer(1, 4))
add(tabla([["Tarea", "Filas usadas", "Train", "Validación", "Test", "Partición"],
           ["Binaria", mil(len(trb) + len(vab) + len(teb)), mil(len(trb)), mil(len(vab)), mil(len(teb)),
            "Test 20 % (semilla 42); 10 % del resto para validación"],
           ["Multiclase", mil(len(trm) + len(vam) + len(tem)), mil(len(trm)), mil(len(vam)), mil(len(tem)),
            "Estratificada 70 / 10 / 20 (semilla 42)"]],
          [2.2, 2.2, 1.6, 1.9, 1.5, 7.3]))
add(p("La <b>validación</b> se usa para early stopping y para decidir; el <b>test</b> solo se reporta. El vocabulario se ajusta solo con el train.", S))

# ------------------------------------------------------------------ 2 metodología
add(CondPageBreak(4 * cm)); add(p("2. Modelo y protocolo", H1))
add(p("La arquitectura es la del ejemplo de clase (TensorFlow): <b>TextVectorization → Embedding → Bidirectional(LSTM) → Dense(relu) → Dense(salida)</b>, "
      "con <i>mask_zero</i> para que la RNN ignore el relleno. La estandarización del texto se adaptó al español (se eliminan ¿ ¡ « »). "
      "Configuración base: vocabulario 10.000, embedding 64, BiLSTM de 64 unidades, 1 capa, densa de 64, sin dropout, Adam con lr 1e-3, "
      "batch 64, 128 tokens por texto, early stopping sobre la pérdida de validación con restauración de la mejor época."))
add(p("<b>Métrica de decisión:</b> F1 macro en validación (pesa igual a todas las clases). <b>Ruido:</b> la base se entrena con 3 semillas; "
      "un efecto se considera real si |Δ| supera 2 desviaciones estándar (2σ) entre semillas."))
add(p("<b>Decisiones tomadas durante el experimento</b> (registradas en DISENO_EXPERIMENTAL.md):", N))
for x in ["<b>Diseño reducido.</b> Por costo de cómputo (CPU, 2–8 min por corrida) se variaron 9 factores con un nivel alternativo cada uno, "
          "en lugar de los 44 niveles inicialmente planeados.",
          f"<b>Base multiclase con pesos de clase.</b> Sin pesos, el F1 macro de validación varía {f3(ms(mb0,'val_f1_macro')[0])} ± {f3(ms(mb0,'val_f1_macro')[1])} "
          f"entre semillas y casi nada sería distinguible del ruido; con pesos, {f3(ms(mcw,'val_f1_macro')[0])} ± {f3(ms(mcw,'val_f1_macro')[1])}.",
          "<b>Paciencia 1 en multiclase</b> (3 en binaria), para que cada corrida durara pocos minutos. Esto puede penalizar las configuraciones "
          "que aprenden más despacio (ver limitaciones)."]:
    add(bullet(x))

# ------------------------------------------------------------------ 3 diseño
add(CondPageBreak(4 * cm)); add(p("3. Diseño experimental", H1))
add(p("Se identificaron 19 hiperparámetros en cuatro familias. Los 9 marcados con ✔ se variaron; el resto quedó fijo en su valor base.", N))
fam = [["Familia", "Hiperparámetro", "Base", "Variación probada"],
       ["Arquitectura", "Tipo de celda", "LSTM", "✔ GRU, SimpleRNN"], ["", "Bidireccional", "sí", "✔ no"],
       ["", "Capas RNN", "1", "✔ 2"], ["", "Unidades por capa", "64", "✔ 128"], ["", "Dimensión del embedding", "64", "fijo"],
       ["", "Capa densa (unidades / activación)", "64 / relu", "fijo"], ["", "Combinación bidireccional", "concatenar", "fijo"],
       ["Regularización", "Dropout", "0", "✔ 0,2"], ["", "Dropout recurrente · L2", "0 · 0", "fijo"],
       ["", "Early stopping · paciencia", "sí · 3 (bin) / 1 (multi)", "fijo"], ["", "Pesos de clase (solo multiclase)", "no", "✔ sí (luego adoptado como base)"],
       ["Optimización", "Learning rate", "1e-3", "✔ 3e-4"], ["", "Optimizador · recorte de gradiente · batch", "Adam · no · 64", "fijo"],
       ["Datos", "Tamaño del vocabulario", "10.000", "✔ 1.000"], ["", "Longitud máxima (tokens)", "128", "✔ 256"]]
add(tabla(fam, [2.6, 6.2, 3.6, 4.6]))
add(p("<b>Variables controladas:</b> dataset, particiones, preprocesamiento, función de pérdida, criterio de decisión. "
      "<b>Respuestas:</b> F1 macro (val y test), accuracy, recall por clase, tiempo, parámetros, épocas.", S))

# ------------------------------------------------------------------ 4 resultados
add(CondPageBreak(4 * cm)); add(p("4. Resultados", H1))
add(p("4.1 Ruido entre semillas", H2))
add(tabla([["Configuración (3 semillas)", "F1 macro val", "F1 macro test", "Accuracy test", "Recall T. personalidad"],
           ["Binaria — base", f"{f4(ms(bb,'val_f1_macro')[0])} ± {f4(ms(bb,'val_f1_macro')[1])}",
            f"{f4(ms(bb,'test_f1_macro')[0])} ± {f4(ms(bb,'test_f1_macro')[1])}",
            f"{f4(ms(bb,'test_accuracy')[0])} ± {f4(ms(bb,'test_accuracy')[1])}", "—"],
           ["Multiclase — base sin pesos", f"{f4(ms(mb0,'val_f1_macro')[0])} ± {f4(ms(mb0,'val_f1_macro')[1])}",
            f"{f4(ms(mb0,'test_f1_macro')[0])} ± {f4(ms(mb0,'test_f1_macro')[1])}",
            f"{f4(ms(mb0,'test_accuracy')[0])} ± {f4(ms(mb0,'test_accuracy')[1])}",
            f"{f3(mb0['test_recall_Personality_disorder'].min())} – {f3(mb0['test_recall_Personality_disorder'].max())}"],
           ["Multiclase — base con pesos de clase", f"{f4(ms(mcw,'val_f1_macro')[0])} ± {f4(ms(mcw,'val_f1_macro')[1])}",
            f"{f4(ms(mcw,'test_f1_macro')[0])} ± {f4(ms(mcw,'test_f1_macro')[1])}",
            f"{f4(ms(mcw,'test_accuracy')[0])} ± {f4(ms(mcw,'test_accuracy')[1])}",
            f"{f3(mcw['test_recall_Personality_disorder'].min())} – {f3(mcw['test_recall_Personality_disorder'].max())}"]],
          [5.0, 3.0, 3.0, 3.0, 3.0]))
add(Spacer(1, 4))
add(fig("variabilidad_semillas.png", 15.5, "Figura 1. Cada punto es una semilla. En multiclase sin pesos, el F1 macro salta según si el modelo "
        "\"descubre\" la clase minoritaria (Personality disorder, 2 % de los datos)."))
add(p("Lectura: en multiclase sin pesos, la accuracy es estable pero el F1 macro no, porque la clase minoritaria se detecta o se ignora según "
      "la semilla. Los pesos de clase estabilizan esa clase y reducen el ruido; a cambio, la accuracy baja ~3,5 pp porque se sacrifica "
      "parte del acierto en las clases grandes (sobre todo Depression). La mejora del F1 macro promedio con pesos (+2,7 pp en validación) "
      f"<b>no es estadísticamente significativa</b> con 3 semillas (t de Welch = {fc(stats.ttest_ind(mcw.val_f1_macro, mb0.val_f1_macro, equal_var=False).statistic)}, p = {fc(stats.ttest_ind(mcw.val_f1_macro, mb0.val_f1_macro, equal_var=False).pvalue)})."))

add(p("4.2 Efecto de cada hiperparámetro", H2))
filas = [["Variación (vs base)", "Binaria: Δ F1 val", "Binaria: acc. test", "Multiclase: Δ F1 val", "Multiclase: acc. test"]]
for f in F.FACTORES:
    rb, rm = E.get(("binaria", f)), E.get(("multiclase", f))
    filas.append([F.ETIQ[f], ef("binaria", f) if rb else "—", f4(rb.test_acc) if rb else "—",
                  ef("multiclase", f) if rm else "—", f4(rm.test_acc) if rm else "—"])
filas.append(["Base (media de 3 semillas)", f"F1 val {f4(ms(bb,'val_f1_macro')[0])}", f4(ms(bb, 'test_accuracy')[0]),
              f"F1 val {f4(ms(mcw,'val_f1_macro')[0])}", f4(ms(mcw, 'test_accuracy')[0])])
add(tabla(filas, [4.6, 3.3, 2.4, 3.6, 2.6], resaltar=[len(filas) - 1]))
add(p("Δ en puntos porcentuales de F1 macro de validación; entre paréntesis, en desviaciones estándar del ruido de la base de cada tarea. "
      "Se considera efecto real |Δ| > 2σ.", S))
add(fig("efectos_por_factor.png", 16.5, "Figura 2. Efecto de cada variación sobre el F1 macro de validación. Cada panel tiene su propia escala; "
        "la banda gris es el ruido (±2σ) de esa tarea."))

add(p("4.3 Binaria vs multiclase", H2))
add(fig("binaria_vs_multiclase_sigma.png", 14.5, "Figura 3. Los mismos efectos expresados en unidades de ruido (σ) de cada tarea, para compararlas en una sola escala."))
for x in [f"<b>Bidireccionalidad:</b> irrelevante en binaria ({ef('binaria','bidireccional=False')}) y el factor más importante en multiclase "
          f"({ef('multiclase','bidireccional=False')}). Para distinguir clases parecidas (Depression, Suicidal, Stress) ayuda leer el texto en ambos sentidos.",
          f"<b>Tipo de celda:</b> SimpleRNN empeora en ambas tareas, pero mucho más en multiclase ({ef('multiclase','celda=simplernn')}). "
          f"GRU es la única variación que mejora en ambas (binaria {ef('binaria','celda=gru')}, multiclase {ef('multiclase','celda=gru')}) y además entrena más rápido que LSTM.",
          f"<b>Vocabulario:</b> con 1.000 palabras la binaria no cambia ({ef('binaria','vocab=1000')}), la multiclase cae {ef('multiclase','vocab=1000')}: "
          "distinguir 7 clases requiere términos menos frecuentes.",
          f"<b>Longitud:</b> 256 tokens empeora la binaria ({ef('binaria','max_len=256')}) y no cambia de forma medible la multiclase ({ef('multiclase','max_len=256')}), "
          "con el doble de costo.",
          f"<b>Tamaño (128 unidades) y dropout 0,2:</b> sin efecto medible en ninguna tarea.",
          f"<b>2 capas y lr 3e-4:</b> mejoran levemente la binaria ({ef('binaria','capas=2')}; {ef('binaria','lr=0.0003')}) y empeoran la multiclase "
          f"({ef('multiclase','capas=2')}; {ef('multiclase','lr=0.0003')}). Ambas aprenden más despacio y en multiclase se entrenaron con paciencia 1, "
          "por lo que este resultado puede ser un efecto del protocolo y se toma como débil."]:
    add(bullet(x))

add(p("4.4 Clases y errores en multiclase", H2))
add(fig("recall_por_clase.png", 15.5, "Figura 4. Recall en test por clase (media ± desviación estándar de 3 semillas)."))
add(fig("confusion_multiclase.png", 11, "Figura 5. Matriz de confusión en test de la mejor combinación multiclase (semilla 42). Filas: clase real."))
cmj = __import__("json").load(open(RAIZ / "results" / "histories" / "multiclase__mejor_s42.json"))
import numpy as _np
_cm = _np.array(cmj["cm_test"]); _cn = _cm / _cm.sum(1, keepdims=True); _i = {c: k for k, c in enumerate(cmj["clases"])}
q = lambda a, b_: f"{100*_cn[_i[a], _i[b_]]:.0f} %"
add(p(f"La confusión dominante es entre <b>Depression y Suicidal</b>: {q('Depression','Suicidal')} de los textos de Depression se predicen como Suicidal "
      f"y {q('Suicidal','Depression')} en sentido contrario. <b>Stress</b> se confunde sobre todo con Depression ({q('Stress','Depression')}) y Anxiety "
      f"({q('Stress','Anxiety')}). Normal ({q('Normal','Normal')}) y Anxiety ({q('Anxiety','Anxiety')}) son las clases más fáciles."))

add(p("4.5 Curvas de aprendizaje", H2))
add(fig("curvas_aprendizaje.png", 16.5, "Figura 6. Pérdida de entrenamiento y validación. La línea punteada marca la mejor época (pesos restaurados)."))
add(p("En todas las corridas la pérdida de validación toca su mínimo entre la época 1 y la 3, y luego sube mientras la de entrenamiento "
      "sigue bajando: el modelo sobreajusta muy temprano. Early stopping es imprescindible."))

# ------------------------------------------------------------------ 5 mejor
add(CondPageBreak(4 * cm)); add(p("5. Mejor combinación", H1))
add(p("Se combinaron, en cada tarea, las variaciones que mejoraron el F1 de validación en esa tarea, y se entrenó con 3 semillas: "
      "<b>binaria</b> = GRU + 2 capas + lr 3e-4 (las tres superaban 2σ por separado); <b>multiclase</b> = GRU + 256 tokens + dropout 0,2 + pesos de clase "
      "(todas positivas pero ninguna significativa por separado; combinación exploratoria)."))
filas = [["Tarea", "Modelo", "F1 macro val", "F1 macro test", "Accuracy test", "Tiempo (s)"]]
for r in res.itertuples():
    filas.append([r.tarea.capitalize(), "Base" if r.modelo == "base" else "Mejor combinación",
                  f"{f4(r.val_f1_macro_media)} ± {f4(r.val_f1_macro_de)}", f"{f4(r.test_f1_macro_media)} ± {f4(r.test_f1_macro_de)}",
                  f"{f4(r.test_accuracy_media)} ± {f4(r.test_accuracy_de)}", f"{r.tiempo_s_media:.0f}"])
add(tabla(filas, [2.2, 3.2, 3.2, 3.2, 3.2, 1.8]))
add(Spacer(1, 3))
from scipy import stats
tb = stats.ttest_ind(bm.val_f1_macro, bb.val_f1_macro, equal_var=False)
tm = stats.ttest_ind(mm.val_f1_macro, mcw.val_f1_macro, equal_var=False) if len(mm) > 1 else None
add(p(f"<b>Binaria:</b> la combinación no mejora (t de Welch sobre F1 val: p = {fc(tb.pvalue)}; accuracy de test prácticamente igual) y cuesta "
      f"{fc(bm.tiempo_s.mean()/bb.tiempo_s.mean())[:-1]} veces más tiempo. Los efectos individuales no se suman: la base ya está en el techo de la tarea."))
if tm is not None:
    d_val = (mm.val_f1_macro.mean() - mcw.val_f1_macro.mean()) * 100
    d_acc = (mm.test_accuracy.mean() - mcw.test_accuracy.mean()) * 100
    add(p(f"<b>Multiclase:</b> la combinación sube el F1 macro de validación {pp(d_val)} y la accuracy de test {pp(d_acc)} frente a la base con pesos "
          f"(t de Welch sobre F1 val: p = {fc(tm.pvalue)}). " +
          ("Es una mejora consistente entre semillas." if tm.pvalue < 0.05 else
           "La dirección es favorable pero, con 3 semillas, no alcanza significancia estadística.")))
add(fig("costo_vs_efecto.png", 14, "Figura 7. Costo de entrenamiento vs. efecto sobre el F1 de validación para cada variación."))

# ------------------------------------------------------------------ 6 limitaciones
add(CondPageBreak(4 * cm)); add(p("6. Limitaciones", H1))
for x in ["<b>Un factor a la vez:</b> no mide interacciones; la etapa de mejor combinación muestra que en binaria los efectos no se suman.",
          "<b>Una semilla por variación:</b> los efectos grandes (más de 5σ) son confiables; los cercanos a 2σ deben leerse con cautela.",
          "<b>Paciencia 1 en multiclase:</b> puede cortar antes de tiempo las configuraciones lentas (2 capas, lr bajo). En binaria, con paciencia 3, "
          "lr 3e-4 mejoró, lo que es consistente con esa sospecha.",
          "<b>Factores no variados</b> (embedding, densa, L2, optimizador, batch, etc.) por costo de cómputo; quedan identificados en el diseño.",
          "<b>Duplicados:</b> el CSV tiene textos repetidos; 79 textos del test binario y 156 del multiclase también aparecen en el train.",
          "<b>Cómputo:</b> todo se entrenó en CPU; los tiempos reportados son comparables entre sí pero no absolutos."]:
    add(bullet(x))

# ------------------------------------------------------------------ 7 conclusiones
add(CondPageBreak(4 * cm)); concl = [p("7. Conclusiones", H1)]
for x in [f"Una RNN bidireccional simple resuelve la tarea binaria con ~{f3(ms(bb,'test_accuracy')[0])} de accuracy; ajustar hiperparámetros no aporta mejoras medibles.",
          "En multiclase los hiperparámetros sí importan, y los que más pesan son los que dan capacidad de discriminar matices: "
          "bidireccionalidad, celdas con compuertas (LSTM/GRU) y un vocabulario suficiente.",
          "El desbalance de clases domina la variabilidad del F1 macro; los pesos de clase la reducen y hacen comparable el experimento.",
          "GRU es la recomendación práctica: igual o mejor que LSTM en ambas tareas y más rápida.",
          "Medir el ruido entre semillas fue clave: varias diferencias que parecían mejoras (por ejemplo, GRU +6,5 pp contra una sola semilla de la base multiclase) "
          "resultaron estar dentro del azar."]:
    concl.append(bullet(x))

add(KeepTogether(concl[:3]))
for c in concl[3:]:
    add(c)
add(CondPageBreak(4 * cm)); add(p("8. Reproducibilidad", H1))
add(p(f"Repositorio: <link href='{REPO}' color='#2a78d6'><u>{REPO}</u></link>. Contiene el código (<i>src/</i>), el notebook ejecutado "
      "(<i>notebooks/Taller05_RNN.ipynb</i>), el diseño experimental, los resultados de cada corrida (<i>results/</i>) y las figuras (<i>figs/</i>). "
      "Para repetir una corrida: <i>python src/correr.py multiclase cw+celda=gru</i>."))


def pie(canvas, doc):
    canvas.saveState()
    canvas.setFont("F", 7.5)
    canvas.setFillColor(colors.HexColor("#8a8984"))
    canvas.drawString(2 * cm, 1.1 * cm, "Taller 05 — RNN para clasificación de texto en español")
    canvas.drawRightString(A4[0] - 2 * cm, 1.1 * cm, f"{doc.page}")
    canvas.restoreState()


SimpleDocTemplate(str(SALIDA), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm,
                  bottomMargin=1.8 * cm, title="Taller 05 - RNN clasificación de texto",
                  author="David").build(cuerpo, onFirstPage=pie, onLaterPages=pie)
print("ok", SALIDA)
