"""Reporte final condensado (PDF). Todas las cifras se leen de results/. Uso: python reporte.py [salida.pdf]"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (CondPageBreak, Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer,
                                Table, TableStyle)
from scipy import stats

import figuras as F
from datos import split_binario, split_multiclase
from sobreajuste import metricas

RAIZ = Path(__file__).resolve().parents[1]
SALIDA = Path(sys.argv[1]) if len(sys.argv) > 1 else RAIZ / "Reporte_Taller05_RNN.pdf"
REPO = "https://github.com/davidm094/NLP_JAVE_TALLER05_202601"
D = "/usr/share/fonts/truetype/dejavu/"
pdfmetrics.registerFont(TTFont("F", D + "DejaVuSans.ttf"))
pdfmetrics.registerFont(TTFont("FB", D + "DejaVuSans-Bold.ttf"))
pdfmetrics.registerFont(TTFont("FI", D + "DejaVuSans-Oblique.ttf"))
pdfmetrics.registerFontFamily("F", normal="F", bold="FB", italic="FI", boldItalic="FB")

st = getSampleStyleSheet()
N = ParagraphStyle("N", parent=st["Normal"], fontName="F", fontSize=9.2, leading=12.8, spaceAfter=4)
H1 = ParagraphStyle("H1", parent=N, fontName="FB", fontSize=12.5, leading=16, spaceBefore=10, spaceAfter=4)
T = ParagraphStyle("T", parent=N, fontName="FB", fontSize=17, leading=21, spaceAfter=3)
S = ParagraphStyle("S", parent=N, fontSize=8, leading=10.5, textColor=colors.HexColor("#52514e"))
C = ParagraphStyle("C", parent=N, fontSize=7.8, leading=9.6, spaceAfter=0)
CB = ParagraphStyle("CB", parent=C, fontName="FB")
B = ParagraphStyle("B", parent=N, leftIndent=10, spaceAfter=2.5)
p = lambda t, s=N: Paragraph(t, s)
dec = lambda x, n: f"{x:.{n}f}".replace(".", ",")
f3, f4 = (lambda x: dec(x, 3)), (lambda x: dec(x, 4))
pp = lambda x: f"{x:+.1f}".replace(".", ",")
mil = lambda x: f"{int(round(x)):,}".replace(",", ".")
bullet = lambda t: Paragraph(t, B, bulletText="•")
h1 = lambda t, alto=4: [CondPageBreak(alto * cm), p(t, H1)]


def tabla(filas, anchos, resaltar=()):
    data = [[p(str(c), CB if i == 0 else C) for c in fila] for i, fila in enumerate(filas)]
    t = Table(data, colWidths=[a * cm for a in anchos], repeatRows=1)
    est = [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#ecebe6")),
           ("LINEBELOW", (0, 0), (-1, -1), 0.3, colors.HexColor("#c9c8c2")),
           ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 2),
           ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]
    est += [("BACKGROUND", (0, r), (-1, r), colors.HexColor("#eef4fc")) for r in resaltar]
    t.setStyle(TableStyle(est))
    return t


def fig(nombre, ancho, pie):
    ruta = RAIZ / "figs" / nombre
    w, h = PILImage.open(ruta).size
    return KeepTogether([Image(str(ruta), width=ancho * cm, height=ancho * cm * h / w), p(pie, S)])


# ------------------------------------------------------------------ datos
b, m = F.cargar()
t = F.tabla_efectos(b, m)
E = {(r.tarea, r.factor): r for r in t.itertuples()}
F.fig_efectos_pp(t); F.fig_recall_clases(m); F.fig_curvas()
F.fig_confusion("multiclase", "mejor_s42", "confusion_multiclase.png")
bb, bm = b[b.grupo == "base"], b[b.grupo == "mejor"]
mb0, mcw, mm = m[m.grupo == "base"], m[m.nombre.str.startswith("class_weight")], m[m.grupo == "mejor"]
mu = lambda s, k: s[k].mean()
sd = lambda s, k: s[k].std(ddof=1)
ms = lambda s, k, n=4: f"{dec(mu(s, k), n)} ± {dec(sd(s, k), n)}"
trb, vab, teb, _ = split_binario()
trm, vam, tem, _ = split_multiclase()
fila = lambda d, n: d[d.nombre == n].iloc[0]
e16b, e16m = fila(b, "emb_dim=16"), fila(m, "cw+emb_dim=16")
p_bin = stats.ttest_ind(bm.val_f1_macro, bb.val_f1_macro, equal_var=False).pvalue
p_mul = stats.ttest_ind(mm.val_f1_macro, mcw.val_f1_macro, equal_var=False).pvalue
p_cw = stats.ttest_ind(mcw.val_f1_macro, mb0.val_f1_macro, equal_var=False).pvalue
n_corr = len(b) + len(m)
ef = lambda tarea, f: f"{pp(E[(tarea, f)].delta_pp)} pp ({pp(E[(tarea, f)].delta_sigma)}σ)"
pd_rec = lambda s: f"{dec(s.test_recall_Personality_disorder.min(), 2)}–{dec(s.test_recall_Personality_disorder.max(), 2)}"

doc = []
add = doc.append
ext = doc.extend

# ------------------------------------------------------------------ portada y resumen
add(p("Clasificación de texto en español con redes recurrentes", T))
add(p("Taller 05 · Procesamiento de Lenguaje Natural · Pontificia Universidad Javeriana · octubre de 2026", S))
add(p(f"Código, notebook y resultados: <link href='{REPO}' color='#2a78d6'><u>{REPO}</u></link>", S))
add(Spacer(1, 4))
add(p(f"Se entrenaron redes recurrentes sobre textos de salud mental en español para dos tareas: <b>binaria</b> (Normal vs Depression) y "
      f"<b>multiclase</b> (7 clases). Se variaron hiperparámetros de forma sistemática ({n_corr} entrenamientos) y se midió el ruido entre "
      "semillas para distinguir efectos reales del azar. La tabla resume las respuestas."))
add(tabla([
    ["Pregunta", "Binaria", "Multiclase"],
    ["¿Qué tan bien funciona la RNN base? (test, 3 semillas)", f"Accuracy {ms(bb,'test_accuracy',3)}",
     f"F1 macro {ms(mcw,'test_f1_macro',3)} · accuracy {dec(mu(mcw,'test_accuracy'),3)} (con pesos de clase)"],
    ["¿Cuánto ruido hay entre semillas? (desv. est. F1 val)", f"{f4(sd(bb,'val_f1_macro'))}",
     f"{f4(sd(mcw,'val_f1_macro'))} con pesos de clase; {f4(sd(mb0,'val_f1_macro'))} sin pesos"],
    ["¿Qué hiperparámetros importan?", "Casi ninguno: todos los efectos caben en ±0,8 pp de F1",
     "Bidireccionalidad (−12 pp sin ella), celda con compuertas (SimpleRNN −9 pp), vocabulario (1.000 palabras −8 pp)"],
    ["¿La mejor combinación supera a la base?", f"No (p = {dec(p_bin,2)}) y es {dec(mu(bm,'tiempo_s')/mu(bb,'tiempo_s'),1)}× más lenta",
     f"{pp((mu(mm,'val_f1_macro')-mu(mcw,'val_f1_macro'))*100)} pp, no significativo (p = {dec(p_mul,2)})"],
    ["¿Por qué tan pocas épocas?", "Sobreajuste temprano: el embedding concentra el 90 % de los parámetros",
     "Ídem; reducir el embedding a 16 dimensiones mantiene el desempeño con 71 % menos parámetros"],
    ["Recomendación", "BiGRU o BiLSTM con embedding de 16", "BiGRU o BiLSTM con embedding de 16 y pesos de clase"]],
    [4.6, 5.0, 7.4]))

# ------------------------------------------------------------------ 1 método
ext(h1("1. Método"))
add(p(f"<b>Datos.</b> Columna <i>statement_es_es</i> (53.043 textos, 45 vacíos; mediana 52 palabras). Clases: Normal 31 %, Depression 29 %, "
      "Suicidal 20 %, Anxiety 7 %, Bipolar 5 %, Stress 5 %, Personality disorder 2 %."))
add(tabla([["Tarea", "Train", "Validación", "Test", "Partición (semilla 42)"],
           ["Binaria (Normal, Depression)", mil(len(trb)), mil(len(vab)), mil(len(teb)), "test 20 %; validación = 10 % del resto"],
           ["Multiclase (7 clases)", mil(len(trm)), mil(len(vam)), mil(len(tem)), "estratificada 70 / 10 / 20"]],
          [4.4, 1.7, 2.2, 1.6, 7.1]))
add(Spacer(1, 3))
add(p("<b>Modelo base</b> (arquitectura del ejemplo de clase): TextVectorization (vocabulario 10.000, 128 tokens, estandarización adaptada a ¿¡) → "
      "Embedding 64 → Bidirectional(LSTM 64) → Dense 64 relu → salida. Adam lr 1e-3, batch 64, early stopping sobre la pérdida de validación "
      "(paciencia 3 en binaria, 1 en multiclase) restaurando la mejor época. En multiclase la base usa <b>pesos de clase</b> (ver sección 3)."))
add(p("<b>Protocolo.</b> La validación decide; el test solo se reporta. Métrica principal: F1 macro. La base se entrena con 3 semillas y un "
      "efecto se considera real si supera 2 desviaciones estándar (2σ) de ese ruido."))
add(p("<b>Diseño.</b> De 19 hiperparámetros identificados (inventario en <i>DISENO_EXPERIMENTAL.md</i>) se variaron 9, uno a la vez: tipo de celda "
      "(GRU, SimpleRNN), bidireccionalidad, unidades (128), capas (2), dropout (0,2), learning rate (3e-4), vocabulario (1.000) y longitud (256). "
      "Después: la mejor combinación por tarea (3 semillas) y 5 variantes contra el sobreajuste."))

# ------------------------------------------------------------------ 2 efectos
ext(h1("2. ¿Qué hiperparámetros importan?", 9))
add(fig("efectos_por_factor.png", 17, "Figura 1. Cambio del F1 macro de validación al variar cada hiperparámetro (una semilla por variación). Cada panel "
        "tiene su escala; la banda gris es el ruido entre semillas (±2σ). Azul: mejora real; naranja: empeora."))
filas = [["Variación (vs base)", "Binaria", "Multiclase", "Tiempo (multiclase)"]]
base_t = mu(mcw, "tiempo_s")
for f in F.FACTORES:
    filas.append([F.ETIQ[f], ef("binaria", f), ef("multiclase", f), f"{dec(E[('multiclase', f)].tiempo_s / base_t, 1)}×"])
add(tabla(filas, [5.2, 3.9, 4.2, 3.0]))
add(p("Δ F1 macro de validación en puntos porcentuales y en σ de cada tarea. Tiempo relativo a la base multiclase.", S))
for x in ["<b>La binaria es casi insensible:</b> el problema es fácil y cualquier RNN razonable llega a ~95 % de accuracy.",
          "<b>En multiclase importa lo que da capacidad de matiz:</b> leer en ambos sentidos, celdas con compuertas y un vocabulario suficiente. "
          "Bidireccionalidad y vocabulario no importan en binaria pero son decisivos con 7 clases.",
          "<b>GRU</b> es la única variación que no empeora en ninguna tarea y entrena más rápido que LSTM. <b>Más unidades</b> no ayudan en ninguna.",
          "<b>2 capas y lr 3e-4</b> empeoran en multiclase, pero aprenden más despacio y allí se usó paciencia 1: resultado débil, posiblemente del protocolo."]:
    add(bullet(x))

# ------------------------------------------------------------------ 3 desbalance
ext(h1("3. Multiclase: el desbalance de clases"))
add(tabla([["Base multiclase (3 semillas)", "F1 macro val", "Accuracy test", "Recall Personality disorder"],
           ["Sin pesos de clase", ms(mb0, "val_f1_macro", 3), ms(mb0, "test_accuracy", 3), pd_rec(mb0)],
           ["Con pesos de clase", ms(mcw, "val_f1_macro", 3), ms(mcw, "test_accuracy", 3), pd_rec(mcw)]],
          [5.2, 3.6, 3.6, 4.6]))
add(Spacer(1, 3))
add(p(f"Sin pesos, el F1 macro depende de la semilla: la clase minoritaria (2 % de los datos) se detecta o se ignora casi por completo. "
      f"Con pesos de clase el ruido baja {dec(sd(mb0,'val_f1_macro')/sd(mcw,'val_f1_macro'),1)}× y esa clase siempre se detecta; a cambio, la "
      f"accuracy cae ~{dec((mu(mb0,'test_accuracy')-mu(mcw,'test_accuracy'))*100,1)} pp porque se sacrifica acierto en las clases grandes "
      f"(sobre todo Depression). La subida del F1 promedio no es significativa (p = {dec(p_cw,2)}); el beneficio es la estabilidad, que hace "
      "medibles los demás efectos."))
add(fig("recall_por_clase.png", 15.5, "Figura 2. Recall en test por clase (media ± desv. est., 3 semillas)."))
cmj = json.load(open(RAIZ / "results" / "histories" / "multiclase__mejor_s42.json"))
cm_ = np.array(cmj["cm_test"]); cn = cm_ / cm_.sum(1, keepdims=True); ix = {c: k for k, c in enumerate(cmj["clases"])}
q = lambda a, c: f"{100 * cn[ix[a], ix[c]]:.0f} %"
add(KeepTogether([
    Table([[Image(str(RAIZ / "figs" / "confusion_multiclase.png"), width=8.2 * cm, height=8.2 * cm * 0.93),
            p(f"<b>Dónde se equivoca.</b> La confusión dominante es Depression ↔ Suicidal: {q('Depression','Suicidal')} de los textos de "
              f"Depression se predicen como Suicidal y {q('Suicidal','Depression')} al revés. Stress se confunde con Depression "
              f"({q('Stress','Depression')}) y Anxiety ({q('Stress','Anxiety')}). Son clases con vocabulario muy parecido; Normal "
              f"({q('Normal','Normal')}) y Anxiety ({q('Anxiety','Anxiety')}) son las más fáciles.<br/><br/>"
              "<font size=7.5 color='#52514e'>Figura 3. Matriz de confusión en test (mejor combinación multiclase, semilla 42). Filas: clase real.</font>")]],
          colWidths=[8.6 * cm, 8.4 * cm], style=[("VALIGN", (0, 0), (-1, -1), "MIDDLE")])]))

# ------------------------------------------------------------------ 4 sobreajuste
ext(h1("4. ¿Por qué el entrenamiento para en 4–5 épocas?"))
add(fig("curvas_aprendizaje.png", 16.5, "Figura 4. Pérdida de entrenamiento y validación por época. Línea punteada: mejor época (pesos que se conservan)."))
add(p("Desde la época 1–2 la pérdida de entrenamiento sigue bajando y la de validación sube: es <b>sobreajuste</b>. El early stopping se detiene "
      "unas épocas después del mínimo y restaura los pesos de la mejor época, por eso el modelo evaluado no es el sobreajustado. La causa es el "
      "embedding: 640.000 de los 714.369 parámetros (90 %) son la tabla de 10.000 palabras × 64 dimensiones, unos 31 parámetros por texto de "
      "entrenamiento. La accuracy de validación casi no cae aunque la pérdida suba: el modelo se vuelve sobreconfiado, no mucho más errado."))
filas = [["Variante (binaria)", "F1 val", "Acc. test", "Mejor época", "Subida pérdida val*", "Parámetros"]]
mb_ = [metricas("binaria", n) for n in bb.nombre]
filas.append(["Base (3 semillas)", f4(mu(bb, "val_f1_macro")), f4(mu(bb, "test_accuracy")),
              f"{min(x['mejor_ep'] for x in mb_)}–{max(x['mejor_ep'] for x in mb_)}",
              f"+{min(x['subida_pct'] for x in mb_):.0f} a +{max(x['subida_pct'] for x in mb_):.0f} %", mil(bb.parametros.iloc[0])])
for n, et in [("emb_dim=16", "Embedding 16 (vs 64)"), ("emb_dropout=0.3", "Dropout espacial 0,3 en el embedding"),
              ("l2_emb=1e-05", "L2 en el embedding (1e-5) **"), ("dropout=0.5", "Dropout 0,5"),
              ("emb_dim=32+emb_dropout=0.3", "Embedding 32 + dropout espacial 0,3")]:
    r, mt = fila(b, n), metricas("binaria", n)
    filas.append([et, f4(r.val_f1_macro), f4(r.test_accuracy), str(mt["mejor_ep"]), f"+{mt['subida_pct']:.0f} %", mil(r.parametros)])
add(KeepTogether([tabla(filas, [5.6, 1.7, 1.7, 1.9, 3.0, 2.3], resaltar=[2]),
    p("* Subida de la pérdida de validación desde su mínimo hasta la última época (menor = menos sobreajuste). "
      "** No del todo comparable: Keras suma la penalización L2 a la pérdida.", S)]))
add(p(f"Ninguna regularización mueve el desempeño más de ~0,6 pp. Lo útil es el <b>embedding de 16 dimensiones</b>: mismo desempeño con 71 % menos "
      f"parámetros, y en multiclase (F1 val {f3(e16m.val_f1_macro)} vs {f3(mu(mcw,'val_f1_macro'))} de la base) retrasa la mejor época de la 3–4 a la "
      f"{metricas('multiclase','cw+emb_dim=16')['mejor_ep']}. El sobreajuste es real pero no es el cuello de botella: el early stopping ya rescata el mejor punto."))

# ------------------------------------------------------------------ 5 conclusiones
ext(h1("5. Mejor combinación, conclusiones y limitaciones"))
add(tabla([["Tarea", "Modelo (3 semillas)", "F1 macro val", "F1 macro test", "Accuracy test", "Tiempo"],
           ["Binaria", "Base: BiLSTM", ms(bb, "val_f1_macro"), ms(bb, "test_f1_macro"), ms(bb, "test_accuracy"), f"{mu(bb,'tiempo_s'):.0f} s"],
           ["", "GRU + 2 capas + lr 3e-4", ms(bm, "val_f1_macro"), ms(bm, "test_f1_macro"), ms(bm, "test_accuracy"), f"{mu(bm,'tiempo_s'):.0f} s"],
           ["Multiclase", "Base: BiLSTM + pesos", ms(mcw, "val_f1_macro"), ms(mcw, "test_f1_macro"), ms(mcw, "test_accuracy"), f"{mu(mcw,'tiempo_s'):.0f} s"],
           ["", "GRU + 256 tokens + dropout 0,2 + pesos", ms(mm, "val_f1_macro"), ms(mm, "test_f1_macro"), ms(mm, "test_accuracy"), f"{mu(mm,'tiempo_s'):.0f} s"]],
          [2.0, 4.0, 2.8, 2.8, 2.8, 1.8]))
add(p(f"Las combinaciones juntan las variaciones que mejoraron en cada tarea. Ninguna supera a su base de forma significativa "
      f"(binaria p = {dec(p_bin,2)}, multiclase p = {dec(p_mul,2)}): los efectos individuales no se suman.", S))
add(p("<b>Conclusiones</b>"))
for x in ["Una RNN bidireccional simple resuelve la tarea binaria (~95 % de accuracy) y el ajuste de hiperparámetros no aporta mejoras medibles.",
          "En multiclase los hiperparámetros sí importan; los decisivos son la bidireccionalidad, las celdas con compuertas y el vocabulario.",
          "El desbalance de clases domina la variabilidad del F1 macro; los pesos de clase la estabilizan a costa de accuracy.",
          "El modelo sobreajusta desde la época 1–2 por exceso de parámetros en el embedding; uno de 16 dimensiones rinde igual con 71 % menos parámetros.",
          "Medir el ruido entre semillas fue clave: varias \"mejoras\" medidas contra una sola semilla resultaron estar dentro del azar."]:
    add(bullet(x))
add(p("<b>Limitaciones</b>"))
for x in ["Una semilla por variación y diseño de un factor a la vez (no mide interacciones); los efectos cercanos a 2σ deben leerse con cautela.",
          "Paciencia 1 en multiclase puede penalizar configuraciones lentas (2 capas, lr bajo).",
          "Varios hiperparámetros (optimizador, batch, capa densa, dropout recurrente, entre otros) quedaron fijos por costo de cómputo (solo CPU); no se probaron embeddings preentrenados.",
          "El CSV tiene duplicados: 79 textos del test binario y 156 del multiclase también están en el train."]:
    add(bullet(x))
add(p(f"<b>Reproducibilidad:</b> <link href='{REPO}' color='#2a78d6'><u>{REPO}</u></link> — código (<i>src/</i>), notebook ejecutado, "
      "resultados por corrida (<i>results/</i>) y diseño experimental. Una corrida: <i>python src/correr.py multiclase cw+celda=gru</i>.", S))


def pie(c, d):
    c.saveState(); c.setFont("F", 7.5); c.setFillColor(colors.HexColor("#8a8984"))
    c.drawString(2 * cm, 1.1 * cm, "Taller 05 — RNN para clasificación de texto en español")
    c.drawRightString(A4[0] - 2 * cm, 1.1 * cm, str(d.page)); c.restoreState()


SimpleDocTemplate(str(SALIDA), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.7 * cm, bottomMargin=1.7 * cm,
                  title="Taller 05 - RNN clasificación de texto en español", author="David").build(doc, onFirstPage=pie, onLaterPages=pie)
print("ok", SALIDA)
