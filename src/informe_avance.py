"""Genera el documento corto de avance (PDF) desde results/resultados_binaria.csv."""
import io
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

RAIZ = Path(__file__).resolve().parents[1]
SALIDA = Path(sys.argv[1]) if len(sys.argv) > 1 else RAIZ / "Avance_Taller05_RNN.pdf"
D = "/usr/share/fonts/truetype/dejavu/"
pdfmetrics.registerFont(TTFont("F", D + "DejaVuSans.ttf"))
pdfmetrics.registerFont(TTFont("FB", D + "DejaVuSans-Bold.ttf"))

df = pd.read_csv(RAIZ / "results" / "resultados_binaria.csv")
base = df[df.grupo == "base"]
mu, sd = base.val_f1_macro.mean(), base.val_f1_macro.std(ddof=1)
mu_acc, sd_acc = base.test_accuracy.mean(), base.test_accuracy.std(ddof=1)
otros = df[df.grupo != "base"].copy()
otros["delta"] = otros.val_f1_macro - mu
otros["veredicto"] = [("mejora (> 2 d.e.)" if d > 2 * sd else "empeora (> 2 d.e.)" if d < -2 * sd
                       else "dentro del ruido") for d in otros.delta]

# --- figura: F1 val por corrida con banda de ruido
fig, ax = plt.subplots(figsize=(6.6, 0.38 * (len(otros) + 1) + 0.9), dpi=170)
nombres = ["base (3 semillas)"] + otros.nombre.tolist()
vals = [mu] + otros.val_f1_macro.tolist()
y = list(range(len(vals)))[::-1]
ax.axvspan(mu - 2 * sd, mu + 2 * sd, color="#e6e5e0", lw=0)
for yi, v, n in zip(y, vals, nombres):
    ax.plot([v], [yi], "o", color="#2b2b2a" if n.startswith("base") else "#2a78d6", ms=6)
for yi in y[1:]:
    pass
ax.errorbar([mu], [y[0]], xerr=[[mu - base.val_f1_macro.min()], [base.val_f1_macro.max() - mu]],
            color="#2b2b2a", lw=1.2, capsize=3)
ax.set_yticks(y, nombres, fontsize=7.5)
ax.set_xlabel("F1 macro en validación", fontsize=8)
ax.tick_params(axis="x", labelsize=7.5)
ax.grid(axis="x", color="#e6e5e0", lw=0.6); ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.set_title("Banda gris: base ± 2 desviaciones estándar entre semillas", fontsize=8, loc="left", color="#52514e")
buf = io.BytesIO(); fig.savefig(buf, format="png", bbox_inches="tight"); plt.close(fig); buf.seek(0)

# --- documento
st = getSampleStyleSheet()
N = ParagraphStyle("N", parent=st["Normal"], fontName="F", fontSize=9, leading=12.5)
H = ParagraphStyle("H", parent=N, fontName="FB", fontSize=11.5, spaceBefore=10, spaceAfter=4)
T = ParagraphStyle("T", parent=N, fontName="FB", fontSize=15, leading=19, spaceAfter=4)
S = ParagraphStyle("S", parent=N, fontSize=8.5, textColor=colors.HexColor("#52514e"))
C = ParagraphStyle("C", parent=N, fontSize=8, leading=10)
f = lambda x: f"{x:.4f}".replace(".", ",")
p = lambda t, s=N: Paragraph(t, s)

filas = [[p("<b>Corrida</b>", C), p("<b>F1 val</b>", C), p("<b>Δ vs base</b>", C), p("<b>Veredicto</b>", C),
          p("<b>Acc. test</b>", C), p("<b>Tiempo (s)</b>", C), p("<b>Parámetros</b>", C)],
         [p("base (media de 3)", C), p(f"{f(mu)} ± {f(sd)}", C), p("-", C), p("-", C),
          p(f"{f(mu_acc)} ± {f(sd_acc)}", C), p(f"{base.tiempo_s.mean():.0f}", C), p(f"{int(base.parametros.iloc[0]):,}".replace(",", "."), C)]]
for _, r in otros.iterrows():
    filas.append([p(r.nombre, C), p(f(r.val_f1_macro), C), p(f"{r.delta*100:+.2f} pp".replace(".", ","), C), p(r.veredicto, C),
                  p(f(r.test_accuracy), C), p(f"{r.tiempo_s:.0f}", C), p(f"{int(r.parametros):,}".replace(",", "."), C)])
tabla = Table(filas, colWidths=[3.4*cm, 2.8*cm, 1.8*cm, 2.9*cm, 2.8*cm, 1.5*cm, 2.3*cm], repeatRows=1)
tabla.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#ecebe6")),
                           ("LINEBELOW", (0, 0), (-1, -1), 0.3, colors.HexColor("#c9c8c2")),
                           ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 2),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]))

n = len(df)
cuerpo = [
    p("Taller 05 — RNN para clasificación de texto en español", T),
    p("Avance parcial · NLP, Javeriana · código: github.com/davidm094/NLP_JAVE_TALLER05_202601", S),
    p("1. Qué se está haciendo", H),
    p("Se entrena una red recurrente (Embedding → RNN → Densa) sobre la columna <b>statement_es_es</b> para clasificar "
      "<b>Normal vs Depression</b>. El test es exactamente el del Taller 03 (6.351 textos), así que los resultados son comparables con Naive Bayes y LDA. "
      "La validación (2.541 textos) sale solo del train y es lo único que se usa para decidir; el test solo se reporta."),
    p("2. Diseño experimental", H),
    p("19 hiperparámetros en 4 familias (arquitectura, regularización, optimización, datos), variados <b>de a uno</b> alrededor de una configuración base "
      "(BiLSTM de 64 unidades, embedding 64, vocabulario 10.000, 128 tokens, Adam 1e-3, batch 64, early stopping con paciencia 3). "
      "El ruido por semilla se mide repitiendo la base 3 veces: desviación estándar de F1 val = " + f(sd) + ", "
      "por lo que solo se considera real un cambio mayor a ~" + f(2 * sd) + " (2 desviaciones). Etapa 2 (pendiente): factorial 2⁴ para interacciones. "
      "Detalle completo en <i>DISENO_EXPERIMENTAL.md</i>."),
    p(f"3. Resultados hasta ahora ({n} corridas terminadas)", H),
    tabla, Spacer(1, 6), Image(buf, width=15 * cm, height=15 * cm * 0.36 * (len(otros) + 1) / 3.6 if False else None) if False else Image(buf, width=14.5 * cm,
        height=14.5 * cm * (0.38 * (len(otros) + 1) + 0.9) / 6.6),
    p("Δ en puntos porcentuales de F1 macro de validación. d.e. = desviación estándar entre semillas de la base.", S),
    p("4. Lo que se puede afirmar (y lo que no)", H),
]
bul = [
    "<b>Confirmado:</b> la RNN base alcanza " + f(mu_acc) + " de accuracy en test, por encima de LDA+TF-IDF (0,9343) y Naive Bayes (0,8375) del Taller 03. "
    "Matiz: aquí el train es ~10 % más pequeño (22.863 vs 25.404 textos) porque se reserva una validación.",
    "<b>Confirmado:</b> SimpleRNN es peor que LSTM (más de 2 desviaciones de diferencia).",
    "<b>Por confirmar:</b> GRU supera a LSTM por poco más que el ruido y con una sola semilla. Hay que repetirlo con 3 semillas antes de afirmarlo.",
    "<b>Sin efecto detectable:</b> pasar a unidireccional no cambia el desempeño de forma distinguible del ruido, y el entrenamiento cuesta aproximadamente la mitad.",
    "<b>Observación:</b> en todas las corridas el mejor modelo aparece en la época 1–2 y luego la pérdida de validación sube mientras la de entrenamiento baja: "
    "sobreajuste temprano. Dropout, L2 y learning rate menor son los candidatos naturales a mejorarlo y están en el barrido.",
]
for b in bul:
    cuerpo.append(p("• " + b))
cuerpo += [p("5. Lo que sigue", H),
           p("El barrido de las 44 corridas de la etapa 1 está corriendo (≈ 2–3 min por corrida). Después: factorial 2⁴ sobre los factores más influyentes "
             "(incluyendo celda LSTM vs GRU), confirmación con 3 semillas, entrenamiento en las 7 clases y reporte final.")]
SimpleDocTemplate(str(SALIDA), pagesize=A4, leftMargin=2*cm, rightMargin=2*cm, topMargin=1.8*cm,
                  bottomMargin=1.6*cm, title="Avance Taller 05 - RNN").build(cuerpo)
print("ok", SALIDA, n)
