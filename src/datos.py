"""Carga del CSV y particiones train / val / test.

Binaria (Normal vs Depression): se reproduce exactamente el split de test del
Taller 03 (subconjunto reindexado + train_test_split(test_size=0.2,
random_state=42)), para poder comparar la RNN con Naive Bayes y LDA.
Multiclase (7 clases): split estratificado 70/10/20 con semilla 42.
En ambos casos la validación (para early stopping y para elegir
hiperparámetros) sale SOLO del train; el test no se toca para decidir nada.
"""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

RAIZ = Path(__file__).resolve().parents[1]
CSV = RAIZ / "data" / "Combined_Data_es-ES.csv"
TEXTO = "statement_es_es"
SEMILLA = 42


def cargar():
    df = pd.read_csv(CSV)
    return df.drop(columns=[c for c in df.columns if c.startswith("Unnamed") or c == ""], errors="ignore")


def split_binario():
    df = cargar()
    b = df[df.status.isin(["Normal", "Depression"])].reset_index(drop=True)
    b["y"] = (b.status == "Depression").astype(int)   # 1 = Depression
    tr, te = train_test_split(b, test_size=0.2, random_state=SEMILLA, shuffle=True)
    tr, va = train_test_split(tr, test_size=0.1, random_state=SEMILLA, stratify=tr.y)
    clases = ["Normal", "Depression"]
    return tr, va, te, clases


def split_multiclase():
    df = cargar().dropna(subset=[TEXTO]).reset_index(drop=True)
    clases = sorted(df.status.unique())
    df["y"] = df.status.map({c: i for i, c in enumerate(clases)})
    tr, te = train_test_split(df, test_size=0.2, random_state=SEMILLA, stratify=df.y)
    tr, va = train_test_split(tr, test_size=0.125, random_state=SEMILLA, stratify=tr.y)  # 70/10/20
    return tr, va, te, clases


def obtener(tarea):
    return split_binario() if tarea == "binaria" else split_multiclase()
