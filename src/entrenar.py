"""Entrena una configuración y guarda métricas, historia y matriz de confusión."""
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.utils.class_weight import compute_class_weight

from datos import RAIZ, TEXTO, obtener
from modelo import construir, optimizador, vectorizar

RES = RAIZ / "results"
(RES / "histories").mkdir(parents=True, exist_ok=True)

BASE = dict(celda="lstm", bidireccional=True, vocab=10000, emb_dim=64,
            unidades=64, capas=1, dropout=0.0, dense=64, optimizador="adam",
            lr=1e-3, batch=64, max_len=128, epocas=20, paciencia=3,
            early_stopping=True, class_weight=False,
            rec_dropout=0.0, l2=0.0, merge="concat", dense_act="relu", clipnorm=0.0,
            emb_dropout=0.0, l2_emb=0.0)

_splits, _vec = {}, {}


def _datos(tarea, vocab, max_len):
    if tarea not in _splits:
        _splits[tarea] = obtener(tarea)
    tr, va, te, clases = _splits[tarea]
    k = (tarea, vocab, max_len)
    if k not in _vec:
        txt = [d[TEXTO].fillna("").astype(str).tolist() for d in (tr, va, te)]
        enc, (Xtr, Xva, Xte) = vectorizar(txt[0], txt, vocab, max_len)
        _vec.clear()  # solo un vocabulario en memoria
        _vec[k] = (enc, Xtr, Xva, Xte)
    enc, Xtr, Xva, Xte = _vec[k]
    return enc, Xtr, tr.y.values, Xva, va.y.values, Xte, te.y.values, clases


def _predecir(model, X, binaria):
    z = model.predict(X, batch_size=512, verbose=0)
    return (z[:, 0] >= 0).astype(int) if binaria else z.argmax(1)


def _metricas(y, p, clases, pref):
    m = {f"{pref}_accuracy": accuracy_score(y, p),
         f"{pref}_f1_macro": f1_score(y, p, average="macro"),
         f"{pref}_f1_weighted": f1_score(y, p, average="weighted"),
         f"{pref}_precision_macro": precision_score(y, p, average="macro", zero_division=0),
         f"{pref}_recall_macro": recall_score(y, p, average="macro")}
    rec = recall_score(y, p, average=None, labels=range(len(clases)))
    for c, r in zip(clases, rec):
        m[f"{pref}_recall_{c.replace(' ', '_')}"] = r
    return m


def entrenar(nombre, cambios, tarea="binaria", grupo="", semilla=42, guardar_modelo=False):
    archivo = RES / f"resultados_{tarea}.csv"
    if archivo.exists() and ((pd.read_csv(archivo).nombre == nombre).any()):
        print(f"[skip] {nombre}")
        return
    cfg = {**BASE, **cambios}
    tf.keras.utils.set_random_seed(semilla)
    enc, Xtr, ytr, Xva, yva, Xte, yte, clases = _datos(tarea, cfg["vocab"], cfg["max_len"])
    binaria = len(clases) == 2
    n_vocab = len(enc.get_vocabulary())
    model = construir(cfg, n_vocab, len(clases))
    loss = (tf.keras.losses.BinaryCrossentropy(from_logits=True) if binaria
            else tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    model.compile(loss=loss, optimizer=optimizador(cfg["optimizador"], cfg["lr"], cfg["clipnorm"]),
                  metrics=["accuracy"])
    model.build((None, Xtr.shape[1]))
    cw = None
    if cfg["class_weight"]:
        w = compute_class_weight("balanced", classes=np.arange(len(clases)), y=ytr)
        cw = dict(enumerate(w))
    cb = []
    if cfg["early_stopping"]:
        cb.append(tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=cfg["paciencia"], restore_best_weights=True))
    t0 = time.time()
    h = model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=cfg["epocas"],
                  batch_size=cfg["batch"], class_weight=cw, callbacks=cb, verbose=2)
    t = time.time() - t0
    ep = len(h.history["loss"])
    best = int(np.argmin(h.history["val_loss"])) + 1 if cfg["early_stopping"] else ep
    pva, pte = _predecir(model, Xva, binaria), _predecir(model, Xte, binaria)
    ptr = _predecir(model, Xtr, binaria)
    fila = dict(nombre=nombre, grupo=grupo, tarea=tarea, semilla=semilla,
                **{k: cfg[k] for k in BASE}, parametros=model.count_params(),
                vocab_real=n_vocab, epocas_corridas=ep, mejor_epoca=best,
                tiempo_s=round(t, 1), s_por_epoca=round(t / ep, 1),
                train_accuracy=accuracy_score(ytr, ptr),
                **_metricas(yva, pva, clases, "val"), **_metricas(yte, pte, clases, "test"))
    nueva = pd.DataFrame([fila])
    if archivo.exists():  # compatible con corridas hechas antes de añadir hiperparámetros
        previo = pd.read_csv(archivo)
        for k in BASE:
            if k not in previo.columns:
                previo[k] = BASE[k]
        nueva = pd.concat([previo, nueva], ignore_index=True)
    nueva.to_csv(archivo, index=False)
    json.dump({"history": {k: [float(v) for v in vs] for k, vs in h.history.items()},
               "clases": clases,
               "cm_test": confusion_matrix(yte, pte, labels=range(len(clases))).tolist()},
              open(RES / "histories" / f"{tarea}__{nombre}.json", "w"))
    if guardar_modelo:
        (RAIZ / "modelos").mkdir(exist_ok=True)
        model.save(RAIZ / "modelos" / f"{tarea}__{nombre}.keras")
    print(f"[ok] {nombre}: val_f1={fila['val_f1_macro']:.4f} test_acc={fila['test_accuracy']:.4f} "
          f"test_f1={fila['test_f1_macro']:.4f} ({t:.0f}s, {ep} ep)", flush=True)
    tf.keras.backend.clear_session()
    return fila
