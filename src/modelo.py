"""Vectorización y construcción de la RNN (misma arquitectura del notebook de
ejemplo, parametrizada)."""
import re
import string
import numpy as np
import tensorflow as tf

# La estandarización por defecto de TextVectorization no quita los signos
# de apertura del español (¿ ¡) ni comillas tipográficas; se añaden aquí.
_PUNT = re.escape(string.punctuation + "¿¡«»“”‘’…")


def estandarizar(x):
    x = tf.strings.lower(x)
    return tf.strings.regex_replace(x, f"[{_PUNT}]", " ")


def vectorizar(textos_train, conjuntos, vocab, max_len):
    """Ajusta el vocabulario SOLO con train y codifica cada conjunto.
    max_len=None -> sin truncar (padding a la secuencia más larga)."""
    enc = tf.keras.layers.TextVectorization(
        max_tokens=vocab, standardize=estandarizar,
        output_sequence_length=max_len)
    enc.adapt(tf.constant(textos_train))
    out = [enc(tf.constant(t)).numpy().astype("int32") for t in conjuntos]
    return enc, out


CELDAS = {"lstm": tf.keras.layers.LSTM, "gru": tf.keras.layers.GRU,
          "simplernn": tf.keras.layers.SimpleRNN}


def construir(cfg, n_vocab, n_clases):
    reg = tf.keras.regularizers.l2(cfg["l2"]) if cfg["l2"] > 0 else None
    reg_emb = tf.keras.regularizers.l2(cfg.get("l2_emb", 0.0)) if cfg.get("l2_emb", 0.0) > 0 else None
    capas = [tf.keras.layers.Embedding(n_vocab, cfg["emb_dim"], mask_zero=True, embeddings_regularizer=reg_emb)]
    if cfg.get("emb_dropout", 0.0) > 0:  # apaga dimensiones completas del embedding (SpatialDropout1D)
        capas.append(tf.keras.layers.SpatialDropout1D(cfg["emb_dropout"]))
    Celda = CELDAS[cfg["celda"]]
    for i in range(cfg["capas"]):
        ultima = i == cfg["capas"] - 1
        rnn = Celda(cfg["unidades"], return_sequences=not ultima, dropout=cfg["dropout"],
                    recurrent_dropout=cfg["rec_dropout"], kernel_regularizer=reg)
        capas.append(tf.keras.layers.Bidirectional(rnn, merge_mode=cfg["merge"])
                     if cfg["bidireccional"] else rnn)
    if cfg["dense"] > 0:
        capas.append(tf.keras.layers.Dense(cfg["dense"], activation=cfg["dense_act"],
                                           kernel_regularizer=reg))
        if cfg["dropout"] > 0:
            capas.append(tf.keras.layers.Dropout(cfg["dropout"]))
    capas.append(tf.keras.layers.Dense(1 if n_clases == 2 else n_clases))
    return tf.keras.Sequential(capas)


def optimizador(nombre, lr, clipnorm=0.0):
    kw = {"clipnorm": clipnorm} if clipnorm > 0 else {}
    if nombre == "sgd":
        return tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9, **kw)
    return {"adam": tf.keras.optimizers.Adam, "rmsprop": tf.keras.optimizers.RMSprop}[nombre](
        learning_rate=lr, **kw)
