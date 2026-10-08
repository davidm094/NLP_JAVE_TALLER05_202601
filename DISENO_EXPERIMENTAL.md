# Diseño experimental — RNN para clasificación de texto en español

Datos: únicamente la columna `statement_es_es` (texto en español) y `status` (etiqueta).
No se usa IMDB ni el texto en inglés del notebook de ejemplo; de ahí solo se toma la arquitectura.

## 1. Pregunta

¿Cómo afecta cada hiperparámetro de una RNN (y qué interacciones hay entre los más influyentes) al desempeño, al costo de entrenamiento y a la tendencia a sobreajustar?

## 2. Tipos de variables

### Variables respuesta (lo que se mide)
| Variable | Rol |
|---|---|
| **F1 macro en validación** | Respuesta principal; es la única con la que se *decide* |
| Accuracy y F1 macro en test | Se reportan al final; no se usan para elegir |
| Recall por clase | Detectar clases que el promedio esconde (relevante en 7 clases) |
| Brecha train − validación (accuracy) | Medida de sobreajuste |
| Parámetros entrenables, tiempo de entrenamiento, épocas hasta el mejor modelo | Costo |

### Factores (lo que se varía), por familia
| Familia | Factor | Nivel base | Niveles probados |
|---|---|---|---|
| Arquitectura | Tipo de celda | LSTM | **GRU**, SimpleRNN |
| | Bidireccional | sí | no |
| | Capas RNN apiladas | 1 | 2, 3 |
| | Unidades por capa | 64 | 16, 32, 128, 256 |
| | Dimensión del embedding | 64 | 16, 32, 128, 256 |
| | Unidades de la capa densa | 64 | 0 (sin capa), 16, 128 |
| | Activación de la capa densa | relu | tanh |
| | Combinación de las dos direcciones (bidireccional) | concatenar | sumar |
| Regularización | Dropout (sobre la entrada de la RNN y tras la capa densa) | 0 | 0,2 · 0,5 |
| | Dropout recurrente | 0 | 0,2 |
| | Regularización L2 (pesos de la RNN y la densa) | 0 | 1e-4 · 1e-3 |
| | Early stopping | sí | no (20 épocas fijas) |
| | Paciencia del early stopping | 3 | 1, 5 |
| Optimización | Learning rate | 1e-3 | 1e-4, 3e-4, 3e-3, 1e-2 |
| | Optimizador | Adam | RMSprop (1e-3), SGD+momentum (1e-2) |
| | Recorte de gradiente (norma) | ninguno | 1,0 |
| | Batch size | 64 | 32, 128, 256 |
| Datos / representación | Tamaño del vocabulario | 10 000 | 1 000, 5 000, 20 000, 40 000 |
| | Longitud máxima (tokens) | 128 | 32, 64, 256, 512 |

### Variables controladas (fijas en todas las corridas)
Dataset y columna de texto, particiones (el test binario es el mismo del Taller 03; la validación sale solo del train), preprocesamiento (minúsculas, sin puntuación incluyendo ¿ ¡), vocabulario ajustado solo con train, función de pérdida, criterio de elección (F1 macro de validación), máximo de épocas (20).

### Ruido (variable aleatoria)
Semilla de inicialización y de barajado. Se mide repitiendo la configuración base con 3 semillas; su desviación estándar fija el umbral de "diferencia real".

### Hiperparámetros que se dejan fijos (y por qué)
| Hiperparámetro | Valor fijo | Razón |
|---|---|---|
| Inicializadores de pesos | Glorot / ortogonal (por defecto de Keras) | Su efecto se captura con la variación de semilla |
| Máscara de relleno (`mask_zero`) | sí | Sin máscara el padding contamina la RNN; no es una decisión de ajuste |
| Dirección del truncado | se conservan los primeros tokens | `TextVectorization` no lo permite cambiar; sí se varía la longitud |
| Representación de salida de la RNN | último estado | Cambiarla (p. ej. max-pooling) es otra arquitectura, no un hiperparámetro |
| Tokenización | por palabras, minúsculas, sin puntuación | Es preprocesamiento; se varía solo el tamaño del vocabulario |
| Embeddings preentrenados | no | Sin acceso a descargas en el entorno de trabajo |
| Función de pérdida | entropía cruzada | Estándar para clasificación |

## 3. Etapa 1 — tamizaje reducido, un factor a la vez (en curso, binaria y multiclase)

**Decisión (7-oct):** por tiempo de cómputo en CPU se corre una versión reducida: 9 variaciones (un nivel alternativo por factor) + 3 semillas de la base, en ambas tareas; en multiclase además pesos de clase. Factores: celda (GRU, SimpleRNN), bidireccional (no), unidades (128), capas (2), dropout (0,2), learning rate (3e-4), vocabulario (1 000), longitud máxima (256). Quedan en la tabla de factores como *identificados pero no variados*: dimensión del embedding, capa densa (unidades y activación), combinación bidireccional, dropout recurrente, L2, early stopping y su paciencia, optimizador, recorte de gradiente y batch size. Las corridas extra que ya existían en binaria (vocab = 5 000) se conservan.

### Diseño original (44 corridas, no ejecutado completo)

Cada fila cambia **un** factor respecto a la base y deja los demás fijos: 44 corridas + 3 semillas de la base + 1 réplica del notebook original (vocabulario 1 000, lr 1e-4, 10 épocas, 512 tokens). Un efecto se considera real si |Δ F1 val| > 2σ (σ entre semillas de la base).

Limitación: este diseño no detecta interacciones (por ejemplo, que GRU solo mejore si es bidireccional). Para eso es la etapa 2.

## 4. Etapa 2 — factorial completo 2⁴ con réplicas

Se eligen 4 factores: **tipo de celda (LSTM vs GRU)**, que se incluye siempre por ser pregunta del taller, y los 3 factores restantes con mayor efecto en la etapa 1. Cada uno con dos niveles (la base y su mejor alternativa de la etapa 1). Son 16 combinaciones × 2 semillas = 32 corridas. Con ellas se estima, por ANOVA, el efecto principal de cada factor y sus interacciones de 2.º orden.

## 5. Etapa 3 — confirmación

La mejor combinación se entrena con 3 semillas en la tarea binaria y luego en las 7 clases (con y sin pesos de clase). Aquí se mira el test por primera vez para decidir algo, y solo para reportar.

## 6. Qué no se hace (y por qué)

No se corre la grilla completa de todas las combinaciones: con 13 factores y 2–5 niveles cada uno son decenas de miles de corridas, y cada una toma 2–5 minutos en CPU. El diseño por etapas cubre efectos principales de todos los factores e interacciones de los cuatro más relevantes.
