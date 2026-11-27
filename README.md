# Práctica 5.

**Autores:** Alejandro Alejo Santana
**Vídeo/GIF de demostración**

<!-- ![Demo](demo.gif) -->

Se desarrollan **dos prototipos** que reaccionan a la información extraída del rostro con la webcam:

| Prototipo | Notebook | Qué extrae del rostro | Cómo reacciona |
|---|---|---|---|
| **Face Effects** (detector de emociones) | `FaceEffects.ipynb` | Expresión facial con un **modelo entrenado por nosotros** + landmarks | Dibuja un efecto distinto sobre la cara según la emoción |
| **Face Invaders** (temática libre) | `FaceInvaders.ipynb` | Posición de la nariz y apertura de la boca (landmarks y *blendshapes*) | Controla una nave en un videojuego |

---

## 1. Escenarios de aplicación



**Escenario 1: comunicación y entretenimiento (Face Effects).**
Filtros para videollamadas, *streaming* o redes sociales que reaccionan a la emoción del usuario sin que este tenga que elegir nada: estrellas si sonríe, lágrimas si está triste, llamas si está enfadado, etc. Es una aplicación de bajo riesgo, donde un error de clasificación solo provoca un efecto equivocado.

**Escenario 2: accesibilidad y juegos manos libres (Face Invaders).**
Interfaz que se maneja solo con el movimiento de la cabeza y la apertura de la boca, útil para personas con movilidad reducida en las manos o para juegos con control corporal. El juego es la demostración de una interfaz de entrada facial que podría usarse para otras tareas.



## 2. Prototipo 1: Face Effects (detector de emociones)

### Funcionamiento
1. **MediaPipe Face Landmarker** detecta el rostro y sus landmarks en cada fotograma.
2. A partir de los landmarks se calcula el recuadro del rostro (con un 10 % de margen) y se recorta.
3. Cada 0,25 s se clasifica el recorte con el modelo elegido (HOG+SVM, CNN o DeepFace).
4. **Suavizado temporal:** se guardan las últimas 5 predicciones y la expresión solo cambia si una tiene al menos 3 votos. Así el efecto no parpadea.
5. El efecto se dibuja **anclado a landmarks** y escalado con el tamaño de la cara.
6. Si se pierde el rostro, se reinicia el historial.

| Emoción | Efecto | Landmarks usados |
|---|---|---|
| happy | Estrellas que giran alrededor de la cara | Recuadro del rostro |
| sad | Lágrimas desde los párpados inferiores | 145 y 374 |
| angry | Llamas sobre la frente | 10 |
| surprise | Rayos alrededor y signos de exclamación | 10 y recuadro |
| fear | Contorno violeta y gotas de sudor en las sienes | Recuadro |
| disgust | Burbujas verdes ascendentes | Recuadro |
| neutral | Gafas de sol que siguen los ojos | 33, 133, 362 y 263 |

**Controles:** `1` HOG+SVM · `2` CNN · `3` DeepFace · `L` mostrar landmarks · `ESC` salir. La pantalla muestra el modelo activo, el tiempo de inferencia y los FPS.

---

## 3. Modelo biométrico entrenado: reconocimiento de expresiones

### 3.1. Dataset y adecuación
Se usa **FER2013** (7 clases, imágenes de 48×48 en escala de grises):

| Clase | angry | disgust | fear | happy | neutral | sad | surprise | Total |
|---|---|---|---|---|---|---|---|---|
| Train | 3995 | 436 | 4097 | 7215 | 4965 | 4830 | 3171 | 28 709 |
| Test | 958 | 111 | 1024 | 1774 | 1233 | 1247 | 831 | 7 178 |

Se eligió porque es el conjunto estándar de expresiones y porque el mismo detector de DeepFace se evalúa sobre él, lo que permite comparar. Sus problemas conocidos:
- **Desbalanceo fuerte:** *disgust* tiene 436 muestras frente a 7215 de *happy* (16,5 veces menos).
- **Baja resolución** (48×48) y **etiquetas ruidosas**, que limitan la precisión máxima alcanzable.

**Adecuación aplicada:**
- SVM: `class_weight='balanced'`.
- CNN mejorada: pesos por clase `1/√n` normalizados (*disgust* 2,44; *happy* 0,60). Se usa la raíz para no sobreponderar demasiado la clase minoritaria.
- Para las CNN, partición de validación estratificada del 15 % del *train* (24 402 / 4 307), con semilla fija. **El conjunto de test solo se usa al final.**

### 3.2. Modelos comparados

| Modelo | Descripción | Notebook |
|---|---|---|
| **HOG + SVM** | HOG con 4×4 celdas y 9 orientaciones (144 características), normalización MinMax y SVM RBF | `EntrenamientoExpresiones.ipynb` |
| **CNN básica** | 3 bloques conv (32-64-128) + capa densa de 128 con *dropout* (~0,68 M parámetros). Adam, *early stopping* | `EntrenamientoCNN.ipynb` |
| **CNN mejorada** | 3 bloques de 2 conv (32-64-128) con *BatchNorm* y *Dropout2d* creciente (~0,88 M parámetros). *Data augmentation* (volteo, rotación ±10°, desplazamiento 8 %, escala 0,9–1,1), pesos por clase, *weight decay* y `ReduceLROnPlateau` | `EntrenamientoCNN_Mejorada.ipynb.ipynb` |
| **DeepFace (`analyze`)** | Modelo preentrenado de emociones, sin entrenar nosotros | `EvaluacionDeepFace.ipynb` |

### 3.3. Resultados sobre el test (7 178 imágenes)

| Modelo | Accuracy | F1 macro | Entrenamiento | Predicción sobre el test |
|---|---|---|---|---|
| HOG + SVM | 47,3 % | 0,47 | 111 s (CPU) | 29 s (CPU) |
| CNN básica | 55,3 % | 0,50 | 32 s (GPU) | 0,56 s (GPU) |
| **CNN mejorada** | **62,5 %** | **0,58** | 1 193 s (GPU) | 0,39 s (GPU) |
| DeepFace | 56,4 % | 0,55 | — | 88 s |

Matrices de confusión: `matriz_confusion.png` (SVM), `matriz_confusion_cnn.png`, `matriz_confusion_cnn_mejorado.png` y `matriz_confusion_deepface.png`.

> Los tiempos se midieron con hardware distinto (CPU frente a GPU RTX 3070 portátil), por lo que **no son estrictamente comparables**; sirven como orden de magnitud.

---

## 4. Discusión de resultados

**La CNN mejorada es la mejor opción.** Supera en 15 puntos al SVM, en 7 a la CNN básica y en 6 a DeepFace sobre el mismo test. Los HOG son descriptores diseñados a mano que capturan bordes, mientras que la CNN aprende sus propias características. La mejora de la CNN básica a la mejorada se debe a la combinación de *BatchNorm*, *dropout*, *data augmentation* y pesos por clase. No se hizo un estudio de ablación, así que no podemos atribuir la mejora a un solo factor. Aun así, la mejorada también rinde mejor en las clases minoritarias: el recall de *disgust* pasa de 0,15 (CNN básica) a 0,42.

**Compromiso entre modelos.** El SVM es sencillo y no necesita GPU, pero es el más lento prediciendo sobre el test y el menos preciso. La CNN requiere más tiempo de entrenamiento, pero la inferencia es muy rápida con GPU. DeepFace no requiere entrenar, pero tiene un coste de inferencia alto.

**Qué se confunde.**
- *Happy* es la clase más fácil (recall 0,88, F1 0,86): es la mayoritaria y su patrón visual es muy distintivo.
- *Fear* es la más difícil (recall 0,25 en la CNN mejorada). Sus muestras se reparten sobre todo entre *sad* (307), *angry* (170) y *surprise* (133).
- *Neutral* y *sad* se confunden entre sí (259 neutral clasificadas como sad y 218 al revés), porque son expresiones sutiles y parecidas en baja resolución.
- *Disgust* mejora con los pesos, pero con solo 111 muestras de test el resultado es poco estable.

**Limitaciones.**
- La CNN mejorada **no llegó a converger**: completó las 50 épocas, su mejor pérdida de validación fue en la época 49 y la tasa de aprendizaje nunca se redujo. Con más épocas probablemente mejoraría.
- FER2013 tiene etiquetas ruidosas e imágenes de baja resolución. Es esperable que con la webcam el rendimiento difiera del medido en el test; por eso el prototipo usa suavizado por votación.
- No se ajustaron hiperparámetros del SVM (C, número de celdas HOG) ni se probaron otras arquitecturas.
- DeepFace se evalúa sin su detector de caras (`detector_backend="skip"`), sobre las imágenes ya recortadas del dataset.

---

## 5. Prototipo 2: Face Invaders (temática libre)

Videojuego estilo *Space Invaders* controlado con el rostro. **MediaPipe Face Landmarker** con *blendshapes* proporciona:

| Señal del rostro | Fuente | Acción en el juego |
|---|---|---|
| Posición horizontal de la nariz | Landmark 1 | Mueve la nave (rango 0,25–0,75 de la imagen mapeado a toda la pantalla, con suavizado) |
| Apertura de la boca | *Blendshape* `jawOpen` > 0,35 | Dispara (con 300 ms entre disparos) |
| Rostro no detectado | — | **Pausa** el juego |

Incluye 3 vidas, niveles con velocidad creciente, panel lateral con la webcam y los valores en vivo.

**Controles:** mover la cabeza · abrir la boca para disparar · `R` reiniciar al perder · `ESC` salir.

**Robustez:** valores seguros por defecto si no hay rostro, timestamps monótonos para el modo VIDEO, compensación de la velocidad según los FPS y liberación de la cámara en un bloque `finally`.

---

## 6. Ejecución

**Entorno** (Python 3.11 recomendado):

```
pip install opencv-python mediapipe pygame numpy matplotlib
pip install torch torchvision scikit-learn scikit-image joblib
pip install deepface tf-keras
```

DeepFace (TensorFlow) y PyTorch pueden dar conflictos en un mismo entorno. Si ocurre, usar un entorno por notebook. `FaceEffects` fija `TF_USE_LEGACY_KERAS=1` antes de importar DeepFace.

**Ficheros necesarios en la misma carpeta:**

| Archivo | Uso |
|---|---|
| `face_landmarker.task` | Modelo de MediaPipe (ambos prototipos) |
| `modelo_expresiones.pkl` | SVM + normalizador + configuración HOG |
| `modelo_cnn_mejorado.pth` | CNN mejorada |
| `modelo_cnn.pth` | CNN básica (solo comparación) |
| `FER2013/train`, `FER2013/test` | Solo para reentrenar o evaluar |



## Referencias
- MediaPipe Face Landmarker: <https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker>
- DeepFace: <https://github.com/serengil/deepface>
- FER2013: Goodfellow et al., *Challenges in Representation Learning: A Report on Three Machine Learning Contests*, 2013.


## Ayudas
- este trabajo se ha realizado con la ayuda de chatgpt