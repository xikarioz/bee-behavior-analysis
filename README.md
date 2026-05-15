# Bee Antennae Behavioral Analysis 🐝

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![DeepLabCut](https://img.shields.io/badge/DeepLabCut-2.x-orange)](https://github.com/DeepLabCut/DeepLabCut)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Pipeline completo de *machine learning* y análisis de series temporales para el estudio del comportamiento de antenas de abejas (*Apis mellifera*). El proyecto extrae cinemática de alta resolución con **DeepLabCut** y aplica métodos de sistemas dinámicos, análisis espectral y topología algebraica para identificar y caracterizar estados comportamentales.

---

## Tabla de contenidos

1. [Descripción general](#descripción-general)
2. [Estructura del repositorio](#estructura-del-repositorio)
3. [Instalación](#instalación)
4. [Uso rápido](#uso-rápido)
5. [Módulos en detalle](#módulos-en-detalle)
6. [Dependencias](#dependencias)
7. [Cita](#cita)
8. [Licencia](#licencia)

---

## Descripción general

El pipeline está organizado en dos etapas principales:

```
Video crudo → [Extracción DLC] → Keypoints (HDF5/CSV) → [Análisis dinámico] → Figuras / estadísticas
```

**Extracción de posturas** — inferencia de keypoints anatómicos sobre videos de alta velocidad usando modelos ResNet-50 entrenados con DeepLabCut. Incluye modos de procesamiento secuencial, paralelo en batch y tiempo real sobre Raspberry Pi con sincronización de estímulos via Arduino.

**Análisis dinámico** — descomposición modal variacional (VMD), transformada de Hilbert-Huang (HHT), representaciones latentes con CEBRA, homología persistente (TDA), sincronización de fase (modelo de Kuramoto) y detección de eventos ultrarrápidos (*twitches*).

---

## Estructura del repositorio

```
bee-behavior-analysis/
│
├── README.md
├── .gitignore
├── requirements.txt
│
├── extraccion_dlc/                  # Inferencia de posturas con DeepLabCut
│   ├── abejas_linux.py              #   Inferencia secuencial estándar → HDF5/CSV
│   ├── abejas_paralelo.py           #   Procesamiento en batch (GPU)
│   ├── abejas_pi_v2.py              #   Inferencia en tiempo real (Raspberry Pi + Arduino)
│   └── analizar_completo.py         #   Pipeline de extracción completo con exportación
│
└── analisis_y_figuras/              # Análisis cuantitativo y figuras de publicación
    ├── paper_maestro.py             #   Figuras polares, VMD, espectro de Hilbert
    ├── cebra_paper.py               #   Embeddings CEBRA, TDA, Kuramoto, Wasserstein
    ├── twitch_analysis.py           #   Detección estadística de eventos ultrarrápidos
    └── vmd_rapido.py                #   VMD de alta frecuencia (fs = 2 Hz, ventana 0.5 s)
```

---

## Instalación

### Requisitos previos

- Python ≥ 3.9
- CUDA ≥ 11.3 (recomendado para inferencia GPU)
- [DeepLabCut](https://github.com/DeepLabCut/DeepLabCut) instalado en un entorno conda separado (ver su documentación oficial)

### Entorno recomendado

```bash
# Clonar el repositorio
git clone https://github.com/TU_USUARIO/bee-behavior-analysis.git
cd bee-behavior-analysis

# Crear entorno virtual
python -m venv .venv
source .venv/bin/activate          # Linux / macOS
# .venv\Scripts\activate           # Windows

# Instalar dependencias
pip install -r requirements.txt
```

> **Nota:** `dlclive` y `deeplabcut` pueden requerir instalación por separado siguiendo las instrucciones de sus repositorios oficiales según el hardware disponible (GPU/CPU/Raspberry Pi).

---

## Uso rápido

### 1. Extraer keypoints de un video

```bash
# Inferencia estándar (un solo video)
python extraccion_dlc/abejas_linux.py \
    --config /ruta/al/config.yaml \
    --video /ruta/al/video.mp4 \
    --output resultados/

# Procesamiento en batch (múltiples videos en paralelo)
python extraccion_dlc/abejas_paralelo.py \
    --config /ruta/al/config.yaml \
    --input_dir videos/ \
    --output_dir resultados/
```

### 2. Generar figuras del paper

```bash
# Figuras polares + VMD + espectro de Hilbert
python analisis_y_figuras/paper_maestro.py \
    --data resultados/keypoints.h5 \
    --output figuras/

# Embeddings CEBRA y análisis topológico
python analisis_y_figuras/cebra_paper.py \
    --data resultados/keypoints.h5 \
    --output figuras/
```

---

## Módulos en detalle

### `extraccion_dlc/`

| Script | Descripción |
|---|---|
| `abejas_linux.py` | Inferencia secuencial con DeepLabCut. Exporta coordenadas de keypoints en formato HDF5 y CSV con metadatos de likelihood por frame. |
| `abejas_paralelo.py` | Motor de inferencia en batch optimizado para GPU. Permite procesar múltiples videos concurrentemente reduciendo el tiempo total de análisis. |
| `abejas_pi_v2.py` | Inferencia en tiempo real sobre Raspberry Pi usando `dlclive`. Incluye control de hardware vía Arduino para la entrega de estímulos sincronizados con la captura de video. |
| `analizar_completo.py` | Pipeline integrado: preprocesamiento de video → inferencia → filtrado de baja confianza → exportación. |

### `analisis_y_figuras/`

| Script | Descripción |
|---|---|
| `paper_maestro.py` | Generación de gráficas polares de dirección de movimiento, descomposición VMD en modos intrínsecos y espectros de Hilbert-Huang (HHT) para análisis de frecuencia instantánea. |
| `cebra_paper.py` | Entrenamiento y evaluación de modelos CEBRA para representaciones latentes del movimiento antenal. Incluye homología persistente (Ripser), sincronización de fase (modelo de Kuramoto) y distancias de Wasserstein entre distribuciones de estados. |
| `twitch_analysis.py` | Detección automática de eventos ultrarrápidos (*twitches*) mediante umbralización adaptativa. Evalúa la dependencia del estado comportamental en la frecuencia y amplitud de los eventos. |
| `vmd_rapido.py` | VMD sobre ventanas cortas (0.5 s) a alta frecuencia de muestreo para resolver ritmos rápidos en la dinámica antenal. |

---

## Dependencias

Las dependencias principales se listan en `requirements.txt`. A continuación un resumen por categoría:

| Categoría | Paquetes |
|---|---|
| Deep learning / Pose estimation | `torch`, `dlclive`, `deeplabcut` |
| Análisis de señales | `scipy`, `vmdpy`, `astropy` |
| Datos tabulares | `numpy`, `pandas`, `polars`, `h5py` |
| Representación latente | `cebra` |
| Topología algebraica (TDA) | `ripser`, `persim` |
| Visualización | `matplotlib`, `seaborn` |

---

## Cita

Si este código es útil para tu investigación, por favor citarlo como:

```bibtex
@misc{bee-behavior-analysis,
  author       = {Autor, Nombre},
  title        = {Bee Antennae Behavioral Analysis: A DeepLabCut + CEBRA Pipeline},
  year         = {2025},
  publisher    = {GitHub},
  url          = {https://github.com/TU_USUARIO/bee-behavior-analysis}
}
```

---

## Licencia

Este proyecto está bajo la licencia MIT. Ver el archivo [LICENSE](LICENSE) para más detalles.

## Citing this work

If you use this pipeline in your research, please cite it as follows:

```bibtex
@misc{bee-behavior-analysis,
  author       = {Fitte, Franco},
  title        = {Bee Antennae Behavioral Analysis: A DeepLabCut + CEBRA Pipeline},
  year         = {2026},
  publisher    = {GitHub},
  url          = {[https://github.com/xikarioz/bee-behavior-analysis](https://github.com/xikarioz/bee-behavior-analysis)}
}
