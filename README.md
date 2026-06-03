# 🍷 Predicción de Calidad del Vino
### Wine Quality Prediction — Clasificación con Machine Learning

**Bootcamp Data Science — Skillnest / Sonda**  
**Autor:** Marco Brante  
**Dataset:** [Wine Quality Dataset — Kaggle](https://www.kaggle.com/datasets/yasserh/wine-quality-dataset)

---

## 📌 Propósito del Proyecto

Este proyecto aplica técnicas de clasificación supervisada para predecir la calidad de vinos tintos a partir de sus características físico-químicas. El objetivo es identificar qué modelo de Machine Learning ofrece mejor rendimiento para este problema, analizando el impacto del desbalance de clases y justificando cada decisión de preprocesamiento.

---

## 📂 Estructura del Repositorio

```
wine-quality-prediction/
│
├── Core_12_Marco_Brante_Predicción_de_Calidad_del_Vino.ipynb   # Notebook principal
├── WineQT.csv                                                    # Dataset
├── README.md                                                     # Este archivo
└── informe/
    └── Core_12_Marco_Brante_Informe.pdf                         # Informe detallado
```

---

## 🔬 Descripción del Dataset

El dataset contiene **1.143 muestras** de vino tinto con 11 variables físico-químicas y una variable objetivo (`quality`, escala 0–10).

| Columna | Descripción |
|---------|-------------|
| `fixed acidity` | Acidez fija (principalmente ácido tartárico) |
| `volatile acidity` | Acidez volátil — altos niveles producen sabor avinagrado |
| `citric acid` | Ácido cítrico — añade frescura al vino |
| `residual sugar` | Azúcar residual tras la fermentación |
| `chlorides` | Cantidad de sal en el vino |
| `free sulfur dioxide` | SO₂ libre — previene oxidación y crecimiento microbiano |
| `total sulfur dioxide` | SO₂ total (libre + ligado) |
| `density` | Densidad del vino |
| `pH` | Nivel de acidez/basicidad (escala 0–14) |
| `sulphates` | Sulfatos — actúan como antimicrobiano y antioxidante |
| `alcohol` | Porcentaje de alcohol |
| `quality` | **Variable objetivo** — puntuación sensorial (0–10) |

---

## ⚙️ Técnicas Utilizadas

### Preprocesamiento
- **Capping (Winsorizing)** al percentil 1%–99% para tratamiento de outliers sin pérdida de registros
- **Agrupación del target** en 3 clases (Baja/Media/Alta) para manejar el desbalance extremo de clases originales
- **Train/Test Split** 80/20 estratificado (`stratify=y`) para mantener proporciones de clases
- **StandardScaler** para normalización de variables — evitando data leakage al ajustar solo con datos de entrenamiento

### Modelos de Clasificación
| Modelo | Hiperparámetros optimizados |
|--------|----------------------------|
| K-Nearest Neighbors (KNN) | `n_neighbors`: [3, 5, 7, 9, 11, 15] |
| Random Forest | `n_estimators`, `max_depth`, `min_samples_split` |
| Regresión Logística | `C`, `solver` |

- **GridSearchCV** con **StratifiedKFold (k=5)** para selección de hiperparámetros

### Evaluación
- Accuracy, Precisión, Recall, F1-Score (weighted)
- Matrices de confusión comparativas
- Curva ROC con AUC (One-vs-Rest) para el mejor modelo
- Análisis de importancia de características

---

## 📊 Resultados Principales

| Modelo | Accuracy | F1-Score |
|--------|---------|----------|
| **Random Forest** | **0.908** | **0.889** |
| KNN | 0.869 | 0.847 |
| Regresión Logística | 0.847 | 0.819 |

**Random Forest** fue el modelo con mejor rendimiento en todas las métricas. Las variables más determinantes fueron `alcohol`, `volatile acidity` y `sulphates`.

> ⚠️ Ningún modelo logró predecir correctamente la clase **Baja** debido a su escasa representación (3.4% del dataset — solo 39 muestras).

---

## 🚀 Cómo Ejecutar el Código

### Requisitos

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

### En Google Colab (recomendado)

1. Montar Google Drive:
```python
from google.colab import drive
drive.mount('/content/drive')
```

2. Subir `WineQT.csv` a Drive y ajustar la ruta en la celda de carga:
```python
df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/datasets/WineQT.csv')
```

3. Ejecutar todas las celdas en orden (`Runtime > Run all`)

### En entorno local (Jupyter)

```bash
git clone https://github.com/TU_USUARIO/wine-quality-prediction.git
cd wine-quality-prediction
jupyter notebook Core_12_Marco_Brante_Predicción_de_Calidad_del_Vino.ipynb
```

Cambiar la ruta del dataset en la celda de carga:
```python
df = pd.read_csv('WineQT.csv')
```

---

## 📦 Dependencias

```
numpy
pandas
matplotlib
seaborn
scikit-learn
jupyter
```

---

## 👤 Autor

**Marco Brante**  
Bootcamp Data Science — Skillnest / Sonda  
📧 brantemarco777@gmail.com
