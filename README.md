# Titanic-KNN-Clasificacion
Este proyecto implementa un modelo de **K-Nearest Neighbors (KNN)** para predecir la **supervivencia de los pasajeros del Titanic, utilizando scikit-learn, pandas y numpy.




##  Descripción

El proyecto incluye:
- Limpieza y preprocesamiento de datos del dataset Titanic.
- Conversión de variables categóricas a numéricas.
- Escalamiento de características con `MinMaxScaler`.
- Búsqueda de hiperparámetros con `GridSearchCV`.
- Evaluación del modelo mediante **accuracy** y **matriz de confusión**.

---

##  Modelo

Se utiliza el algoritmo **KNeighborsClassifier** con una búsqueda en rejilla (`GridSearchCV`) para optimizar los parámetros:
- `n_neighbors`
- `metric`
- `weights`

---

##  Requisitos

Instala las dependencias ejecutando:

```bash
pip install -r requirements.txt

```

## Ejecución

Coloca el archivo titanic.csv en la carpeta data/.

Ejecuta el script principal:

python titanic.py

El modelo mostrará:

-El porcentaje de acierto (accuracy).

-La matriz de confusión.

## Resultados esperados

El modelo logra una exactitud entre 75% y 85%, dependiendo de los parámetros y división de datos.

## Estructura 

├── data/
│   └── titanic.csv
├── main.py
├── README.md
├── requirements.txt
├── LICENSE
└── docs/


## Autor

Verónica Balza


