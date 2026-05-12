# Pipeline Automatizado de Machine Learning (Data Mining)

Este proyecto es una plataforma modular para la automatización de flujos de Machine Learning, desde la carga de datos crudos hasta la inferencia de modelos optimizados.

## 🚀 Flujo Principal (main.py)
El archivo `main.py` es el punto de entrada recomendado. Actualmente está configurado para analizar el dataset de **Credit Risk** (`credit_risk_dataset.csv`).

1. **Carga Interactiva**: Permite seleccionar archivos dinámicamente.
2. **Limpieza Avanzada**: 
   - Manejo automático de nulos y atípicos.
   - **WOE (Weight of Evidence)**: Optimización de variables numéricas mediante bins y escalas de evidencia (crucial para modelos financieros).
   - **Target Encoding y Dummies**: Transformación inteligente de categorías.
3. **Modelado Automático**:
   - Soporta **Regresión Lineal** (para valores continuos) y **Regresión Logística** (para clasificación binaria).
   - Aplicación de **GridSearchCV** para encontrar los mejores parámetros.
   - Selección de variables por significancia estadística.

## 🛠️ Componentes del Sistema

### 1. CleanData.py (Cerebro del Preprocesamiento)
Contiene la clase `Transformar_Df`. Su mayor ventaja es que guarda el estado de la limpieza (medias, modas, mapeos WOE, bins) para que la inferencia sea idéntica al entrenamiento.
- **WOE & IV**: Solo aplica WOE a columnas con un Information Value (IV) > 0.1.
- **Filtrado de Varianza**: Elimina automáticamente columnas con más del 96% de valores constantes.

### 2. MODELS.py (Motor de ML)
Funciones optimizadas para entrenamiento:
- `Regresion_lineal`: Incluye métricas R2 (Train/Test), MSE, MAE.
- `Regresion_logistica`: Incluye Accuracy, Precision, Recall, F1-Score y ROC-AUC.

### 3. workflow.py (Orquestador)
Contiene las funciones modulares que conectan la carga, la limpieza y el entrenamiento. Es la base que utiliza `main.py` para mantener un código limpio y legible.

## 📊 Ejemplo de Configuración (reglas_dict)
Puedes personalizar el tratamiento de cada columna mediante un diccionario:
```python
reglas_dict = {
    "ingresos": {"WOE": True, "bins_woe": 5},
    "categoria_producto": {"Dummies": True},
    "ciudad": {"TargetEncoding": True},
    "descripcion": {"Lematizar": True}
}
```

## 📦 Requisitos
Instala las dependencias necesarias con:
```bash
pip install -r requirements.txt
```

---
*Nota: Este proyecto está diseñado para ser escalable. Puedes añadir nuevos modelos en `MODELS.py` y registrarlos en el orquestador de `workflow.py`.*
