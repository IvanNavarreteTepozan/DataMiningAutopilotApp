import pandas as pd
import numpy as np
import json
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score,
                             accuracy_score, precision_score, recall_score, f1_score, roc_auc_score)
from sklearn.feature_selection import SelectKBest, f_regression, f_classif
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

from sklearn.compose import TransformedTargetRegressor

def Regresion_lineal(X, y, test_size=0.3, cv_folds=5, model_filename='modelo_lineal.pkl', seed=123, p_value_threshold=0.05):
    
 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)


    columnas_validas = X_train.columns[X_train.nunique() > 1]
    
    if len(columnas_validas) < len(X_train.columns):
        columnas_eliminadas = set(X_train.columns) - set(columnas_validas)   
    X_train = X_train[columnas_validas]
    X_test = X_test[columnas_validas]

    # 2. Selección de Variables (Filtrando por significancia estadística)
    kb = SelectKBest(k="all", score_func=f_regression)
    kb.fit(X_train, y_train)
    
    mascara_significativas = kb.pvalues_ < p_value_threshold
    columnas_significativas = X_train.columns[mascara_significativas].tolist()
    
    # Fallback de seguridad
    if len(columnas_significativas) == 0:
        print(f"Advertencia: Ninguna variable cumplió el umbral p<{p_value_threshold}. Seleccionando las 3 mejores.")
        kb_fallback = SelectKBest(k=min(3, X.shape[1]), score_func=f_regression)
        kb_fallback.fit(X_train, y_train)
        columnas_significativas = X_train.columns[kb_fallback.get_support()].tolist()

    X_train_sel = X_train[columnas_significativas]
    X_test_sel = X_test[columnas_significativas]


    modelo_base = TransformedTargetRegressor(
        regressor=LinearRegression(),
        func=np.log1p,
        inverse_func=np.expm1
    )
    parametros = {
        'regressor__fit_intercept': [True, False],
        'regressor__copy_X': [True, False]
    }
    
    grid_search = GridSearchCV(estimator=modelo_base, param_grid=parametros, 
                               scoring='neg_mean_squared_error', cv=cv_folds, n_jobs=-1)

    grid_search.fit(X_train_sel, y_train)
    mejor_modelo = grid_search.best_estimator_


    # 4. Predicciones y Métricas de Precisión
    # Score en Entrenamiento (Train)
    score=mejor_modelo.score(X_train_sel, y_train)
    
    # Score en Prueba (Test)
    y_pred = mejor_modelo.predict(X_test_sel)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    r2_test = r2_score(y_test, y_pred)

    cv_scores = cross_val_score(mejor_modelo, X_train_sel, y_train, cv=cv_folds, scoring='r2', n_jobs=-1)
    cv_r2_mean = cv_scores.mean()
    cv_r2_std = cv_scores.std()

    resultados = {
        "seleccion_variables": {
            "cantidad_original": X.shape[1],
            "cantidad_final": len(columnas_significativas),
            "variables_utilizadas": columnas_significativas
        },
        "metricas_precision": {
            "modelo.score": float(score),
            "R2": float(r2_test),
            "MSE": float(mse),
            "RMSE": float(rmse),
            "MAE": float(mae),
            "MAPE": float(mape)
        },
        "cross_validation_train_R2": {
            "media": float(cv_r2_mean),
            "desviacion_estandar": float(cv_r2_std),
            "folds": cv_folds
        },
        "mejores_hiperparametros": grid_search.best_params_,
        "ruta_modelo_guardado": model_filename
    }
    
    json_resultado = json.dumps(resultados, indent=4, ensure_ascii=False)

    paquete_modelo = {
        "modelo": mejor_modelo,
        "columnas": columnas_significativas
    }
    joblib.dump(paquete_modelo, model_filename)

    return json_resultado, mejor_modelo, columnas_significativas

def Arbol_decision(X, y, test_size=0.3, cv_folds=5, model_filename='modelo_arbol.pkl', seed=123, p_value_threshold=0.05):
    # 1. Validación de variable objetivo (Debe ser binaria para este flujo)
    unique_values = np.sort(np.unique(y))
    if len(unique_values) != 2:
        raise ValueError(f"Este flujo de Árbol de Decisión requiere una variable binaria. Se encontraron {len(unique_values)} valores.")
    
    # 2. Separación de datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)

    # 3. Selección de Variables
    kb = SelectKBest(k="all", score_func=f_classif)
    kb.fit(X_train, y_train)
    
    mascara_significativas = kb.pvalues_ < p_value_threshold
    columnas_significativas = X_train.columns[mascara_significativas].tolist()
    
    # Fallback
    if len(columnas_significativas) == 0:
        kb_fallback = SelectKBest(k=min(3, X.shape[1]), score_func=f_classif)
        kb_fallback.fit(X_train, y_train)
        columnas_significativas = X_train.columns[kb_fallback.get_support()].tolist()

    X_train_sel = X_train[columnas_significativas]
    X_test_sel = X_test[columnas_significativas]

    # 4. Grid Search para Árbol de Decisión
    parametros = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 5, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    modelo_base = DecisionTreeClassifier(random_state=seed)
    grid_search = GridSearchCV(estimator=modelo_base, param_grid=parametros, 
                               scoring='accuracy', cv=cv_folds, n_jobs=-1)

    grid_search.fit(X_train_sel, y_train)
    mejor_modelo = grid_search.best_estimator_

    # 5. Visualización del Árbol
    plt.figure(figsize=(20,10))
    plot_tree(mejor_modelo, feature_names=columnas_significativas, class_names=[str(v) for v in unique_values], filled=True, rounded=True)
    image_path = model_filename.replace('.pkl', '.png')
    plt.savefig(image_path)
    plt.close()
    print(f"Imagen del árbol guardada en: {image_path}")

    # 6. Predicciones y Métricas
    y_pred = mejor_modelo.predict(X_test_sel)
    y_prob = mejor_modelo.predict_proba(X_test_sel)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label=unique_values[1])
    rec = recall_score(y_test, y_pred, pos_label=unique_values[1])
    f1 = f1_score(y_test, y_pred, pos_label=unique_values[1])
    auc = roc_auc_score(y_test, y_prob)

    cv_scores = cross_val_score(mejor_modelo, X_train_sel, y_train, cv=cv_folds, scoring='accuracy', n_jobs=-1)

    resultados = {
        "seleccion_variables": {
            "cantidad_original": X.shape[1],
            "cantidad_final": len(columnas_significativas),
            "variables_utilizadas": columnas_significativas
        },
        "metricas_precision": {
            "Accuracy": float(acc),
            "Precision": float(prec),
            "Recall": float(rec),
            "F1-Score": float(f1),
            "ROC-AUC": float(auc)
        },
        "cross_validation_train_Accuracy": {
            "media": float(cv_scores.mean()),
            "desviacion_estandar": float(cv_scores.std()),
            "folds": cv_folds
        },
        "mejores_hiperparametros": grid_search.best_params_,
        "ruta_modelo_guardado": model_filename,
        "ruta_imagen_arbol": image_path
    }
    
    json_resultado = json.dumps(resultados, indent=4, ensure_ascii=False)

    paquete_modelo = {
        "modelo": mejor_modelo,
        "columnas": columnas_significativas
    }
    joblib.dump(paquete_modelo, model_filename)

    return json_resultado, mejor_modelo, columnas_significativas

def Regresion_logistica(X, y, test_size=0.3, cv_folds=5, model_filename='modelo_logistico.pkl', seed=123, p_value_threshold=0.05):
    # 1. Validación de variable objetivo (Debe ser binaria)
    unique_values = np.sort(np.unique(y))
    if len(unique_values) != 2:
        raise ValueError(f"La Regresión Logística requiere una variable objetivo binaria (2 valores). Se encontraron {len(unique_values)} valores: {unique_values}")
    
    # 2. Separación de datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)

    # Eliminar columnas constantes
    columnas_validas = X_train.columns[X_train.nunique() > 1]
    X_train = X_train[columnas_validas]
    X_test = X_test[columnas_validas]

    # 3. Selección de Variables (f_classif para clasificación)
    kb = SelectKBest(k="all", score_func=f_classif)
    kb.fit(X_train, y_train)
    
    mascara_significativas = kb.pvalues_ < p_value_threshold
    columnas_significativas = X_train.columns[mascara_significativas].tolist()
    
    # Fallback de seguridad
    if len(columnas_significativas) == 0:
        print(f"Advertencia: Ninguna variable cumplió el umbral p<{p_value_threshold}. Seleccionando las 3 mejores.")
        kb_fallback = SelectKBest(k=min(3, X.shape[1]), score_func=f_classif)
        kb_fallback.fit(X_train, y_train)
        columnas_significativas = X_train.columns[kb_fallback.get_support()].tolist()

    X_train_sel = X_train[columnas_significativas]
    X_test_sel = X_test[columnas_significativas]

    # 4. Grid Search para Regresión Logística
    parametros = {
        'C': [0.01, 0.1, 1, 10],
        'solver': ['liblinear', 'lbfgs'],
        'max_iter': [100, 200]
    }
    
    modelo_base = LogisticRegression(random_state=seed)
    grid_search = GridSearchCV(estimator=modelo_base, param_grid=parametros, 
                               scoring='accuracy', cv=cv_folds, n_jobs=-1)

    grid_search.fit(X_train_sel, y_train)
    mejor_modelo = grid_search.best_estimator_

    # 5. Predicciones y Métricas
    y_pred = mejor_modelo.predict(X_test_sel)
    y_prob = mejor_modelo.predict_proba(X_test_sel)[:, 1] if hasattr(mejor_modelo, "predict_proba") else None

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label=unique_values[1])
    rec = recall_score(y_test, y_pred, pos_label=unique_values[1])
    f1 = f1_score(y_test, y_pred, pos_label=unique_values[1])
    auc = roc_auc_score(y_test, y_prob) if y_prob is not None else 0.0

    cv_scores = cross_val_score(mejor_modelo, X_train_sel, y_train, cv=cv_folds, scoring='accuracy', n_jobs=-1)

    resultados = {
        "seleccion_variables": {
            "cantidad_original": X.shape[1],
            "cantidad_final": len(columnas_significativas),
            "variables_utilizadas": columnas_significativas
        },
        "metricas_precision": {
            "Accuracy": float(acc),
            "Precision": float(prec),
            "Recall": float(rec),
            "F1-Score": float(f1),
            "ROC-AUC": float(auc)
        },
        "cross_validation_train_Accuracy": {
            "media": float(cv_scores.mean()),
            "desviacion_estandar": float(cv_scores.std()),
            "folds": cv_folds
        },
        "mejores_hiperparametros": grid_search.best_params_,
        "ruta_modelo_guardado": model_filename
    }
    
    json_resultado = json.dumps(resultados, indent=4, ensure_ascii=False)

    paquete_modelo = {
        "modelo": mejor_modelo,
        "columnas": columnas_significativas
    }
    joblib.dump(paquete_modelo, model_filename)

    return json_resultado, mejor_modelo, columnas_significativas