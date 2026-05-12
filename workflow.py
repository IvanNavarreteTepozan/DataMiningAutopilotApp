import pandas as pd
import numpy as np
import json
import os
from CargarDatos import CargarDatos, obtener_dataframe_reciente
from CleanData import Transformar_Df
from MODELS import Regresion_lineal, Regresion_logistica, Arbol_decision

def funcion_borrar_columnas(df, columnas):
    """Elimina una lista de columnas del DataFrame."""
    columnas_a_borrar = [c for c in columnas if c in df.columns]
    if columnas_a_borrar:
        print(f"Borrando columnas: {columnas_a_borrar}")
        return df.drop(columns=columnas_a_borrar)
    return df

def aplicar_limpieza(df, col_target, id_column=None, es_pca=False, reglas_dict=None):
    """Aplica la limpieza de datos usando la clase Transformar_Df."""
    print("Iniciando preprocesamiento de datos...")
    transformador = Transformar_Df(df, col_target=col_target, id_column=id_column)
    reporte = transformador.Clean_All_Rows(reglas_dict=reglas_dict, EsPCA=es_pca)
    
    # El reporte es una lista de diccionarios, lo mostramos resumido
    print("### Reporte de Limpieza ###")
    for r in reporte:
        print(f"- {r['columna']}: {r['metodo']} (Relleno: {r['Valor_de_relleno']})")
        
    return transformador, transformador.df, transformador.y

def orquestador_modelos(X, y, tipo_modelo, **kwargs):
    """Aplica el modelo de ML seleccionado y devuelve métricas."""
    print(f"Ejecutando modelo: {tipo_modelo}")
    
    if tipo_modelo.lower() in ['regresion', 'regresion_lineal']:
        json_res, modelo, columnas_usadas = Regresion_lineal(X, y, **kwargs)
        metricas = json.loads(json_res)
        return modelo, metricas, columnas_usadas
    elif tipo_modelo.lower() in ['logistica', 'regresion_logistica', 'clasificacion']:
        json_res, modelo, columnas_usadas = Regresion_logistica(X, y, **kwargs)
        metricas = json.loads(json_res)
        return modelo, metricas, columnas_usadas
    elif tipo_modelo.lower() in ['arbol', 'decision_tree', 'arbol_decision']:
        json_res, modelo, columnas_usadas = Arbol_decision(X, y, **kwargs)
        metricas = json.loads(json_res)
        return modelo, metricas, columnas_usadas
    else:
        # En el futuro se pueden añadir: 'clasificacion', 'clustering', etc.
        raise ValueError(f"Modelo '{tipo_modelo}' no implementado todavía.")

def mostrar_resumen(modelo, metricas, columnas_usadas):
    """Muestra un resumen de las métricas y detalles del modelo."""
    print("\n" + "="*30)
    print("RESUMEN DEL MODELO")
    print("="*30)
    print(f"Variables utilizadas ({len(columnas_usadas)}): {columnas_usadas}")
    print("\nMétricas de precisión:")
    for k, v in metricas['metricas_precision'].items():
        print(f"- {k}: {v:.4f}")
    
    print("\nMejores hiperparámetros:")
    print(json.dumps(metricas['mejores_hiperparametros'], indent=2))
    print("="*30 + "\n")

def realizar_predicciones(cleaner, modelo, columnas_usadas):
    """Pide nuevos datos al usuario y realiza una predicción."""
    print("¿Deseas realizar una predicción? (s/n)")
    if input().lower() != 's':
        return

    print("Introduce los valores para una nueva tupla o el nombre de un archivo para predecir múltiples:")
    entrada = input("Entrada: ")

    if entrada.endswith('.csv'):
        df_nuevo = pd.read_csv(entrada)
    elif entrada.endswith('.xlsx'):
        df_nuevo = pd.read_excel(entrada)
    elif entrada.endswith('.json'):
        df_nuevo = pd.read_json(entrada)
    else:
        # Intento de parsear entrada manual (ej: valor1, valor2...)
        # Nota: En un script real esto sería más robusto, aquí es un ejemplo simple
        print("Para entrada manual, se asume que el usuario conoce el orden o se usará IA (en el script de app.py).")
        print("De momento, por favor usa un archivo CSV para predicciones en este script.")
        return

    df_procesado = cleaner.transformar_nueva_tupla(df_nuevo)
    
    # Alinear con las columnas que el modelo realmente usa
    X_pred = df_procesado.copy()
    if cleaner.id_column and cleaner.id_column in X_pred.columns:
        ids = X_pred[cleaner.id_column]
        X_pred = X_pred.drop(columns=[cleaner.id_column])
    
    X_pred = X_pred[columnas_usadas]
    predicciones = modelo.predict(X_pred)
    
    # Crear un DataFrame de resultados con ID y Predicción
    if cleaner.id_column and cleaner.id_column in df_procesado.columns:
        resultados_finales = pd.DataFrame({
            cleaner.id_column: df_procesado[cleaner.id_column],
            'Prediccion': predicciones
        })
    else:
        resultados_finales = pd.DataFrame({
            'Indice': range(len(predicciones)),
            'Prediccion': predicciones
        })

    print("\nResultados de predicción (ID + Valor):")
    print(resultados_finales.to_string(index=False))
    
    # Guardar en excel el original con la predicción añadida para contexto completo
    df_nuevo['Prediccion'] = predicciones
    df_nuevo.to_excel("resultados_workflow.xlsx", index=False)
    print("\nResultados completos guardados en 'resultados_workflow.xlsx'")

def run_workflow():
    # 1. Cargar Datos
    print("¿Deseas cargar predicciones? (s/n)")
    if input().lower() == 's':
        from CargarDatos import seleccionar_y_cargar_df
        df = seleccionar_y_cargar_df()
    else:
        df = obtener_dataframe_reciente()
    
    if df is None:
        print("No se pudo cargar el DataFrame. Asegúrate de que existan datos en la carpeta 'data' o selecciona uno.")
        return

    # 2. Borrar columnas (ejemplo de columnas a borrar)
    # En un caso real, esto podría venir de un input del usuario
    print(f"Columnas disponibles: {list(df.columns)}")
    cols_a_borrar = ["movie_imdb_link", "movie_title"] # Ejemplo
    df = funcion_borrar_columnas(df, cols_a_borrar)

    # 3. Limpieza de datos
    # Estos parámetros podrían ser dinámicos
    target = 'gross' # Ejemplo para movies.csv
    id_col = None
    pca = True
    
    cleaner, X, y = aplicar_limpieza(df, col_target=target, id_column=id_col, es_pca=pca)

    # 4. Orquestador de Modelos
    modelo, metricas, cols_usadas = orquestador_modelos(X, y, tipo_modelo='regresion')

    # 5. Mostrar Resumen
    mostrar_resumen(modelo, metricas, cols_usadas)

    # 6. Predicción
    realizar_predicciones(cleaner, modelo, cols_usadas)

