from workflow import (
    CargarDatos, 
    obtener_dataframe_reciente, 
    funcion_borrar_columnas, 
    aplicar_limpieza, 
    orquestador_modelos, 
    mostrar_resumen, 
    realizar_predicciones
)

def main():
    """
    Pipeline Automatizado para el dataset de Credit Risk.
    Este script aplica transformaciones WOE y entrena una Regresión Logística.
    """
    
    # 1. CARGA DE DATOS
    # Selecciona 'data/credit_risk_dataset.csv' cuando se abra el explorador
    print("Por favor, selecciona el archivo 'credit_risk_dataset.csv' en la carpeta data.")
    df = obtener_dataframe_reciente() # Intenta cargar el último o pide uno
    if df is None:
        CargarDatos()
        df = obtener_dataframe_reciente()
    
    if df is None:
        print("Error: No se cargaron datos.")
        return

    # 2. LIMPIEZA INICIAL
    # El dataset de riesgo crediticio es estructuralmente limpio, no borramos columnas iniciales.
    df_filtrado = df.copy()

    # 3. CONFIGURACIÓN DEL PIPELINE (WOE + LOGÍSTICA)
    # Definimos reglas específicas para maximizar el poder predictivo en Credit Scoring
    config_pipeline = {
        "col_target": "loan_status",
        "id_column": None, # Puedes añadir un ID si el dataset lo tuviera
        "es_pca": False,   # WOE es preferible para interpretabilidad en finanzas
        "reglas_dict": {
            # Aplicamos WOE a variables numéricas (Optimización solicitada)
            "person_income": {"WOE": True, "bins_woe": 5},
            "loan_amnt": {"WOE": True, "bins_woe": 5},
            "loan_int_rate": {"WOE": True, "bins_woe": 5},
            "loan_percent_income": {"WOE": True, "bins_woe": 5},
            "person_age": {"WOE": True, "bins_woe": 4},
            "person_emp_length": {"WOE": True, "bins_woe": 4},
            
            # Tratamiento de categóricas
            "person_home_ownership": {"Dummies": True},
            "loan_intent": {"Dummies": True},
            "loan_grade": {"TargetEncoding": True},
            "cb_person_default_on_file": {"Dummies": True}
        }
    }

    # 4. EJECUCIÓN DE LA LIMPIEZA
    # El sistema calculará IV, aplicará WOE si IV > 0.1 y preparará los datos.
    cleaner, X, y = aplicar_limpieza(df_filtrado, **config_pipeline)

    # 5. ENTRENAMIENTO (REGRESIÓN LOGÍSTICA)
    # El orquestador aplicará GridSearch y devolverá métricas de clasificación.
    modelo, metricas, columnas_usadas = orquestador_modelos(X, y, tipo_modelo="logistica")

    # 6. RESULTADOS
    mostrar_resumen(modelo, metricas, columnas_usadas)

    # 7. PREDICCIONES
    realizar_predicciones(cleaner, modelo, columnas_usadas)

if __name__ == "__main__":
    main()
