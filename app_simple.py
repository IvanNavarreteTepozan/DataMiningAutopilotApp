from google.generativeai.types import file_types
from google.ai import generativelanguage
import streamlit as st
import pandas as pd
import json
import os
import asyncio
import google.generativeai as genai
from CleanData import Transformar_Df
from MODELS import Regresion_lineal, Regresion_logistica, Arbol_decision
from CargarDatos import AnalizarDatos
import streamlit.components.v1 as components

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Autopilot", page_icon="⚡", layout="wide")

# --- CSS PREMIUM ---
st.markdown("""
<style>
    /* Fondo General Profundo */
    .stApp {
        background: radial-gradient(circle at top right, #1e293b, #0f172a);
        color: #e2e8f0;
    }
    
    /* Contenedores con Efecto Cristal (Solo con contenido real, no el bloque de CSS) */
    [data-testid="stVerticalBlock"] > div:has(.stMarkdown):not(:has(style)) {
        background: rgba(30, 41, 59, 0.5);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 25px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        margin-bottom: 20px;
    }

    /* Títulos y Texto */
    h1 {
        background: linear-gradient(90deg, #10b981, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
        letter-spacing: -1px;
    }
    
    h3 { color: #10b981 !important; font-weight: 600 !important; }

    /* Botones Pro */
    .stButton>button {
        background: linear-gradient(135deg, #059669 0%, #10b981 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        padding: 0.75rem 1.5rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3) !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(16, 185, 129, 0.5) !important;
        background: linear-gradient(135deg, #10b981 0%, #34d399 100%) !important;
    }

    /* Tablas y JSON */
    .stTable {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    thead tr th {
        background-color: #1e293b !important;
        color: #10b981 !important;
    }

    /* Tabs Estilizados */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: transparent;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: #1e293b;
        border-radius: 10px 10px 0 0;
        padding: 10px 20px;
        color: #94a3b8;
    }

    .stTabs [aria-selected="true"] {
        background-color: #10b981 !important;
        color: white !important;
    }

    /* Fuente global equilibrada */
    .stMarkdown p, .stMarkdown li, .stTable, .stTextArea textarea, .stMarkdown span, .stMarkdown div {
        font-size: 1.25rem !important;
        line-height: 1.6 !important;
    }

    /* Títulos y Encabezados */
    h1 { font-size: 2.8rem !important; }
    h2 { font-size: 2.2rem !important; }
    h3 { font-size: 1.8rem !important; }

    /* Contenedores de altura igualada */
    .equal-height-box {
        height: 600px;
        overflow-y: auto;
        padding: 25px;
        background: rgba(30, 41, 59, 0.3);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        scrollbar-width: thin;
        scrollbar-color: #10b981 #1e293b;
    }

    /* Caja de Ajuste (Fuera de columnas) */
    .bottom-feedback-area {
        margin-top: 30px;
        padding: 30px;
        background: rgba(15, 23, 42, 0.6);
        border-radius: 25px;
        border: 1px solid rgba(16, 185, 129, 0.2);
    }

    /* st.text (Subtítulo) a la mitad (1.5rem) */
    [data-testid="stText"] {
        font-size: 1.5rem !important;
        line-height: 1.4 !important;
        color: #94a3b8;
    }

    /* Ocultar bloques de markdown vacíos */
    div:has(> .stMarkdown:empty) {
        display: none;
    }
</style>
""", unsafe_allow_html=True)

# --- INICIALIZACIÓN DE IA Y GUÍA TÉCNICA ---
api_key = None

# Prioridad 1: Streamlit Secrets (Producción)
try:
    if "GEMINI_API_KEY" in st.secrets:
        api_key = st.secrets["GEMINI_API_KEY"]
except:
    # No hay secretos configurados (común en ejecución local)
    pass

# Prioridad 2: Archivo local (Desarrollo) si no se encontró en secretos
if not api_key and os.path.exists("GEMINI_KEY.txt"):
    try:
        with open("GEMINI_KEY.txt", "r") as f:
            api_key = f.read().strip()
    except:
        pass

if api_key:
    genai.configure(api_key=api_key)
else:
    st.error("🔑 No se encontró la API Key válida. Configura 'GEMINI_API_KEY' en los Secrets de Streamlit.")
    st.stop()

guia_tecnica = ""
if os.path.exists("GUIA_TECNICA_IA.txt"):
    with open("GUIA_TECNICA_IA.txt", "r", encoding="utf-8") as f:
        guia_tecnica = f.read()

model_ia = genai.GenerativeModel("gemini-2.5-flash")

# --- LÓGICA DE NEGOCIO INTEGRADA ---
def aplicar_limpieza_interna(df, col_target, reglas_dict=None):
    transformador = Transformar_Df(df, col_target=col_target)
    reporte = transformador.Clean_All_Rows(reglas_dict=reglas_dict)
    return transformador, transformador.df, transformador.y

def orquestador_modelos_interno(X, y, tipo_modelo):
    m_type = tipo_modelo.lower()
    if 'lineal' in m_type or 'regresion' in m_type:
        json_res, modelo, cols = Regresion_lineal(X, y)
    elif 'arbol' in m_type or 'decision' in m_type:
        json_res, modelo, cols = Arbol_decision(X, y)
    else:
        json_res, modelo, cols = Regresion_logistica(X, y)
    return modelo, json.loads(json_res), cols

# --- ESTADO DE LA SESIÓN ---
if "phase" not in st.session_state: st.session_state.phase = "CARGA"
if "df" not in st.session_state: st.session_state.df = None
if "proposal" not in st.session_state: st.session_state.proposal = None
if "config_pipeline" not in st.session_state: st.session_state.config_pipeline = None
if "results" not in st.session_state: st.session_state.results = None
if "cleaner" not in st.session_state: st.session_state.cleaner = None
if "report_html" not in st.session_state: st.session_state.report_html = None

# --- FUNCIONES DE APOYO ---
def get_ia_proposal(df, feedback=""):
    dtypes = df.dtypes.apply(lambda x: str(x)).to_dict()
    nulls = df.isnull().sum().to_dict()
    
    # Instrucciones dinámicas según si es actualización o inicio
    narrativa_solicitada = ""
    if feedback:
        narrativa_solicitada = "EMPIEZA TU RESPUESTA DICIENDO: 'Entendido, he procesado tus ajustes. Este es el nuevo plan estratégico...'"
    else:
        narrativa_solicitada = "Presenta un plan inicial de ciencia de datos."

    prompt = f"""
    Eres un Consultor de Negocio y Estratega de Datos experto.
    
    CONVOCATORIA TÉCNICA:
    - Guía interna: {guia_tecnica}
    - Metadatos del Dataset: {json.dumps(dtypes)}
    - Valores Nulos: {json.dumps(nulls)}
    
    INSTRUCCIONES DEL USUARIO: {feedback if feedback else 'Análisis inicial sin instrucciones previas.'}
    
    TU OBJETIVO:
    1. {narrativa_solicitada} Explica la estrategia de negocio y por qué elegiste el modelo.
    2. NUNCA menciones nombres de funciones técnicas internas de Python.
    3. Es OBLIGATORIO que al final incluyas un bloque de código JSON con la siguiente estructura exacta:
    
    ```json
    {{
      "col_target": "NOMBRE_DE_LA_COLUMNA_OBJETIVO",
      "tipo_modelo": "regresion_lineal" | "regresion_logistica" | "arbol_decision",
      "metodos_imputacion": {{
        "NOMBRE_COLUMNA": {{
          "metodo": "mean" | "median" | "mode" | "drop-column",
          "Dummies": true | false
        }}
      }}
    }}
    ```

    IMPORTANTE: 
    - `col_target` DEBE ser una columna existente en los metadatos.
    - `tipo_modelo` DEBE ser uno de los tres valores permitidos.
    - El JSON debe ser válido y estar al final de tu respuesta.
    """
    response = model_ia.generate_content(prompt)
    return response.text

# --- RENDERIZADO POR FASES ---
st.title("⚡ Data Mining Autopilot")
st.text("Proyecto para la automatización del preprocesamiento de datos y entrenamiento de modelos de machine learning")

if st.session_state.phase == "CARGA":
    uploaded_file = st.file_uploader("Sube tu archivo", type=["csv", "xlsx"])
    if uploaded_file:
        with st.spinner("🚀 Cargando y procesando datos..."):
            if uploaded_file.name.endswith(".csv"):
                try:
                    st.session_state.df = pd.read_csv(uploaded_file, encoding='utf-8')
                except UnicodeDecodeError:
                    uploaded_file.seek(0)
                    st.session_state.df = pd.read_csv(uploaded_file, encoding='latin1')
            else:
                st.session_state.df = pd.read_excel(uploaded_file)
            st.session_state.phase = "PROPUESTA"
            st.rerun()

elif st.session_state.phase == "PROPUESTA":
    tab1, tab2 = st.tabs(["📝 Estrategia de IA", "📊 Reporte de Datos"])
    
    with tab1:
        if not st.session_state.proposal:
            with st.spinner("🧠 El agente está diseñando la estrategia inicial..."):
                st.session_state.proposal = get_ia_proposal(st.session_state.df)
        
        import re
        # Extraer JSON y limpiar texto
        json_match = re.search(r"```json\s*(\{.*?\})\s*```", st.session_state.proposal, re.DOTALL)
        json_str = json_match.group(1) if json_match else "{}"
        explicacion = re.sub(r"```json.*?```", "", st.session_state.proposal, flags=re.DOTALL).strip()
        
        # Diseño de Columnas Igualadas
        col1, col2 = st.columns([1.6, 1.4], gap="large")
        
        with col1:
            st.markdown("### 📝 Propuesta Estratégica")
            st.markdown(f'<div class="equal-height-box">{explicacion}</div>', unsafe_allow_html=True)
            
        with col2:
            st.markdown("### 🛠️ Configuración Técnica")
            try:
                conf_data = json.loads(json_str)
                st.markdown(f"**🎯 Target:** `{conf_data.get('col_target')}`")
                st.markdown(f"**🧠 Modelo:** `{conf_data.get('tipo_modelo')}`")
                st.markdown("### 📊 Tratamiento de nulos y columnas")
                rules = conf_data.get('metodos_imputacion', conf_data.get('reglas_dict', {}))
                if rules:
                    table_data = []
                    for col, params in rules.items():
                        table_data.append({
                            "Columna": col,
                            "Tratamiento": params.get("metodo"),
                            "Dummies": "✅" if params.get("Dummies") else "❌"
                        })
                    st.table(pd.DataFrame(table_data))
            except: st.error("Error en configuración JSON")
            
            if st.button("🚀 Ejecutar Pipeline", use_container_width=True):
                st.session_state.config_pipeline = json.loads(json_str)
                st.session_state.phase = "EJECUCION"
                st.rerun()
        # SECCIÓN DE AJUSTES (FUERA DE COLUMNAS)
        st.markdown("### 🎯 Refinar Plan y Recomendaciones")
        feedback_val = st.text_area("Añade tus ajustes o contexto adicional:", placeholder="Ej: No utilices la columna X, cambia el modelo a Y, elimina la columna Z, etc...", label_visibility="collapsed")
        
        if st.button("🔄 Actualizar Propuesta Estratégica", use_container_width=True):
            instruction = f"Usa este json como base: {json_str}. Cambia solo lo que pida el feedback: {feedback_val}"
            with st.spinner("🔄 Ajustando estrategia..."):
                st.session_state.proposal = get_ia_proposal(st.session_state.df, instruction)
                st.rerun()
    with tab2:
        st.markdown("### 📊 Reporte Exploratorio Detallado")
        if not st.session_state.report_html:
            with st.spinner("Generando reporte interactivo de calidad de datos..."):
                st.session_state.report_html = AnalizarDatos(st.session_state.df)
        
        components.html(st.session_state.report_html, height=1000, scrolling=True)

elif st.session_state.phase == "EJECUCION":
    conf = st.session_state.config_pipeline
    try:
        # Detección flexible de llaves
        target = conf.get('col_target', conf.get('target'))
        reglas = conf.get('metodos_imputacion', conf.get('reglas_dict', {}))
        modelo_t = conf.get('tipo_modelo', conf.get('modelo', 'regresion_lineal'))

        with st.spinner("🧹 Iniciando Limpieza Automatizada..."):
            cleaner, X, y = aplicar_limpieza_interna(st.session_state.df, col_target=target, reglas_dict=reglas)
            st.session_state.cleaner = cleaner
            
            # --- GUARDADO AUTOMÁTICO INTERNO ---
            try:
                df_export = pd.concat([X, y], axis=1)
                df_export.to_excel("dataset_limpio.xlsx", index=False)
                st.success("✅ Dataset limpio guardado como 'dataset_limpio.xlsx'")
            except Exception as e:
                st.warning(f"Error al guardar excel: {e}")
        
        with st.spinner(f"🧠 Optimizando y Entrenando {modelo_t}..."):
            X_numeric = X.select_dtypes(include=['number'])
            cols_eliminadas = set(X.columns) - set(X_numeric.columns)
            if cols_eliminadas: st.warning(f"⚠️ Columnas eliminadas por seguridad: {list(cols_eliminadas)}")
            
            modelo_obj, metricas, cols = orquestador_modelos_interno(X_numeric, y, tipo_modelo=modelo_t)
            st.session_state.results = {"modelo": modelo_obj, "metricas": metricas, "cols": cols}
        
        st.session_state.phase = "RESULTADOS"
        st.rerun()
    except Exception as e:
        st.error(f"Error en el Pipeline: {e}")
        if st.button("Reintentar Propuesta"): st.session_state.phase = "PROPUESTA"; st.rerun()

elif st.session_state.phase == "RESULTADOS":
    res = st.session_state.results
    st.balloons()
    
    st.markdown("### 🧠 Interpretación Estratégica del Autopilot")
    with st.spinner("Analizando resultados del modelo..."):
        # Prompt robusto para interpretación profunda
        interp_prompt = f"""
        Actúa como un Consultor de Data Science Senior. Analiza estos resultados:
        Métricas: {json.dumps(res['metricas'])}
        Variables Usadas: {res['cols']}
        
        TAREA:
        1. Explica la relevancia del modelo entrenado y su fiabilidad.
        2. Traduce las métricas técnicas a impacto de negocio.
        3. Detalla cómo influyeron las variables en el resultado.
        4. Concluye con una recomendación estratégica.
        """
        explicacion = model_ia.generate_content(interp_prompt).text
        st.markdown(explicacion)
    
    st.write(f"**Variables procesadas:** {', '.join(res['cols'])}")
    entrada = st.text_input("🎯 Realizar Predicción (formato CSV):")
    if st.button("Predecir"):
        try:
            from io import StringIO
            df_n = pd.read_csv(StringIO(entrada))
            df_p = st.session_state.cleaner.transformar_nueva_tupla(df_n)
            p = res['modelo'].predict(df_p[res['cols']])
            st.success(f"Resultado Predicho: {p[0]}")
        except Exception as e: st.error(f"Error: {e}")
    
    if st.button("🔄 Iniciar Nuevo Proyecto"): st.session_state.clear(); st.rerun()
