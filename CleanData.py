import pandas as pd
import numpy as np
import spacy
import unicodedata
import nltk
import re
from nltk.corpus import stopwords
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas.api.types as ptypes

nltk.download('stopwords', quiet=True)

class Transformar_Df:
    def __init__(self, dataFrame, col_target, id_column=None, idioma='spanish', modelo_nlp='es_core_news_sm'):
        self.df = dataFrame.copy()
        
        # Validar existencia de la columna objetivo
        if col_target not in self.df.columns:
            columnas_disponibles = list(self.df.columns)
            raise ValueError(f"La columna objetivo '{col_target}' no se encuentra en el DataFrame. "
                             f"Columnas disponibles: {columnas_disponibles}")
        
        # Validar existencia de la columna ID (si se proporciona)
        if id_column and id_column not in self.df.columns:
            columnas_disponibles = list(self.df.columns)
            raise ValueError(f"La columna ID '{id_column}' no se encuentra en el DataFrame. "
                             f"Columnas disponibles: {columnas_disponibles}")

        self.col_target_name = col_target 
        self.id_column = id_column
        self.y = None
        
        self.stop_words = set(stopwords.words(idioma))
        self.nlp = spacy.load(modelo_nlp)
        self.scaler = None
        self.pca = None
        self.target_encodings = {}
        self.columnas_entrenamiento = None
        
        # --- NUEVAS VARIABLES DE "MEMORIA" PARA INFERENCIA ---
        self.reglas_entrenamiento = {}
        self.imputaciones_nulos = {}
        self.backup_imputaciones = {} # Para cuando el método fue 'drop-values'
        self.categorias_validas = {}
        self.columnas_texto_separadas = {}
        self.woe_mappings = {} # Almacena bins y valores WOE
        self.ordinal_mappings = {} # Almacena el orden de las categorías ordinales

    def separar_columna(self, columna_nombre, umbral=0.05):
        df_dummies = self.df[columna_nombre].str.get_dummies(sep=' ')
        frecuencias = df_dummies.mean()
        columnas_a_mantener = frecuencias[frecuencias >= umbral].index.tolist()
        
        # GUARDAR ESTADO: Recordar qué palabras sobrevivieron para aplicarlo a datos nuevos
        self.columnas_texto_separadas[columna_nombre] = columnas_a_mantener
        
        df_filtrado = df_dummies[columnas_a_mantener]
        self.df = pd.concat([self.df.drop(columns=[columna_nombre]), df_filtrado], axis=1)

    def lematizar(self, Registro):
        doc = self.nlp(str(Registro))
        lemas = [token.lemma_ for token in doc if token.is_alpha]
        return " ".join(lemas) 

    def ExtraerTexto(self, Registro):
        palabras = str(Registro).split()
        palabras = [p for p in palabras if p not in self.stop_words]
        return " ".join(palabras)

    def Limpiar_solo_cadenas(self, columna, lematizar=False):
        if columna not in self.df.columns: return
        self.df[columna] = self.df[columna].apply(lambda x: unicodedata.normalize('NFKD', str(x)).encode('ASCII', 'ignore').decode('utf-8') if pd.notnull(x) else x)
        self.df[columna] = self.df[columna].str.lower()
        self.df[columna] = self.df[columna].apply(lambda x: re.sub(r'[^a-zA-Z0-9áéíóúÁÉÍÓÚñÑ\s]', ' ', str(x)) if pd.notnull(x) else x)
        
        if lematizar:
            self.df[columna] = self.df[columna].apply(self.ExtraerTexto)
            self.df[columna] = self.df[columna].apply(self.lematizar)
            self.separar_columna(columna)

    # --- 2. OUTLIERS Y NULOS ---
    def Manejo_Atipicos(self, columna):
        Q1 = self.df[columna].quantile(0.25)
        Q3 = self.df[columna].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        self.df.loc[(self.df[columna] < lower) | (self.df[columna] > upper), columna] = np.nan

    def No_nulos(self, columna, metodo=None, toleranciaSustitucion=0.05, toleranciaMantenerColumna=0.1, valorPorDefecto=None):
        if not self.df[columna].isnull().any():
            self.imputaciones_nulos[columna] = None
            return {'columna': columna, 'metodo': 'Ninguno', 'Valor_de_relleno': None}
            
        # Asegurar cálculo de proporción numérico
        total_rows = len(self.df)
        null_count = int(self.df[columna].isna().sum())
        proporcion_nulls = null_count / total_rows if total_rows > 0 else 0
        
        if proporcion_nulls >= toleranciaMantenerColumna or metodo == 'drop-column':
            self.df.drop(columns=[columna], inplace=True)
            return {'columna': columna, 'metodo': 'drop-column', 'Valor_de_relleno': None}

        if valorPorDefecto is not None:
            self.df[columna] = self.df[columna].fillna(valorPorDefecto)
            self.imputaciones_nulos[columna] = valorPorDefecto
            return {'columna': columna, 'metodo': 'valor_preestablecido', 'Valor_de_relleno': valorPorDefecto}

        chosen = metodo
        if metodo is None:
            if proporcion_nulls >= toleranciaSustitucion:
                chosen = 'drop-values'
            elif ptypes.is_numeric_dtype(self.df[columna]):
                skew = self.df[columna].dropna().skew()
                chosen = 'mean' if abs(skew) < 1 else 'median'
            else:
                chosen = 'mode'

        # CALCULAR VALOR DE RESPALDO (Siempre se calcula por si en inferencia faltan datos)
        backup_value = None
        if ptypes.is_numeric_dtype(self.df[columna]):
            skew = self.df[columna].dropna().skew()
            backup_value = self.df[columna].mean() if abs(skew) < 1 else self.df[columna].median()
        else:
            modas = self.df[columna].mode()
            backup_value = modas[0] if not modas.empty else None
        self.backup_imputaciones[columna] = backup_value

        fill_value = None
        if chosen == 'mean':
            fill_value = self.df[columna].mean()
            self.df[columna] = self.df[columna].fillna(fill_value)
        elif chosen == 'median':
            fill_value = self.df[columna].median()
            self.df[columna] = self.df[columna].fillna(fill_value)
        elif chosen == 'mode':
            modas = self.df[columna].mode()
            fill_value = modas[0] if not modas.empty else None
            self.df[columna] = self.df[columna].fillna(fill_value)
        elif chosen == 'drop-values':
            self.df.dropna(subset=[columna], inplace=True)
            
        # GUARDAR ESTADO: Guardar el valor calculado para usarlo en inferencia
        self.imputaciones_nulos[columna] = fill_value
        return {'columna': columna, 'metodo': chosen, 'Valor_de_relleno': fill_value}

    # --- MÉTODOS PARA WOE (Weight of Evidence)---
    def calcular_woe_iv(self, df_temp, col, target_name):
        """Calcula el diccionario WOE y el valor IV total para una columna."""
        # Asegurar que el target sea numérico (0/1) para el cálculo matemático
        # Forzamos conversión a float para evitar errores de tipo en divisiones posteriores
        target_series = pd.to_numeric(df_temp[target_name], errors='coerce').fillna(0).astype(float)
        df_temp[target_name] = target_series
        
        # Agregación por grupos
        stats = df_temp.groupby(col)[target_name].agg(['count', 'sum'])
        stats.columns = ['Total', 'Events']
        
        # Asegurar que las columnas de stats sean float para evitar ufunc errors
        stats = stats.astype(float)
        stats['NonEvents'] = stats['Total'] - stats['Events']
        
        total_events = stats['Events'].sum()
        total_non_events = stats['NonEvents'].sum()
        
        # Evitar división por cero con un pequeño epsilon y asegurar casting float
        dist_events = stats['Events'] / (float(total_events) if total_events > 0 else 1.0)
        dist_non_events = stats['NonEvents'] / (float(total_non_events) if total_non_events > 0 else 1.0)
        
        # Pequeño ajuste para evitar log(0) e inf (reemplazo seguro de ceros)
        dist_events = dist_events.replace(0, 0.0001)
        dist_non_events = dist_non_events.replace(0, 0.0001)
        
        # El cociente también debe ser numérico puro
        woe_series = np.log(dist_non_events / dist_events)
        iv_series = (dist_non_events - dist_events) * woe_series
        
        iv_total = float(iv_series.sum())
        woe_dict = woe_series.to_dict()
        
        return woe_dict, iv_total

    def aplicar_woe_columna(self, col, bins_count=5):
        """Aplica la transformación WOE si el IV es > 0.1."""
        if self.df[col].nunique() < 2:
            return None

        try:
            # Intentar qcut para discretizar
            binned_col, bins_edges = pd.qcut(self.df[col], q=bins_count, retbins=True, duplicates='drop')
        except Exception:
            # Fallback a cut uniforme si hay muchos duplicados
            binned_col, bins_edges = pd.cut(self.df[col], bins=bins_count, retbins=True)

        df_temp = pd.DataFrame({col: binned_col, 'target': self.df[self.col_target_name]})
        woe_dict, iv_total = self.calcular_woe_iv(df_temp, col, 'target')

        # Umbral solicitado: IV > 0.1
        if iv_total > 0.1:
            self.woe_mappings[col] = {
                'bins_edges': bins_edges,
                'woe_dict': woe_dict,
                'iv': iv_total
            }
            # Reemplazar valores originales por el WOE correspondiente
            self.df[col] = binned_col.map(woe_dict).astype(float)
            return f"WOE (IV: {iv_total:.3f})"
        else:
            return f"WOE-Omitido (IV: {iv_total:.3f} <= 0.1)"

    # --- 3. CATEGÓRICAS Y ENCODING ---
    def SePuedeCategorizar(self, columna, max_categorias=20, min_prop=0.05):
        if columna not in self.df.columns: return False
        
        freqs_norm = self.df[columna].value_counts(normalize=True)
        if freqs_norm.empty or freqs_norm.count() > max_categorias or freqs_norm.count() <= 1:
            return False
        
        categorias_invalidas = freqs_norm[freqs_norm <= min_prop].index.tolist()
        suma_otros = freqs_norm[freqs_norm <= min_prop].sum()
        if suma_otros > 0.10:
            return False
        
        num_categorias_finales = freqs_norm.count() - len(categorias_invalidas)
        if len(categorias_invalidas) > 0:
            num_categorias_finales += 1 
            
        columnas_extra = num_categorias_finales - 1 
        
        if (self.df.shape[1] + columnas_extra) > self.df.shape[0]:
            return False

        # GUARDAR ESTADO: Guardar qué categorías son válidas
        categorias_validas = freqs_norm[freqs_norm > min_prop].index.tolist()
        self.categorias_validas[columna] = categorias_validas

        self.df[columna] = self.df[columna].where(self.df[columna].isin(categorias_validas), "otros")
        return True
    def aplicar_target_encoding(self, columna):
        """Aplica Target Encoding (reemplaza categorías por la media del target)."""
        medias = self.df.groupby(columna)[self.col_target_name].mean()
        media_global = self.df[self.col_target_name].mean()
        self.target_encodings[columna] = {'medias': medias.to_dict(), 'global': media_global}
        self.df[columna] = self.df[columna].map(medias).fillna(media_global)

    def aplicar_ordinal_encoding(self, columna, orden=None):
        """Aplica Ordinal Encoding a una columna basada en un orden lógico."""
        if orden is None:
            # Si no se provee orden, se usa el orden de aparición o alfabético
            orden = sorted(self.df[columna].unique().tolist())
        
        mapping = {cat: i for i, cat in enumerate(orden)}
        self.ordinal_mappings[columna] = mapping
        
        # Mapear y manejar categorías no encontradas con -1 o un valor por defecto
        self.df[columna] = self.df[columna].map(mapping).fillna(-1).astype(int)
        return "Ordinal Encoding"

    # --- 4. ORQUESTADOR (CLEAN_ALL_ROWS) ---
    def Clean_All_Rows(self, reglas_dict=None, EsPCA=False):
        print(reglas_dict)
        self.df.drop_duplicates(inplace=True)
        reporte = []
        if reglas_dict is None: reglas_dict = {}
        
        # GUARDAR ESTADO: Recordar las reglas introducidas para replicarlas
        self.reglas_entrenamiento = reglas_dict.copy()

        # Separar el ID si existe para que no se procese
        df_id = None
        if self.id_column and self.id_column in self.df.columns:
            df_id = self.df[[self.id_column]].copy()
            self.df.drop(columns=[self.id_column], inplace=True)

        columnas_actuales = [c for c in self.df.columns if c != self.col_target_name]

        for col in columnas_actuales:
            regla = reglas_dict.get(col, {})
            
            if ptypes.is_numeric_dtype(self.df[col]):
                self.Manejo_Atipicos(col)

            res_nulos = self.No_nulos(col, 
                                      metodo=regla.get('metodo'), 
                                      toleranciaSustitucion=regla.get('tolSustitucion', 0.1), 
                                      toleranciaMantenerColumna=regla.get('tolMantenerCols', 0.5),
                                      valorPorDefecto=regla.get('valorDefecto'))
            reporte.append(res_nulos)

            if res_nulos['metodo'] == 'drop-column' or col not in self.df.columns:
                continue

            if regla.get('WOE', False):
                res_woe = self.aplicar_woe_columna(col, bins_count=regla.get('bins_woe', 5))
                if res_woe:
                    reporte[-1]['metodo'] += f" / {res_woe}"
        
            if col in self.woe_mappings:
                continue

            if (ptypes.is_object_dtype(self.df[col]) or ptypes.is_string_dtype(self.df[col]) or ptypes.is_bool_dtype(self.df[col])):
                self.Limpiar_solo_cadenas(col, lematizar=regla.get('Lematizar', False))
                
                es_dummificable = self.SePuedeCategorizar(col, max_categorias=regla.get('MaxDummies', 20))
                if (es_dummificable or (regla.get('Dummies', False))==True):
                    self.df = pd.get_dummies(self.df, columns=[col], drop_first=True)
                    reporte[-1]['metodo'] += ' / Dummies'
                elif regla.get('TargetEncoding', False):
                    self.aplicar_target_encoding(col)
                    reporte[-1]['metodo'] += ' / Target Encoding'
                elif regla.get('Ordinal', False):
                    res_ord = self.aplicar_ordinal_encoding(col, orden=regla.get('orden'))
                    reporte[-1]['metodo'] += f' / {res_ord}'
                elif regla.get('Dummies', False):
                    self.df.drop(columns=[col], inplace=True)
                    reporte[-1]['metodo'] += ' / Borrada (No dummificable)'

        self.df.dropna(subset=[self.col_target_name], inplace=True)
        
        # --- CODIFICACIÓN AUTOMÁTICA DEL TARGET SI ES TEXTO ---
        if ptypes.is_object_dtype(self.df[self.col_target_name]) or ptypes.is_string_dtype(self.df[self.col_target_name]):
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            self.df[self.col_target_name] = le.fit_transform(self.df[self.col_target_name].astype(str))
            
        self.y = self.df[self.col_target_name].copy()
        self.df.drop(columns=[self.col_target_name], inplace=True)
        
        # Asegurar que todas las columnas sean numéricas puras antes del escalado
        # Esto evita el error ufunc 'divide' al tratar con dtypes mixtos o booleanos residuales
        for col in self.df.columns:
            if ptypes.is_numeric_dtype(self.df[col]) or ptypes.is_bool_dtype(self.df[col]):
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce').fillna(0).astype(float)

        # GUARDAR ESTADO: Las columnas finales exactas antes de escalar
        self.columnas_entrenamiento = self.df.columns.tolist()

        cols_num = self.df.select_dtypes(include=[np.number]).columns
        if not cols_num.empty:
            self.scaler = StandardScaler()
            self.df[cols_num] = self.scaler.fit_transform(self.df[cols_num])
            
        if EsPCA:
            # Asegurar que solo haya columnas numéricas para PCA
            self.df = self.df.select_dtypes(include=[np.number])
            self.columnas_entrenamiento = self.df.columns.tolist() # Actualizar columnas para inferencia
            
            if self.df.empty:
                raise ValueError("No quedan columnas numéricas para aplicar PCA.")

            pca_obj = PCA()
            pca_obj.fit(self.df)
            var_acc = np.cumsum(pca_obj.explained_variance_ratio_)
            n_comp = np.argmax(var_acc >= 0.7) + 1
            self.pca = PCA(n_components=n_comp)
            componentes = self.pca.fit_transform(self.df)
            self.df = pd.DataFrame(componentes, columns=[f'PC{i+1}' for i in range(n_comp)], index=self.df.index)

        # No re-adjuntamos el ID aquí para que no se use en el entrenamiento.
        # El ID se manejará únicamente en la fase de predicción/inferencia.
        print(reporte)
        return reporte

    # --- 5. TRANSFORMACIÓN DE DATOS NUEVOS (REESCRITA) ---
    def transformar_nueva_tupla(self, nuevo_df):
        df_pred = nuevo_df.copy()
        
        # Manejo opcional del ID
        df_id_pred = None
        # Si el usuario especificó un id_column en el entrenamiento y este existe en los datos nuevos
        if self.id_column and self.id_column in df_pred.columns:
            df_id_pred = df_pred[[self.id_column]].copy()
            df_pred.drop(columns=[self.id_column], inplace=True)
        
        if self.col_target_name in df_pred.columns:
            df_pred.drop(columns=[self.col_target_name], inplace=True)

        # 1. Aplicar Imputación de Nulos
        for col in self.backup_imputaciones.keys():
            if col in df_pred.columns:
                # Usar imputación oficial si existe, si no, usar la de respaldo
                val_relleno = self.imputaciones_nulos.get(col)
                if val_relleno is None:
                    val_relleno = self.backup_imputaciones.get(col)
                
                if val_relleno is not None:
                    df_pred[col] = df_pred[col].fillna(val_relleno)

        # 2. Textos: Limpieza, Lematización y Separación de Dummies de texto
        for col in list(df_pred.columns):
            regla = self.reglas_entrenamiento.get(col, {})
            
            # Replicar lógica de "Limpiar_solo_cadenas"
            if col in self.columnas_texto_separadas or regla.get('Lematizar', False):
                df_pred[col] = df_pred[col].apply(lambda x: unicodedata.normalize('NFKD', str(x)).encode('ASCII', 'ignore').decode('utf-8') if pd.notnull(x) else x)
                df_pred[col] = df_pred[col].str.lower()
                df_pred[col] = df_pred[col].apply(lambda x: re.sub(r'[^a-zA-Z0-9áéíóúÁÉÍÓÚñÑ\s]', ' ', str(x)) if pd.notnull(x) else x)
                
                if regla.get('Lematizar', False):
                    df_pred[col] = df_pred[col].apply(self.ExtraerTexto)
                    df_pred[col] = df_pred[col].apply(self.lematizar)
            
            # Replicar lógica de "separar_columna"
            if col in self.columnas_texto_separadas:
                df_dummies_texto = df_pred[col].str.get_dummies(sep=' ')
                cols_a_mantener = self.columnas_texto_separadas[col]
                
                # Crear la matriz de texto solo con las palabras que sobrevivieron en entrenamiento
                for c_mantener in cols_a_mantener:
                    if c_mantener not in df_dummies_texto.columns:
                        df_dummies_texto[c_mantener] = 0 # Llenar con 0 si la palabra no viene en los datos nuevos
                
                df_filtrado = df_dummies_texto[cols_a_mantener]
                df_pred = pd.concat([df_pred.drop(columns=[col]), df_filtrado], axis=1)

        # 3. Categorías: Reemplazar por "otros" si son categorías nuevas o no pasaron el umbral en entrenamiento
        for col, categorias_validas in self.categorias_validas.items():
            if col in df_pred.columns:
                df_pred[col] = df_pred[col].where(df_pred[col].isin(categorias_validas), "otros")

        # 4. Target Encoding
        for col, info in self.target_encodings.items():
            if col in df_pred.columns:
                df_pred[col] = df_pred[col].map(info['medias']).fillna(info['global'])

        # --- APLICAR WOE (Weight of Evidence) ---
        for col, mapping in self.woe_mappings.items():
            if col in df_pred.columns:
                binned = pd.cut(df_pred[col], bins=mapping['bins_edges'], include_lowest=True)
                df_pred[col] = binned.map(mapping['woe_dict']).astype(float)

        # --- APLICAR ORDINAL ENCODING ---
        for col, mapping in self.ordinal_mappings.items():
            if col in df_pred.columns:
                df_pred[col] = df_pred[col].map(mapping).fillna(-1).astype(int)

        # 5. Dummies Normales
        df_pred = pd.get_dummies(df_pred)

        # Si una columna de entrenamiento no existe (ej. un dummy), la crea con 0.
        # Si hay columnas extra en los datos nuevos, las borra.
        df_pred = df_pred.reindex(columns=self.columnas_entrenamiento, fill_value=0)

        # Convertir booleanos residuales a enteros
        columnas_bool = df_pred.select_dtypes(include=['bool']).columns
        if not columnas_bool.empty:
            df_pred[columnas_bool] = df_pred[columnas_bool].astype(int)

        # 7. Escalado
        if self.scaler is not None:
            df_pred[self.columnas_entrenamiento] = self.scaler.transform(df_pred[self.columnas_entrenamiento])

        # 8. PCA
        if self.pca is not None:
            componentes = self.pca.transform(df_pred)
            df_pred = pd.DataFrame(componentes, columns=[f'PC{i+1}' for i in range(componentes.shape[1])], index=df_pred.index)
        
        # 9. Re-adjuntar ID si existía
        if df_id_pred is not None:
            df_pred = pd.concat([df_id_pred.reset_index(drop=True), df_pred.reset_index(drop=True)], axis=1)
            
        return df_pred
