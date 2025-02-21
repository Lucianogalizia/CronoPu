 #SUPUESTAMENTE ANDA MUY BIEN ESTE CODIGO. 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import folium
from geopy.distance import geodesic
import ipywidgets as widgets
from IPython.display import display
import re
import streamlit as st  # 🔹 Importar Streamlit


# 🏗️ Crear la interfaz en Streamlit
st.title("CronoPU - Análisis de Pulling 🚛")

# 📌 Permitir que el usuario suba el archivo Excel
uploaded_file = st.file_uploader("📂 Subí el archivo Excel con el cronograma", type=["xlsx"])

if uploaded_file is not None:
    try:
        # ✅ Cargar el archivo Excel subido por el usuario
        df = pd.read_excel(uploaded_file)

        # 🔍 Verificar si el archivo tiene datos
        if df.empty:
            st.error("❌ El archivo está vacío. Subí un archivo válido.")
        else:
            # 🔍 Verificar si tiene las columnas necesarias
            required_columns = ["NETA [M3/D]", "GEO_LATITUDE", "GEO_LONGITUDE", "TIEMPO PLANIFICADO"]
            missing_cols = [col for col in required_columns if col not in df.columns]

            if missing_cols:
                st.error(f"❌ Faltan las siguientes columnas en el archivo: {', '.join(missing_cols)}")
            else:
                # 🔍 Limpieza y conversión optimizada
                for col in required_columns:
                    df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", "."), errors='coerce')

                df.dropna(inplace=True)  # Eliminar valores nulos

                # 📊 Mostrar los primeros datos
                st.write("✅ Archivo cargado con éxito:")
                st.write(df.head())  # Muestra las primeras filas

    except Exception as e:
        st.error(f"❌ Error al procesar el archivo: {e}")
else:
    st.warning("⚠️ Esperando que subas un archivo Excel para analizar.")


# 🔍 Limpieza y conversión optimizada
for col in ["NETA [M3/D]", "GEO_LATITUDE", "GEO_LONGITUDE", "TIEMPO PLANIFICADO"]:
    df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", "."), errors='coerce')

df.dropna(inplace=True)  # Eliminar valores nulos


# 🔹 Función para ordenar correctamente nombres con números y letras
def natural_sort_key(s):
    """ Ordena alfabéticamente considerando números y letras correctamente """
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]


# Filtro por zona
zonas_disponibles = df["ZONA"].unique().tolist()
zona_seleccionada = st.multiselect("Seleccionar Zonas", zonas_disponibles)

if zona_seleccionada:
    df_filtrado = df[df["ZONA"].isin(zona_seleccionada)].copy()
    pozos_disponibles = df_filtrado["POZO"].unique().tolist()
    
    # Selección de cantidad de Pulling
    pulling_count = st.slider("Número de Pulling", min_value=1, max_value=10, value=3)
    
    # Selección de pozos
    pulling_data = {}
    for i in range(1, pulling_count + 1):
        pulling_name = f"Pulling {i}"
        selected_pozo = st.selectbox(f"Seleccionar pozo para {pulling_name}", pozos_disponibles, key=f"pozo_{i}")
        tiempo_restante = st.number_input(f"Tiempo restante en horas para {pulling_name}", min_value=0.0, value=0.0, step=0.1, key=f"tiempo_{i}")
        
        if selected_pozo:
            pulling_data[pulling_name] = {
                "pozo": selected_pozo,
                "tiempo_restante": tiempo_restante,
                "lat": df_filtrado.loc[df_filtrado["POZO"] == selected_pozo, "GEO_LATITUDE"].values[0],
                "lon": df_filtrado.loc[df_filtrado["POZO"] == selected_pozo, "GEO_LONGITUDE"].values[0],
            }
    
    # Confirmar selección
    if st.button("Confirmar Selección"):
        st.success("Selección de pozos confirmada")
        st.write(pulling_data)
    
    # Ingreso de HS Disponibilidad
    hs_disponibilidad = {}
    for pozo in pozos_disponibles:
        hs_disponibilidad[pozo] = st.number_input(f"HS Disponibilidad para {pozo}", min_value=1, max_value=50, value=np.random.randint(1, 51), key=f"hs_{pozo}")
    
    if st.button("Confirmar HS Disponibilidad"):
        st.success("HS Disponibilidad confirmada")
    
    # Proceso de optimización
    if st.button("Ejecutar Proceso"):
        matriz_prioridad = []
        pozos_ocupados = set()
        pulling_lista = list(pulling_data.items())

        def calcular_coeficiente(pozo_referencia, pozo_candidato):
            hs_disp_equipo = hs_disponibilidad.get(pozo_candidato, 0)
            distancia = geodesic(
                (df.loc[df["POZO"] == pozo_referencia, "GEO_LATITUDE"].values[0],
                 df.loc[df["POZO"] == pozo_referencia, "GEO_LONGITUDE"].values[0]),
                (df.loc[df["POZO"] == pozo_candidato, "GEO_LATITUDE"].values[0],
                 df.loc[df["POZO"] == pozo_candidato, "GEO_LONGITUDE"].values[0])
            ).kilometers
            neta = df.loc[df["POZO"] == pozo_candidato, "NETA [M3/D]"].values[0]
            hs_planificadas = df.loc[df["POZO"] == pozo_candidato, "TIEMPO PLANIFICADO"].values[0]
            coeficiente = neta / (hs_planificadas + (distancia * 0.5))
            return coeficiente, distancia

        pulling_asignaciones = {pulling: [] for pulling, _ in pulling_lista}
        for pulling, data in pulling_lista:
            pozo_actual = data["pozo"]
            neta_actual = df.loc[df["POZO"] == pozo_actual, "NETA [M3/D]"].values[0]
            tiempo_restante = data["tiempo_restante"]
            seleccionados = [(pozo, *calcular_coeficiente(pozo_actual, pozo)) for pozo in pozos_disponibles if pozo not in pozos_ocupados]
            seleccionados.sort(key=lambda x: (-x[1], x[2]))
            seleccionados = seleccionados[:3]


            seleccion_n1 = seleccionados[0] if seleccionados else ("N/A", 0, 0)
            seleccion_n2 = seleccionados[1] if len(seleccionados) > 1 else ("N/A", 0, 0)
            seleccion_n3 = seleccionados[2] if len(seleccionados) > 2 else ("N/A", 0, 0)
         
            recomendacion = "Continuar en pozo actual" if len(seleccionados) == 0 else "Abandonar pozo y moverse al mejor candidato"

            matriz_prioridad.append([
                pulling, pozo_actual, neta_actual, tiempo_restante,
                *seleccion_n1, *seleccion_n2, *seleccion_n3,
                         
        
        columns = [
            "Pulling", "Pozo Actual", "Neta Actual", "Tiempo Restante (h)",
            "N+1", "Coeficiente N+1", "Distancia N+1 (km)",
            "N+2", "Coeficiente N+2", "Distancia N+2 (km)",
            "N+3", "Coeficiente N+3", "Distancia N+3 (km)", "Recomendación"
        ]
        df_prioridad = pd.DataFrame(matriz_prioridad, columns=columns)
        st.dataframe(df_prioridad)
