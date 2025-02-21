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
            st.stop()
        else:
            # 🔍 Verificar si tiene las columnas necesarias
            required_columns = ["NETA [M3/D]", "GEO_LATITUDE", "GEO_LONGITUDE", "TIEMPO PLANIFICADO"]
            missing_cols = [col for col in required_columns if col not in df.columns]

            if missing_cols:
                st.error(f"❌ Faltan las siguientes columnas en el archivo: {', '.join(missing_cols)}")
                st.stop()

            # 🔍 Limpieza y conversión optimizada
            for col in required_columns:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", "."), errors='coerce')

            df.dropna(inplace=True)  # Eliminar valores nulos

            # 📊 Mostrar los primeros datos
            st.write("✅ Archivo cargado con éxito:")
            st.write(df.head())  # Muestra las primeras filas

            # Guardar el DataFrame en session_state
            st.session_state.df = df

    except Exception as e:
        st.error(f"❌ Error al procesar el archivo: {e}")
        st.stop()  # Detener ejecución si hay un error grave

else:
    st.warning("⚠️ Esperando que subas un archivo Excel para analizar.")
    st.stop()

# ──────────────────────────────────────────
# 4. Limpieza y conversión de datos
# ──────────────────────────────────────────
for col in required_columns:
    df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", "."), errors='coerce')
df.dropna(inplace=True)

# Guardar el DataFrame en session_state
st.session_state.df = df


# === 1. FILTRO POR ZONA Y SELECCIÓN DE N° PULLING ===
    st.header("1. Filtrado de Zonas y Selección de Pulling")
    # Aquí continúa el resto de la lógica...
    # Selección de zonas disponibles
    zonas_disponibles = st.session_state.df["ZONA"].unique().tolist()
    zonas_seleccionadas = st.multiselect("Selecciona las zonas:", options=zonas_disponibles)
 
# Selección de cantidad de Pulling
pulling_count = st.slider("Número de Pulling:", min_value=1, max_value=10, value=3)
 
if st.button("Filtrar Zonas"):
    if not zonas_seleccionadas:
        st.error("Debes seleccionar al menos una zona.")
    else:
        # Filtrar DataFrame por las zonas seleccionadas y ordenar la lista de pozos
        df_filtrado = st.session_state.df[st.session_state.df["ZONA"].isin(zonas_seleccionadas)].copy()
        st.session_state.df_filtrado = df_filtrado
        pozos = df_filtrado["POZO"].unique().tolist()
        st.session_state.pozos_disponibles = sorted(pozos)
        st.success(f"Zonas seleccionadas: {', '.join(zonas_seleccionadas)}")
 
# === 2. SELECCIÓN DE PULLING (POZOS Y TIEMPO RESTANTE) ===
if st.session_state.df_filtrado is not None:
    st.header("2. Selección de Pozos para Pulling")
    pulling_data = {}
    
    with st.form("form_pulling"):
        st.subheader("Selecciona el pozo y define el tiempo restante para cada Pulling")
        # Para cada pulling se crea una sección con un selectbox y un número de entrada
        for i in range(1, pulling_count + 1):
            st.markdown(f"**Pulling {i}**")
            # Usamos selectbox para escoger el pozo de la lista de pozos disponibles
            pozo_seleccion = st.selectbox(f"Pozo para Pulling {i}:", options=st.session_state.pozos_disponibles, key=f"pulling_pozo_{i}")
            tiempo_restante = st.number_input(f"Tiempo restante (h) para Pulling {i}:", min_value=0.0, value=0.0, key=f"pulling_tiempo_{i}")
            pulling_data[f"Pulling {i}"] = {
                "pozo": pozo_seleccion,
                "tiempo_restante": tiempo_restante,
                # Se agregarán latitud y longitud más adelante, usando df_filtrado
            }
            st.markdown("---")
        submitted = st.form_submit_button("Confirmar Selección de Pulling")
    
    if submitted:
        # Verificar que no se haya seleccionado el mismo pozo más de una vez
        seleccionados = [data["pozo"] for data in pulling_data.values()]
        if len(seleccionados) != len(set(seleccionados)):
            st.error("Error: No puedes seleccionar el mismo pozo para más de un pulling.")
        else:
            # Agregar latitud y longitud obtenidos del DataFrame filtrado
            for pulling, data in pulling_data.items():
                pozo = data["pozo"]
                registro = st.session_state.df_filtrado.loc[st.session_state.df_filtrado["POZO"] == pozo].iloc[0]
                data["lat"] = registro["GEO_LATITUDE"]
                data["lon"] = registro["GEO_LONGITUDE"]
            st.session_state.pulling_data = pulling_data
            # Actualizar lista de pozos disponibles (quitando los seleccionados)
            todos_pozos = st.session_state.df_filtrado["POZO"].unique().tolist()
            st.session_state.pozos_disponibles = sorted([p for p in todos_pozos if p not in seleccionados])
            st.success("Selección de Pulling confirmada.")
 
# === 3. INGRESO DE HS DISPONIBILIDAD ===
if st.session_state.pulling_data is not None:
    st.header("3. Ingreso de HS Disponibilidad de Equipo")
    
    if not st.session_state.pozos_disponibles:
        st.error("No hay pozos disponibles para asignar HS.")
    else:
        hs_disponibilidad = {}
        with st.form("form_hs"):
            st.subheader("Ingresa la disponibilidad de HS para cada pozo")
            for pozo in st.session_state.pozos_disponibles:
                # Se asigna un valor por defecto aleatorio entre 1 y 50
                hs_val = st.number_input(f"{pozo} (HS):", min_value=0.0, value=float(np.random.randint(1, 51)), key=f"hs_{pozo}")
                hs_disponibilidad[pozo] = hs_val
            hs_submitted = st.form_submit_button("Confirmar HS Disponibilidad")
        if hs_submitted:
            st.session_state.hs_disponibilidad = hs_disponibilidad
            st.success("HS Disponibilidad confirmada.")
 
# === 4. EJECUCIÓN DEL PROCESO DE ASIGNACIÓN ===
def ejecutar_proceso():
    """Función que ejecuta la asignación de pozos y genera la matriz de prioridad."""
    matriz_prioridad = []
    pozos_ocupados = set()
    pulling_lista = list(st.session_state.pulling_data.items())
    
    # Función que calcula el coeficiente y la distancia entre dos pozos
    def calcular_coeficiente(pozo_referencia, pozo_candidato):
        hs_disp_equipo = st.session_state.hs_disponibilidad.get(pozo_candidato, 0)
        # Obtener coordenadas de ambos pozos (usando el DataFrame original)
        registro_ref = st.session_state.df.loc[st.session_state.df["POZO"] == pozo_referencia].iloc[0]
        registro_cand = st.session_state.df.loc[st.session_state.df["POZO"] == pozo_candidato].iloc[0]
        distancia = geodesic(
            (registro_ref["GEO_LATITUDE"], registro_ref["GEO_LONGITUDE"]),
            (registro_cand["GEO_LATITUDE"], registro_cand["GEO_LONGITUDE"])
        ).kilometers
        neta = registro_cand["NETA [M3/D]"]
        hs_planificadas = registro_cand["TIEMPO PLANIFICADO"]
        coeficiente = neta / (hs_planificadas + (distancia * 0.5))
        return coeficiente, distancia
    
    # Función para asignar pozos adicionales a cada pulling
    def asignar_pozos(pulling_asignaciones, nivel):
        no_asignados = [p for p in st.session_state.pozos_disponibles if p not in pozos_ocupados]
        for pulling, data in pulling_lista:
            # Para el primer candidato se usa el pozo actual o el último asignado
            pozo_referencia = pulling_asignaciones[pulling][-1][0] if pulling_asignaciones[pulling] else data["pozo"]
            # Seleccionar candidatos que cumplan la condición de disponibilidad de HS
            candidatos = []
            for pozo in no_asignados:
                # Sumar los tiempos planificados de los pozos ya asignados en este pulling
                tiempo_acumulado = sum(
                    st.session_state.df.loc[st.session_state.df["POZO"] == p[0], "TIEMPO PLANIFICADO"].iloc[0]
                    for p in pulling_asignaciones[pulling]
                )
                # Condición: la disponibilidad de HS debe ser menor o igual al tiempo restante + tiempo acumulado
                if st.session_state.hs_disponibilidad.get(pozo, 0) <= (data["tiempo_restante"] + tiempo_acumulado):
                    coef, dist = calcular_coeficiente(pozo_referencia, pozo)
                    candidatos.append((pozo, coef, dist))
            # Ordenar candidatos: mayor coeficiente y menor distancia
            candidatos.sort(key=lambda x: (-x[1], x[2]))
            if candidatos:
                mejor_candidato = candidatos[0]
                pulling_asignaciones[pulling].append(mejor_candidato)
                pozos_ocupados.add(mejor_candidato[0])
                if mejor_candidato[0] in no_asignados:
                    no_asignados.remove(mejor_candidato[0])
            else:
                st.warning(f"⚠️ No hay pozos disponibles para asignar como {nivel} en {pulling}.")
        return pulling_asignaciones
 
    # Inicializar asignaciones para cada pulling
    pulling_asignaciones = {pulling: [] for pulling, _ in pulling_lista}
    # Se asignan tres rondas (N+1, N+2, N+3)
    pulling_asignaciones = asignar_pozos(pulling_asignaciones, "N+1")
    pulling_asignaciones = asignar_pozos(pulling_asignaciones, "N+2")
    pulling_asignaciones = asignar_pozos(pulling_asignaciones, "N+3")
    
    # Construcción de la matriz de prioridad
    for pulling, data in pulling_lista:
        pozo_actual = data["pozo"]
        registro_actual = st.session_state.df.loc[st.session_state.df["POZO"] == pozo_actual].iloc[0]
        neta_actual = registro_actual["NETA [M3/D]"]
        tiempo_restante = data["tiempo_restante"]
        # Se toman los tres primeros candidatos asignados
        seleccionados = pulling_asignaciones.get(pulling, [])[:3]
        # Si hay menos de tres candidatos, se agregan valores por defecto para evitar errores
        while len(seleccionados) < 3:
            seleccionados.append(("N/A", 1, 1))
        
        # Validación de tiempo planificado para el primer candidato (N+1)
        tiempo_planificado_n1 = st.session_state.df.loc[st.session_state.df["POZO"] == seleccionados[0][0], "TIEMPO PLANIFICADO"]
        tiempo_planificado_n1 = tiempo_planificado_n1.iloc[0] if not tiempo_planificado_n1.empty else 1
        
        # Cálculo de suspensión y recomendación
        suspension = (neta_actual / seleccionados[0][1]) * (seleccionados[0][2] * 0.5 + tiempo_planificado_n1)
        recomendacion = "Abandonar pozo actual y moverse al N+1" if (neta_actual / tiempo_restante) < suspension else "Continuar en pozo actual"
        
        matriz_prioridad.append([
            pulling, pozo_actual, neta_actual, tiempo_restante,
            seleccionados[0][0], seleccionados[0][1], seleccionados[0][2],
            seleccionados[1][0], seleccionados[1][1], seleccionados[1][2],
            seleccionados[2][0], seleccionados[2][1], seleccionados[2][2],
            recomendacion
        ])
    
    columns = [
        "Pulling", "Pozo Actual", "Neta Actual", "Tiempo Restante (h)",
        "N+1", "Coeficiente N+1", "Distancia N+1 (km)",
        "N+2", "Coeficiente N+2", "Distancia N+2 (km)",
        "N+3", "Coeficiente N+3", "Distancia N+3 (km)", "Recomendación"
    ]
    
    df_prioridad = pd.DataFrame(matriz_prioridad, columns=columns)
    st.session_state.df_prioridad = df_prioridad
    st.success("Proceso de asignación completado.")
    st.dataframe(df_prioridad)


