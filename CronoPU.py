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


# 📋 Filtro por ZONA (Seleccionar múltiples zonas)
zonas_disponibles = df["ZONA"].unique().tolist()
zona_dropdown = widgets.SelectMultiple(options=zonas_disponibles, description="Zona:")

# 📋 Selección de cantidad de Pulling (Se aplica sobre TODAS las zonas combinadas)
pulling_count = widgets.IntSlider(value=3, min=1, max=10, description="N° Pulling")

confirm_zona_button = widgets.Button(description="Filtrar Zona")

def filtrar_zona(b):
    global df_filtrado, pozos_disponibles
    zonas_seleccionadas = list(zona_dropdown.value)  # Convertimos la selección en lista
    
    if not zonas_seleccionadas:
        display(widgets.HTML("<h3 style='color:red;'>Debes seleccionar al menos una zona.</h3>"))
        return
    
    # Filtrar los datos basados en las zonas seleccionadas (COMBINANDO LAS ZONAS)
    df_filtrado = df[df["ZONA"].isin(zonas_seleccionadas)].copy()
    pozos_disponibles = df_filtrado["POZO"].unique().tolist()

    display(widgets.HTML(f"<h3 style='color:green;'>Zonas seleccionadas: {', '.join(zonas_seleccionadas)} ✔️</h3>"))
    
    # Llamamos a la función para seleccionar los Pulling después de filtrar la zona
    seleccionar_pulling()

confirm_zona_button.on_click(filtrar_zona)

# Mostrar los widgets
display(zona_dropdown)
display(pulling_count)
display(confirm_zona_button)


# 🏗️ SELECCIÓN DE PULLING (Ahora toma la cantidad total y no por cada zona)


def seleccionar_pulling():
    global pulling_widgets, pulling_data
    pulling_widgets = {}
    pulling_data = {}

    if len(pozos_disponibles_ordenados) < pulling_count.value:
        display(widgets.HTML(f"<h3 style='color:red;'>Advertencia: Has seleccionado más Pulling ({pulling_count.value}) que pozos disponibles ({len(pozos_disponibles_ordenados)}). Ajusta la cantidad.</h3>"))
        return

    display(widgets.HTML("<h2>Selecciona los pozos actuales y tiempos restantes</h2>"))

    for i in range(1, pulling_count.value + 1):
        pulling_name = f"Pulling {i}"
        label = widgets.Label(value=pulling_name)

        # 🔹 Barra de búsqueda de pozos
        search_box = widgets.Text(
            placeholder="Buscar pozo...",
            description="🔍",
            layout=widgets.Layout(width='300px')
        )

        # Dropdown independiente para cada pulling
        dropdown = widgets.Dropdown(
            options=pozos_disponibles_ordenados,
            description="Pozo:",
            layout=widgets.Layout(width='300px')
        )

        def actualizar_dropdown(cambio, dd=dropdown):
            """ Filtra la lista del Dropdown según lo que escriba el usuario """
            filtro = cambio['new'].lower()
            opciones_filtradas = [pozo for pozo in pozos_disponibles_ordenados if filtro in pozo.lower()]
            dd.options = opciones_filtradas if opciones_filtradas else ["No encontrado"]

        search_box.observe(actualizar_dropdown, names='value')

        tiempo_restante_entry = widgets.FloatText(value=0.0, description="Tiempo restante (h):")

        pulling_widgets[pulling_name] = {"dropdown": dropdown, "tiempo_restante": tiempo_restante_entry}

        display(label, search_box, dropdown, tiempo_restante_entry)

    confirm_button = widgets.Button(description="Confirmar Selección")
    confirm_button.on_click(confirmar_pozos)
    display(confirm_button)
    
    
def confirmar_pozos(b):
    global pulling_data, pozos_disponibles
    seleccionados = [widget["dropdown"].value for widget in pulling_widgets.values()]
    
    if len(seleccionados) != len(set(seleccionados)):  # Verifica duplicados
        display(widgets.HTML("<h3 style='color:red;'>Error: No puedes seleccionar el mismo pozo para más de un pulling.</h3>"))
        return
    
    pulling_data = {
        pulling: {
            "pozo": widget["dropdown"].value,
            "tiempo_restante": widget["tiempo_restante"].value,
            "lat": df_filtrado.loc[df_filtrado["POZO"] == widget["dropdown"].value, "GEO_LATITUDE"].values[0],
            "lon": df_filtrado.loc[df_filtrado["POZO"] == widget["dropdown"].value, "GEO_LONGITUDE"].values[0],
        }
        for pulling, widget in pulling_widgets.items()
    }
    
    pozos_disponibles = [p for p in df_filtrado["POZO"].unique().tolist() if p not in seleccionados]
    
    display(widgets.HTML("<h3 style='color:green;'>Selección confirmada ✔️</h3>"))
    mostrar_interfaz_hs()


def seleccionar_pulling():
    global pulling_widgets, pulling_data
    pulling_widgets = {}
    pulling_data = {}
    
    display(widgets.HTML("<h2>Selecciona los pozos actuales y tiempos restantes</h2>"))
    for i in range(1, pulling_count.value + 1):
        pulling_name = f"Pulling {i}"
        label = widgets.Label(value=pulling_name)
        dropdown = widgets.Dropdown(options=pozos_disponibles, description="Pozo:")
        tiempo_restante_entry = widgets.FloatText(value=0.0, description="Tiempo restante (h):")
        pulling_widgets[pulling_name] = {"dropdown": dropdown, "tiempo_restante": tiempo_restante_entry}
        display(label, dropdown, tiempo_restante_entry)
    confirm_button = widgets.Button(description="Confirmar Selección")
    confirm_button.on_click(confirmar_pozos)
    display(confirm_button)


   
    
# Confirmación de pozos seleccionados
def confirmar_pozos(b):
    global pulling_data, pozos_disponibles
    seleccionados = [widget["dropdown"].value for widget in pulling_widgets.values()]

    if len(seleccionados) != len(set(seleccionados)):  # Verifica duplicados
        display(widgets.HTML("<h3 style='color:red;'>Error: No puedes seleccionar el mismo pozo para más de un pulling.</h3>"))
        return

    pulling_data = {
        pulling: {
            "pozo": widget["dropdown"].value,
            "tiempo_restante": widget["tiempo_restante"].value,
            "lat": df_filtrado.loc[df_filtrado["POZO"] == widget["dropdown"].value, "GEO_LATITUDE"].values[0],
            "lon": df_filtrado.loc[df_filtrado["POZO"] == widget["dropdown"].value, "GEO_LONGITUDE"].values[0],
        }
        for pulling, widget in pulling_widgets.items()
    }

    pozos_disponibles = [p for p in df_filtrado["POZO"].unique().tolist() if p not in seleccionados]
    display(widgets.HTML("<h3 style='color:green;'>Selección confirmada ✔️</h3>"))
    mostrar_interfaz_hs()

# 🏗️ 3) Ingreso de HS Disponibilidad de Equipo
def mostrar_interfaz_hs():
    global hs_disponibilidad, pozo_widgets
    hs_disponibilidad = {}
    pozo_widgets = {}
    
    if not pozos_disponibles:
        display(widgets.HTML("<h3 style='color:red;'>No hay pozos disponibles para asignar HS.</h3>"))
        return

    display(widgets.HTML("<h2>Ingresa la disponibilidad de equipos por pozo</h2>"))
    
    for pozo in pozos_disponibles:
        label = widgets.Label(value=pozo)
        hs_entry = widgets.FloatText(value=np.random.randint(1, 51), description=f"{pozo} (HS):")
        pozo_widgets[pozo] = hs_entry
        display(label, hs_entry)
    
    confirm_button_hs = widgets.Button(description="Confirmar HS Disponibilidad")
    confirm_button_hs.on_click(confirmar_hs)
    display(confirm_button_hs)

# Confirmación de HS
def confirmar_hs(b):
    global hs_disponibilidad
    hs_disponibilidad = {pozo: widget.value for pozo, widget in pozo_widgets.items()}
    display(widgets.HTML("<h3 style='color:green;'>HS Disponibilidad confirmada ✔️</h3>"))
    ejecutar_proceso()


def ejecutar_proceso():
    matriz_prioridad = []
    pozos_ocupados = set()
    pulling_lista = list(pulling_data.items())
    num_pulling = len(pulling_lista)
    
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
    
    def asignar_pozos(pulling_asignaciones, nivel):
        no_asignados = [p for p in pozos_disponibles if p not in pozos_ocupados]
        for pulling, data in pulling_lista:
            pozo_referencia = pulling_asignaciones[pulling][-1][0] if pulling_asignaciones[pulling] else data["pozo"]
            candidatos = [(pozo, *calcular_coeficiente(pozo_referencia, pozo)) for pozo in no_asignados
                          if hs_disponibilidad.get(pozo, 0) <= (data["tiempo_restante"] + sum(df.loc[df["POZO"] == p[0], "TIEMPO PLANIFICADO"].values[0] for p in pulling_asignaciones[pulling]))]
            
            candidatos.sort(key=lambda x: (-x[1], x[2]))
            
            if candidatos:
                mejor_candidato = candidatos[0]
                pulling_asignaciones[pulling].append(mejor_candidato)
                pozos_ocupados.add(mejor_candidato[0])  # Marcar el pozo como ocupado globalmente
                no_asignados.remove(mejor_candidato[0])  # Eliminarlo de la lista de disponibles inmediatamente
            else:
                print(f"⚠️ Advertencia: No hay pozos disponibles para asignar como {nivel} en {pulling}")
        
        return pulling_asignaciones

    pulling_asignaciones = {pulling: [] for pulling, _ in pulling_lista}
    pulling_asignaciones = asignar_pozos(pulling_asignaciones, "N+1")
    pulling_asignaciones = asignar_pozos(pulling_asignaciones, "N+2")
    pulling_asignaciones = asignar_pozos(pulling_asignaciones, "N+3")
    
    for pulling, data in pulling_lista:
        pozo_actual = data["pozo"]
        neta_actual = df.loc[df["POZO"] == pozo_actual, "NETA [M3/D]"].values[0]
        tiempo_restante = data["tiempo_restante"]
        seleccionados = pulling_asignaciones.get(pulling, [])[:3]

        while len(seleccionados) < 3:
            seleccionados.append(("N/A", 1, 1, 1, 1))  # Evitar errores con "N/A"
        
        # Validación de tiempo planificado
        tiempo_planificado_n1 = df.loc[df["POZO"] == seleccionados[0][0], "TIEMPO PLANIFICADO"].values
        tiempo_planificado_n1 = tiempo_planificado_n1[0] if len(tiempo_planificado_n1) > 0 else 1  # Evita errores si el valor no existe

        suspension = (neta_actual / seleccionados[0][1]) * (seleccionados[0][2] * 0.5 + tiempo_planificado_n1)

        # Nueva lógica para la recomendación
        recomendacion = "Abandonar pozo actual y moverse al N+1" if (neta_actual / tiempo_restante) < suspension else "Continuar en pozo actual"

        matriz_prioridad.append([
            pulling, pozo_actual, neta_actual, tiempo_restante,
            seleccionados[0][0], seleccionados[0][1], seleccionados[0][2],
            seleccionados[1][0], seleccionados[1][1], seleccionados[1][2],
            seleccionados[2][0], seleccionados[2][1], seleccionados[2][2], recomendacion
        ])

    columns = [
        "Pulling", "Pozo Actual", "Neta Actual", "Tiempo Restante (h)",
        "N+1", "Coeficiente N+1", "Distancia N+1 (km)",
        "N+2", "Coeficiente N+2", "Distancia N+2 (km)",
        "N+3", "Coeficiente N+3", "Distancia N+3 (km)", "Recomendación"
    ]
    
    global df_prioridad
    df_prioridad = pd.DataFrame(matriz_prioridad, columns=columns)
    display(df_prioridad)
