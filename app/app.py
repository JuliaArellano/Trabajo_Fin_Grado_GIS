

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
from streamlit.components.v1 import html
import tempfile
import os
import math
import trimesh
import plotly.graph_objects as go
import pythreejs as three

# Configuración general
st.set_page_config(page_title="Visualización del Stent Inteligente", layout="wide")

# Inicializa 'vista_activa' si no existe
if "vista_activa" not in st.session_state:
    st.session_state.vista_activa = "Inicio"

# Estilo CSS general para nombre y logo
st.markdown(f"""
    <style>
    section[data-testid="stSidebar"] {{
        background-color: #002B5B;
        padding: 20px 10px;
        position: relative;
    }}
    .sidebar-title {{
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 16px;
        color: white;
    }}
    .sidebar-button {{
        display: flex;
        align-items: center;
        width: 100%;
        height: 50px;
        padding: 0 16px;
        margin-bottom: 10px;
        background-color: #00509D;
        color: white;
        border: none;
        border-radius: 12px;
        font-size: 16px;
        cursor: pointer;
        transition: background-color 0.2s ease;
    }}
    .sidebar-button:hover {{
        background-color: #1A73E8;
    }}
    .sidebar-button.selected {{
        background-color: #FF4B4B;
        color: white;
    }}
    .header-bar {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 14px;
        color: white;
        margin-bottom: 10px;
        background-color: #002B5B;
        padding: 10px 20px;
    }}
    </style>
    <div class="header-bar">
        <div>Visualización del Stent Inteligente</div>
        <div>Julia Arellano Atienza | Ingeniería de la Salud</div>
    </div>
""", unsafe_allow_html=True)

# Diccionario de vistas
vistas = {
    "Inicio": "\U0001F3E0",
    "Vista 3D del Stent": "\U0001F9CA",
    "Expansión térmica": "\U0001F321️",
    "Velocidad del flujo sanguíneo":  "\U0001FA78",
    "Parámetros del Circuito LC": "\u2699",
    "Factor de calidad del Circuito LC":"\U0001F50D",
    "Cálculo de Tensiones y Factor de Seguridad": "\U000026D3"

}

# Creación de la barra lateral
st.sidebar.markdown('<div class="sidebar-title">Índice</div>', unsafe_allow_html=True)
for vista, icono in vistas.items():
    clase = "sidebar-button"
    if st.session_state.vista_activa == vista:
        clase += " selected"
    if st.sidebar.button(f"{icono} {vista}", key=vista):
        st.session_state.vista_activa = vista
def cargar_visualizar_stl(uploaded_files):
    for archivo in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".stl") as tmp_file:
            tmp_file.write(archivo.read())
            tmp_file.flush()
            mesh = pv.read(tmp_file.name)

            if mesh.n_points == 0:
                st.warning(f"El archivo {archivo.name} está vacío o no se pudo cargar.")
                continue

            st.subheader(f"Modelo: {archivo.name}")
            
            # Crear el plotter interactivo de PyVista
            plotter = pv.Plotter(off_screen=False, window_size=[500, 500])
            plotter.add_mesh(mesh, color="lightblue")
            plotter.add_axes()
            plotter.set_background("white")

            # Generar el archivo HTML de la visualización interactiva
            html_file = tmp_file.name + ".html"
            plotter.export_html(html_file)

            # Mostrar la visualización interactiva en Streamlit
            with open(html_file, "r") as f:
                st.components.v1.html(f.read(), height=600)

            # Eliminar el archivo HTML temporal
            os.remove(html_file)
if st.session_state.vista_activa == "Inicio":
    st.title("Visualización del Stent Inteligente")

    st.markdown("""
    <div style="background-color: #0d47a1; padding: 30px; border-radius: 15px; color: white; text-align: center;">
        <h1>BIENVENIDO</h1>
        <p style="font-size:20px;">Explora el proyecto del <b>Stent Inteligente de Nitinol</b> de manera interactiva.</p><br>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("   ")

    st.markdown(" ### Funcionalidades principales:")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div style="background-color:#f1f8e9; padding: 20px; border-radius:12px; text-align:center;">
            <h3>🧊 Vista 3D</h3>
            <p>Explora el modelo 3D del stent.</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""<br>
        <div style="background-color:#fce4ec; padding: 20px; border-radius: 12px; text-align: center;">
            <h3>🌡️ Expansión térmica</h3>
            <p>Simula la expansión con la temperatura.</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""<br>
        <div style="background-color:#e3f2fd ; padding: 20px; border-radius: 12px; text-align: center;">
            <h3>🩸 Velocidad del flujo sanguíneo </h3>
            <p>Calcula la velocidad , la caída de presión y FFR del flujo en el stent .</p>
        </div>
        """, unsafe_allow_html=True)
elif st.session_state.vista_activa == "Vista 3D del Stent":
    st.title("🧊 Vista 3D del Stent")
    st.markdown("Puedes subir uno o más archivos STL del stent para visualizar su estructura.")
    # Subir archivo STL

    def cargar_y_procesar_stl(uploaded_file):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".stl") as tmp:
            tmp.write(uploaded_file.read())
            tmp.flush()
            mesh = trimesh.load(tmp.name)
    
        return mesh
    
    def mostrar_modelo_stl(nombre_archivo, mesh):
        vertices = mesh.vertices
        faces = mesh.faces
    
        fig = go.Figure(data=[
            go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                color='#B0B0B0',  # Gris metálico
                opacity=1.0,
                flatshading=True,
                lighting=dict(
                    ambient=0.5,
                    diffuse=0.5,
                    specular=0.2,
                    roughness=0.9,
                    fresnel=0.0
                ),
                lightposition=dict(x=100, y=200, z=0)
            )
        ])
    
        # Actualizar el layout para permitir interacción con la cámara, pero ocultar los ejes
        fig.update_layout(
            title=f"Modelo: {nombre_archivo}",
            scene=dict(
                aspectmode='data',
                xaxis=dict(visible=False),  # Ejes ocultos
                yaxis=dict(visible=False),  # Ejes ocultos
                zaxis=dict(visible=False),  # Ejes ocultos
                camera=dict(eye=dict(x=2, y=2, z=2))  # Inicia la cámara en una buena posición para rotar
            ),
            margin=dict(l=0, r=0, t=30, b=0),
            height=700
        )
    
        st.subheader(nombre_archivo)
        st.plotly_chart(fig, use_container_width=True)
    
    
    uploaded_files = st.file_uploader("Sube uno o varios archivos STL", type=["stl"], accept_multiple_files=True)
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            mesh = cargar_y_procesar_stl(uploaded_file)
            mostrar_modelo_stl(uploaded_file.name, mesh)




        

