

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
from streamlit.components.v1 import html
import tempfile
import os
import math
import plotly.graph_objects as go
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
            <h3>🩸 Aproximación del perfil de velocidad del flujo sanguíneo </h3>
            <p>Calcula la resonancia del circuito.</p>
        </div>
        """, unsafe_allow_html=True)
        

