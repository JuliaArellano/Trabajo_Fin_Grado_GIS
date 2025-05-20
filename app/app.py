

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
from plotly.subplots import make_subplots

# Configuración general
st.set_page_config(page_title="Visualización del Stent Inteligente", layout="wide")

# Inicializa 'vista_activa' si no existe
if "vista_activa" not in st.session_state:
    st.session_state.vista_activa = "Inicio"

# Estilo CSS general para nombre y logo
st.markdown(f"""
    <style>
    section[data-testid="stSidebar"] {{
        background-color: #006F29;
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
        background-color: #544BFF;
        color: white;
    }}
    .header-bar {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 14px;
        color: white;
        margin-bottom: 10px;
        background-color: #006F29;
        padding: 10px 20px;
    }}
    </style>
    <div class="header-bar">
        <div>Visualización del Stent Inteligente</div>
        <div>Julia Arellano Atienza | Ingeniería de la Salud</div>
        <div>Contacto: jaa1018@alu.ubu.es</div>    
    </div>
""", unsafe_allow_html=True)


# Diccionario de vistas
vistas = {
    "Inicio": "\U0001F3E0",
    "Vista 3D del Stent": "\U0001F9CA",
    "Expansión térmica": "\U0001F321️",
    "Velocidad del flujo sanguíneo":  "\U0001FA78",
    "Parámetros del Circuito LC": "\u2699",
    "Análisis del Sistema de Comunicación":"\U0001F50D",
    "Análisis mecánico del stent": "\U0001F529"

}

# Creación de la barra lateral
st.sidebar.markdown('<div class="sidebar-title">Índice</div>', unsafe_allow_html=True)
for vista, icono in vistas.items():
    clase = "sidebar-button"
    if st.session_state.vista_activa == vista:
        clase += " selected"
    if st.sidebar.button(f"{icono} {vista}", key=vista):
        st.session_state.vista_activa = vista
        
# Funciones utilizadas 
def cargar_modelo_predeterminado():
    base_path = os.path.dirname(os.path.abspath(__file__))  # Ruta de app.py
    # Los archivos están en la misma carpeta, así que directamente los nombramos
    modelo_1 = os.path.join(base_path, "stent_final.stl")
    modelo_2 = os.path.join(base_path, "Sensor_completo.stl")

    return [modelo_1, modelo_2]
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
        st.plotly_chart(fig, use_container_width=True)
    
    # Función de expansión
def expansion(tipo, T_min, T_max, D_inicial, As, Af, alpha_m, alpha_a): 
    T = np.linspace(T_min, T_max, 500)
    D = np.zeros_like(T)
    for i, temp in enumerate(T):
        if temp < As:
            delta_T = temp - T_min
            D[i] = D_inicial * (1 + alpha_m * delta_T)
        elif As <= temp <= Af:
            f = (temp - As) / (Af - As)
            fraccion_transformada = 0.5 * (1 - np.cos(np.pi * f))
            delta_T1 = As - T_min
            delta_T2 = temp - As
            expansion_termica = (1 + alpha_m * delta_T1) * (1 + alpha_a * delta_T2)
            D[i] = D_inicial * expansion_termica * (1 + fraccion_transformada)
        else:
            delta_T1 = As - T_min
            delta_T2 = temp - As
            expansion_termica = (1 + alpha_m * delta_T1) * (1 + alpha_a * delta_T2)
            fraccion_transformada = 1
            D[i] = D_inicial * expansion_termica * (1 + fraccion_transformada)

    return T, D
    # Función para calcular el flujo sanguíneo
# Función para calcular el flujo sanguíneo
def flujo(Q, R_stent, L, mu, P_entrada):
    delta_P = (8 * mu * L * Q) / (np.pi * R_stent**4)  # Caída de presión
    v_prom = Q / (np.pi * R_stent**2)  # Velocidad promedio 
    P_salida = P_entrada - delta_P  # Presión en salida

    # Perfil de velocidades
    total_points = 100
    r = np.linspace(0, R_stent, total_points)
    v = (1 / (4 * mu)) * (-delta_P / L) * (R_stent**2 - r**2)
    v = np.abs(v)  # Asegurar valores positivos

    # Calcular velocidad máxima en el centro
    v_max = np.max(v)

    #Cálcular FFR o iFR
    ffr= P_salida/P_entrada
    plt.figure(figsize=(10, 6))

    # Perfil de velocidad
    plt.plot(v, r * 1000, label='Perfil de velocidad', color='blue')  # r en mm
    
    # Marcar velocidad máxima en el centro
    plt.scatter([v_max], [0], color='red', zorder=5, label="Velocidad máxima (centro)")
    plt.text(v_max + 0.005, 0, f"Max: {v_max:.5f} m/s", color='red')
    
    # Marcar velocidad cero en el borde
    plt.scatter([0], [R_stent * 1000], color='green', zorder=5, label="Velocidad en la pared (borde)")
    plt.text(0.005, R_stent * 1000 + 0.5, "V_borde: 0 m/s", color='green')
    
    # Configurar el gráfico
    plt.xlabel('Velocidad (m/s)')
    plt.ylabel('Radio (mm)')
    
    plt.title('Perfil de Velocidad de Poiseuille en el Stent')
    plt.legend()
    plt.grid(True)
    plt.gca().invert_yaxis()  # Invertir eje Y para que el centro esté arriba
    plt.show()
    return delta_P, v_prom, P_salida, v, r, v_max,ffr 
def calcular_inductancia(r_cm, l_cm, N):
    L_uH = (r_cm**2 * N**2) / (9 * r_cm + 10 * l_cm)
    return L_uH * 1e-6  # Convertir a Henrios

def calcular_capacitancia(A_electrodo_m2, d_poliamida_m, num_pares_electrodos):
    epsilon_0 = 8.854e-12  # F/m (permitividad del vacío)
    epsilon_r_poliamida = 3.2  # Permitividad relativa de la poliamida
    C_par = epsilon_0 * epsilon_r_poliamida * A_electrodo_m2 / d_poliamida_m
    return num_pares_electrodos * C_par * 2

def calcular_frecuencia_resonancia(L, C):
    return 1 / (2 * np.pi * np.sqrt(L * C))
if st.session_state.vista_activa == "Inicio":
    st.title("Visualización del Stent Inteligente")

    st.markdown("""
    <div style="background-color:#215C43; padding: 30px; border-radius: 15px; color: white; text-align: center;">
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
            <p>Simula la expansión del Nitinol con la temperatura.</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""<br>
        <div style="background-color:#e3f2fd ; padding: 20px; border-radius: 12px; text-align: center;">
            <h3>🩸 Velocidad del flujo sanguíneo </h3>
            <p>Calcula la velocidad , la caída de presión y FFR del flujo en el stent .</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div style="background-color:#fff3e0; padding: 20px; border-radius: 12px; text-align: center;">
            <h3>⚙️ Parámetros del Circuito LC</h3>
            <p>Calcula la inductancia, la capacitancia y la frecuencia de resonancia.</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""<br>
        <div style="background-color:#f3e5f5; padding: 20px; border-radius: 12px; text-align: center;">
            <h3>🔍 Análisis del Sistema de Comunicación</h3>
            <p>Indica la eficiencia del circuito resonante LC y la distancia máxima de comunicación</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""<br>
        <div style="background-color:#D7CCC8; padding: 20px; border-radius: 12px; text-align: center;">
            <h3> 🔩 Análisis de Tensiones en el Stent</h3>
            <p> Se calcula las tensiones y se evalúa la seguridad</p>
        </div>
        """, unsafe_allow_html=True)


elif st.session_state.vista_activa == "Vista 3D del Stent":

    st.title("🧊 Vista 3D del Stent")

    uploaded_files = st.file_uploader("Sube uno o varios archivos STL", type=["stl"], accept_multiple_files=True)

    if uploaded_files:
        # Si se suben archivos, mostrar solo esos archivos
        conjunto=set()
        for uploaded_file in uploaded_files:
            # Cargar y procesar cada archivo STL
            if  uploaded_file.name not in conjunto:
                mesh = cargar_y_procesar_stl(uploaded_file)
                mostrar_modelo_stl(uploaded_file.name, mesh)
                conjunto.add(uploaded_file.name)
            else: 
                 st.warning(f"⚠️ El archivo '{uploaded_file.name}' está duplicado por lo que fue ignorado.")
    else:# Subir archivo STL
        # Si no se suben archivos, mostrar los modelos por defecto
        modelos_predeterminados = cargar_modelo_predeterminado()
        for modelo in modelos_predeterminados:
            mesh = trimesh.load(modelo)
            mostrar_modelo_stl(os.path.basename(modelo), mesh)
                    
elif st.session_state.vista_activa == "Expansión térmica":
    st.title("🌡️ Aproximación matemática de la expansión térmica del Nitinol")
    st.markdown("""
<div style='background-color: #f9f9f9; padding: 20px; border-radius: 10px;'>

<p>La expansión térmica del Nitinol se ha modelado utilizando una aproximación matemática sencilla, 
en la que se establece un comportamiento lineal para las fases de <b>martensita</b> (a bajas temperaturas) 
y <b>austenita</b> (a temperaturas más elevadas), con un cambio de fase suave representado por una ecuación tipo coseno.  
Este modelo estima el cambio de tamaño del material en función de la temperatura.</p>

<p>Durante la transición entre las fases, se aplica un coeficiente de expansión térmica:</p>

<ul>
<li>Para la martensita: <b>αₘ = 6.6 × 10⁻⁶ 1/°C</b></li>
<li>Para la austenita: <b>αₐ = 11.0 × 10⁻⁶ 1/°C</b></li>
</ul>

</div>
""", unsafe_allow_html=True)
    
    st.markdown("---")  # Separador visual

    # Interfaz Streamlit
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Parámetros de Entrada")
        tipo = st.selectbox("Tipo de medida", ["diámetro", "longitud"])
        D_inicial = st.number_input("Valor inicial (mm)", value=1.8 if tipo == "diámetro" else 10.4)
        T_min = st.slider("Temperatura mínima (°C)", 0, 37, 18)
        T_max = st.slider("Temperatura máxima (°C)", 37, 80, 50)
        As = st.number_input("As (°C)", value=24.000)
        Af = st.number_input("Af (°C)", value=40.000)

    with col2:
        st.subheader("Resultado de la Expansión")
        T, D = expansion(tipo, T_min, T_max, D_inicial, As, Af, 6.6e-6, 11e-6)

        # Color por tipo
        color_linea = 'blue' if tipo == "diámetro" else 'green'

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(T, D, label=f'{tipo.capitalize()} inicial = {D_inicial} mm', color=color_linea)
        ax.axvline(37, color='red', linestyle='--', label=f'37°C = {D[np.abs(T - 37).argmin()]:.4f} mm')
        ax.axvspan(As, Af, color='gray', alpha=0.2, label='Rango de transición')
        ax.set_title(f'Expansión del stent de Nitinol - {tipo.capitalize()}')
        ax.set_xlabel('Temperatura (°C)')
        ax.set_ylabel(f'{tipo.capitalize()} (mm)')
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)

elif st.session_state.vista_activa == "Velocidad del flujo sanguíneo":
    st.title("🩸 Aproximación de la velocidad del flujo sanguíneo en el stent")
    st.markdown("""
    <div style='background-color: #f9f9f9; padding: 20px; border-radius: 10px;'>
    <p>Este modelo estudia el comportamiento del flujo sanguíneo a través de un stent implantado en una arteria. 
    Para ello, se ha aplicado la ley de Poiseuille, que describe el flujo laminar de un fluido viscoso en un conducto cilíndrico,
    con el fin de calcular parámetros clave como la caída de presión, la velocidad promedio y el perfil de velocidad en función del caudal sanguíneo. 
    Además, se ha determinado el índice de reserva fraccional de flujo (FFR) y el índice de reserva de flujo instantáneo (iFR), dos métricas fundamentales para evaluar la funcionalidad 
    del stent en estenosis.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")  # Separador visual

    # Interfaz Streamlit
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Parámetros de Entrada")
        tipo = st.selectbox("Tipo de medida", ["En reposo", "En actividad física"])
        Q_input = st.text_input("Flujo sanguíneo (m³/s)", value="3.34e-6" if tipo == "En reposo" else "16.67e-6")
        Q = float(Q_input)
        L = st.number_input("Longitud del stent (m)", value=0.020)
        R_stent_input = st.text_input("Radio del stent (m)", value="1.775e-3")
        R_stent = float(R_stent_input)
        mu_input = st.text_input("Viscosidad de la sangre (Pa·s)", value="3.5e-3")
        mu = float(mu_input)
        P_entrada = st.number_input("Presión de la sangre (Pa)", value= 13332)

    with col2:
        st.subheader("Resultado del Flujo Sanguíneo")
        delta_P, v_prom, P_salida, v, r, v_max, ffr = flujo( Q, R_stent, L, mu,P_entrada)
        resultados = st.empty()  # Crear un espacio vacío

        # Actualizar la caja con los resultados
        resultados.markdown(
            f"""
            - **Caída de presión a través del stent:** {delta_P:.2f} Pa
            - **Velocidad promedio del flujo sanguíneo en el stent:** {v_prom:.5f} m/s
            - **Presión en la salida del stent:** {P_salida:.2f} Pa
            - **FFR: {ffr:.2f}. Al ser > 0.8, el stent funcionando correctamente**
            """
        )
        #st.write(f"Caída de presión a través del stent: {delta_P:.2f} Pa")
        #st.write(f"Velocidad promedio del flujo sanguíneo en el stent: {v_prom:.5f} m/s")
        #st.write(f"Presión en la salida del stent: {P_salida:.2f} Pa")

        # Gráfica
        plt.figure(figsize=(10, 6))
        plt.plot(v, r * 1000, label='Perfil de velocidad', color='blue')  # r en mm

        # Marcar velocidad máxima en el centro
        plt.scatter([v_max], [0], color='red', zorder=5, label="Velocidad máxima (centro)")
        plt.text(v_max + 0.005, 0, f"Max: {v_max:.5f} m/s", color='red')

        # Marcar velocidad cero en el borde
        plt.scatter([0], [R_stent * 1000], color='green', zorder=5, label="Velocidad en la pared (borde)")
        plt.text(0.005, R_stent * 1000 + 0.5, "V_borde: 0 m/s", color='green')

        # Configurar el gráfico
        plt.xlabel('Velocidad (m/s)')
        plt.ylabel('Radio (mm)')
        plt.title('Perfil de Velocidad de Poiseuille en el Stent')
        plt.legend()
        plt.grid(True)
        plt.gca().invert_yaxis()  # Invertir eje Y para que el centro esté arriba
        st.pyplot(plt)

    st.subheader("Evaluación de la estenosis en el stent")
    def evaluar_estenosis(Q_reposo, Q_actividad, R_stent_original, L, mu, P_entrada):
        reducciones = [1.0, 0.75, 0.50, 0.3]  # 100%, 75%, 50% del radio original
        colores = ['blue', 'orange', 'red',"purple"]
        etiquetas = ['Sin oculusión ', '25% de oclusión', '50% de oclusión',"75% de oclusión"]
        
        estados = [('Reposo', Q_reposo), ('Actividad', Q_actividad)]
        
        fig, axs = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
        
        for idx, (estado, Q) in enumerate(estados):
            ax = axs[idx]
            for i, factor in enumerate(reducciones):
                R = R_stent_original * factor
                delta_P = (8 * mu * L * Q) / (np.pi * R**4)
                v_prom = Q / (np.pi * R**2)
                P_salida = P_entrada - delta_P
                ffr = max(P_salida / P_entrada, 0)
    
                total_points = 100
                r = np.linspace(0, R, total_points)
                v = (1 / (4 * mu)) * (-delta_P / L) * (R**2 - r**2)
                v = np.abs(v)
    
                ax.plot(v, r * 1000, color=colores[i], label=f"{etiquetas[i]} - FFR: {ffr:.2f}")
    
            ax.set_title(f"{estado}")
            ax.set_xlabel('Velocidad (m/s)')
            ax.grid(True)
            ax.legend()
            ax.invert_yaxis()
    
        axs[0].set_ylabel('Radio (mm)')
        fig.suptitle('Impacto de la Estenosis en el Perfil de Velocidad y FFR o iFR (Reposo vs Actividad)', fontsize=14)
        plt.tight_layout()
        st.pyplot(fig)
        
    Q_rep = 3.34e-6 # Flujo sanguíneo rep (m^3/s)
    Q_act = 16.67e-6# Flujo sanguíneo act (m^3/s)
    
    # Ejecutar la simulación
    evaluar_estenosis(Q_rep,Q_act, R_stent, L, mu, P_entrada)
elif st.session_state.vista_activa == "Parámetros del Circuito LC":

    # Título de la sección
    st.title("⚙️ Simulación de los parámetros del circuito LC")

    # Descripción interactiva
    st.markdown("""
    <div style='background-color: #f9f9f9; padding: 20px; border-radius: 10px;'>
    <p>En esta sección se podrá calcular y explorar de forma interactiva la capacitancia, la inductancia, la frecuencia de resonancia 
               del circuito resonante LC del stent inteligente.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")  # Separador visual

    # Interfaz Streamlit
    with st.expander("🔧 Parámetros de las bobinas y sensor", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 🌀 Cálculo de la Inductancia")
            r_bobina_cm_input = st.text_input("Radio de la bobina (cm)", value="0.06")
            r_bobina_cm = float(r_bobina_cm_input)
            l_bobina_cm_input = st.text_input("Longitud de la bobina (cm)", value="0.25")
            l_bobina_cm = float(l_bobina_cm_input)
            vueltas_bobina = st.number_input("Número de vueltas", value=10)
        with col2:
            st.markdown("### ⚡ Cálculo de la Capacitancia")
            A_electrodo_m2_input = st.text_input("Área de los electrodos (m²)", value="1.05e-7")
            A_electrodo_m2 = float(A_electrodo_m2_input)
            d_poliamida_m_inicial = st.text_input("Grosor de la capa de poliamida (m)", value="5e-6")
            d_poliamida_m = float(d_poliamida_m_inicial)
            num_pares_electrodos = st.number_input("Número de pares de electrodos", value  = 48)
    st.markdown("### 📊 Resultados")

    # Calcular inductancia
    L_bobina = calcular_inductancia(r_bobina_cm, l_bobina_cm, vueltas_bobina)
    # Calcular la inductancia total en paralelo
    L_total = 1 / (1/L_bobina + 1/L_bobina)

    # Calcular capacitancia
    C_total = calcular_capacitancia(A_electrodo_m2, d_poliamida_m, num_pares_electrodos)

    # Calcular frecuencia de resonancia
    f_resonancia = calcular_frecuencia_resonancia(L_total, C_total)

    # Mostrar resultados en un cuadro con puntos
    st.markdown(f"""
        <div style="background-color:#f9f9f9; padding: 15px; border-radius: 5px;">
        <ul style="list-style-type: disc; padding-left: 20px;">
            <li><strong>Inductancia total:</strong> {L_total * 1e6:.2f} µH</li>
            <li><strong>Capacitancia total:</strong> {C_total * 1e12:.2f} pF</li>
            <li><strong>Frecuencia de resonancia:</strong> {f_resonancia / 1e6:.2f} MHz</li>
        </ul>
        </div>
    """, unsafe_allow_html=True)


    st.markdown("<br>**Selecciona un rango de distancia entre electrodos (µm)**", unsafe_allow_html=True)
    
    # Slider para seleccionar rango de distancia
    d_slider = st.slider(
        "",
        min_value=3.0, max_value=10.0, value=(4.0, 8.0), step=0.1,
    )
    
    # Crear el rango de distancia (convertido a metros)
    d_range = np.linspace(d_slider[0], d_slider[1], 100) * 1e-6
    
    # Calcular capacitancia total para cada distancia
    C_total_array = np.array([calcular_capacitancia(A_electrodo_m2, D, num_pares_electrodos) for D in d_range])
    f_resonancia_array = np.array([calcular_frecuencia_resonancia(L_total, C) for C in C_total_array])
    f_resonancia_MHz_array = f_resonancia_array / 1e6  # Convertir a MHz
    
    # Crear subplots interactivos
    fig = make_subplots(
        rows=1, cols=2, subplot_titles=(
            "Frecuencia vs Distancia entre Electrodos",
            "Frecuencia vs Capacitancia Total"
        )
    )
    
    # Gráfico: Frecuencia vs Distancia
    fig.add_trace(go.Scatter(
        x=d_range * 1e6,  # Convertir a µm
        y=f_resonancia_MHz_array,
        mode='lines',
        name='f vs distancia',
        line=dict(color='blue')
    ), row=1, col=1)
    
    # Gráfico: Frecuencia vs Capacitancia
    fig.add_trace(go.Scatter(
        x=C_total_array * 1e12,  # Convertir a pF
        y=f_resonancia_MHz_array,
        mode='lines',
        name='f vs capacitancia',
        line=dict(color='green')
    ), row=1, col=2)
    
    # Ejes
    axis_style = dict(
    linecolor='black',  # Color negro
    linewidth=2,        # Grosor más grueso
    showline=True       # Mostrar la línea del eje
    )
    fig.update_xaxes(title_text="Distancia (µm)", autorange="reversed", row=1, col=1, **axis_style)
    fig.update_yaxes(title_text="Frecuencia (MHz)", row=1, col=1, **axis_style)
    fig.update_xaxes(title_text="Capacitancia (pF)", row=1, col=2, **axis_style)
    fig.update_yaxes(title_text="Frecuencia (MHz)", row=1, col=2, **axis_style)
    
    # Layout general
    fig.update_layout(
        template="plotly_white",
        height=500,
        title_text="Relaciones entre Frecuencia, Distancia y Capacitancia",
        showlegend=False
    )
    
    # Mostrar en Streamlit
    st.plotly_chart(fig)
elif st.session_state.vista_activa == "Análisis del Sistema de Comunicación":
    st.markdown("## 🔍 Análisis del Sistema de Comunicación")

    st.markdown("""
        <div style='background-color: #f9f9f9; padding: 20px; border-radius: 10px;'>
            <p style='font-size:16px;'>
            En esta sección, se presentarán los resultados del <strong>análisis del sistema de comunicación</strong>. Se detallarán los valores del <strong>factor de calidad (Q)</strong> de los circuitos internos y externos, así como la <strong>distancia máxima de comunicación</strong> estimada, acompañada de su interpretación. Estos resultados permitirán evaluar la eficiencia del sistema y su capacidad para transmitir datos de manera efectiva bajo las condiciones establecidas.
            </p>
        </div>
    """, unsafe_allow_html=True)



    tab1, tab2 = st.tabs(["⚙️ Parámetros Internos", "📡 Parámetros Externos"])

    with tab1:
        col1, col2 = st.columns([2, 1])

        with col1:
            tipo = st.selectbox("Material de las bobinas:", ["Oro", "Otro"])
            resistividad = 2.44e-8 if tipo == "Oro" else st.number_input("Resistividad (ohm·m)", value=2.44e-8)

            r_bobina_m = float(st.text_input("Radio de la bobina (m)", value="0.0006"))
            vueltas_bobina = st.number_input("Número de vueltas", value=10)
            d_m = float(st.text_input("Diámetro del hilo conductor (m)", value="1e-5"))
            L = float(st.text_input("Inductancia total calculada (H)", value="0.06e-6"))

        with col2:
            longitud_hilo = 2 * math.pi * r_bobina_m * vueltas_bobina
            area_seccion = math.pi * (d_m / 2)**2
            R = resistividad * longitud_hilo / area_seccion
            R_total = 1 / (1/R + 1/R)

            C_fija = 57.12e-12  # 57.12 pF
            Q = (1 / R_total) * math.sqrt(L / C_fija)

            st.markdown("""
            <div style="background-color:#f1f1f1; padding: 15px; border-radius: 10px;">
                <h4 style="margin-top: 0;">🧾 Resultados:</h4>
                <ul>
                    <li><strong>Inductancia:</strong> {:.2f} µH</li>
                    <li><strong>Capacitancia:</strong> 57.12 pF</li>
                    <li><strong>Resistencia interna:</strong> {:.2f} Ω</li>
                    <li><strong>Q interno:</strong> {:.2f}</li>
                </ul>
            </div>
            """.format(L * 1e6, R_total, Q), unsafe_allow_html=True)

    with tab2:
        col3, col4 = st.columns([2, 1])

        with col3:
            L_externa = float(st.text_input("Inductancia externa (H)", value="4e-6"))
            R_externa = st.number_input("Resistencia externa (Ω)", value=25.0)
            radio_bobina_ext = st.number_input("Radio de la bobina emisora externa (m)", value=0.035)

        with col4:

            C_externa = 1 / ((2 * math.pi * 86.54e6)**2 * L_externa)
            Q_externa = (1 / R_externa) * math.sqrt(L_externa / C_externa)

            st.markdown("""
            <div style="background-color:#f1f1f1; padding: 15px; border-radius: 10px;">
                <h4 style="margin-top: 0;">🧾 Resultados:</h4>
                <ul>
                    <li><strong>Inductancia_ext:</strong> {:.2f} µH</li>
                    <li><strong>Capacitancia_ext:</strong> {:.2f} pF</li>
                    <li><strong>Resistencia_ext:</strong> {:.2f} Ω</li>
                    <li><strong>Q externo:</strong> {:.2f}</li>
                </ul>
            </div>
            """.format(L_externa * 1e6, C_externa * 1e12, R_externa, Q_externa), unsafe_allow_html=True)


    # Cálculo de distancia final y explicación
    k_eff = 0.8
    d_max = radio_bobina_ext * k_eff * math.sqrt(Q_externa * Q)

    # Título
    st.markdown("## 📊 Resultados Globales del Sistema de Comunicación")

    # Tarjeta de resultados con explicación
    st.markdown(f"""
    <div style="background-color:#f1f1f1; padding: 20px; border-radius: 12px;">
        <h4 style="margin-top: 0;">📏 Cálculo de la Distancia Máxima de Comunicación</h4>
        <p style="font-size:16px;">
            A partir de los parámetros internos y externos definidos, se calcula la distancia máxima a la que el sistema puede transmitir información de manera eficiente. 
            Esta distancia depende del <strong>factor de calidad (Q)</strong> de ambos circuitos </strong>.Esta distancia representa el alcance teórico bajo condiciones óptimas 
            de acoplamiento resonante. Valores bajos indican necesidad de optimización en el diseño del circuito interno o externo.
        </p>
        <ul style="font-size:16px;">
            <li><strong>Q interno:</strong> {Q:.2f}</li>
            <li><strong>Q externo:</strong> {Q_externa:.2f}</li>
            <li><strong>Distancia máxima estimada:</strong> <strong>{d_max*100:.2f} cm</strong></li>
        </ul>
        <p style="font-size:16px;">
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Interpretación visual
    if d_max < 0.01:
        interpretacion = "📉 <strong>Distancia muy baja.</strong> El sistema no será eficiente. Revisa el diseño interno o mejora el acoplamiento."
        color = "#d9534f"
    elif d_max < 0.03:
        interpretacion = "⚠️ <strong>Distancia aceptable pero limitada.</strong> Solo funcionará con buena alineación y proximidad."
        color = "#f0ad4e"
    else:
        interpretacion = "✅ <strong>Buena distancia de comunicación.</strong> El sistema es eficiente en un entorno implantable controlado."
        color = "#5cb85c"

    st.markdown(
        f"<div style='background-color:{color}; padding:15px; border-radius:10px; color:white;'>"
        f"<b> Interpretación final:</b> {interpretacion}</div>",
        unsafe_allow_html=True
    )
elif st.session_state.vista_activa == "Análisis mecánico del stent":
    def tension(p, r_i, r_o, limite_elastico, FS, flag=True):
        t = r_o - r_i
        r_m = (r_i + r_o) / 2
        if t / r_i < 0.1:
            print("📘 Uso de la teoría del cilindro delgado.")

            sigma_theta = (p * r_m) / t
            sigma_r = 0  # despreciada
            sigma_vm = sigma_theta
            limite_admisible = limite_elastico / FS

            if sigma_vm < limite_admisible:
                print("✅ El diseño es MECÁNICAMENTE SEGURO (cilindro delgado).")
            else:
                print("❌ PELIGRO: El diseño supera el límite elástico o no cumple con el FS.")

            if flag:
                r_values = np.linspace(r_i, r_o, 200)
                sigma_theta_f = np.full_like(r_values, sigma_theta / 1e6)
                sigma_r_f = np.zeros_like(r_values)
                sigma_vm_f = sigma_theta_f

                r_mm = r_values * 1000
                plt.figure(figsize=(8, 5))
                plt.plot(r_mm, sigma_theta_f, label='Tensión circunferencial (MPa)', color='blue')
                plt.plot(r_mm, sigma_r_f, label='Tensión radial (MPa)', color='red')
                plt.plot(r_mm, sigma_vm_f, '--', label='Tensión von Mises (MPa)', color='green', marker='.')
                plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
                plt.title('Distribución de tensiones (modelo de cilindro delgado)')
                plt.xlabel('Radio desde el interior hacia el exterior (mm)')
                plt.ylabel('Tensión (MPa)')
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.show()

        else:
            print("📘 Uso de la teoría del cilindro de paredes gruesas (Lame).")

            # Constantes de Lame
            A = (p * r_i**2) / (r_o**2 - r_i**2)
            B = (p * r_i**2 * r_o**2) / (r_o**2 - r_i**2)

            # Máximos en el radio interior (punto crítico)
            sigma_r = A - B / r_i**2
            sigma_theta = A + B / r_i**2
            sigma_vm = np.sqrt(0.5 * ((sigma_theta - sigma_r)**2 + sigma_theta**2 + sigma_r**2))
            limite_admisible = limite_elastico / FS

            if sigma_vm < limite_admisible:
                print("✅ El diseño es MECÁNICAMENTE SEGURO (cilindro grueso).")
            else:
                print("❌ PELIGRO: El diseño supera el límite elástico o no cumple con el FS.")

            if flag:
                r_values = np.linspace(r_i, r_o, 200)
                sigma_r_f = A - B / r_values**2
                sigma_theta_f = A + B / r_values**2
                sigma_vm_f = np.sqrt(0.5 * ((sigma_theta_f - sigma_r_f)**2 + sigma_theta_f**2 + sigma_r_f**2))

                # Conversión a MPa
                sigma_r_f /= 1e6
                sigma_theta_f /= 1e6
                sigma_vm_f /= 1e6
                r_mm = r_values * 1000

                plt.figure(figsize=(8, 5))
                plt.plot(r_mm, sigma_theta_f, label='Tensión circunferencial (MPa)', color='blue')
                plt.plot(r_mm, sigma_r_f, label='Tensión radial (MPa)', color='red')
                plt.plot(r_mm, sigma_vm_f, '--', label='Tensión von Mises (MPa)', color='green', marker='.')
                plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
                plt.title('Distribución de tensiones (modelo de cilindro grueso)')
                plt.xlabel('Radio desde el interior hacia el exterior (mm)')
                plt.ylabel('Tensión (MPa)')
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.show()
        return sigma_theta, sigma_r, sigma_vm, limite_admisible

    st.title("🔩 Análisis de Tensiones en el Stent")
    st.markdown("""
    <div style='background-color: #f9f9f9; padding: 20px; border-radius: 10px;'>
    <p>Este modelo estudia las tensiones internas en un stent bajo presión interna, utilizando la teoría de tensiones para un cilindro delgado o grueso. 
    Calcula la tensión circunferencial, la tensión radial y la tensión de von Mises en función del radio interno y externo del stent, 
    el límite elástico del material y el factor de seguridad. Además, se muestra la distribución de tensiones a lo largo del espesor del stent.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")  # Separador visual

    # Interfaz Streamlit
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Parámetros de Entrada")
        tipo = st.selectbox("Tipo de medida", ["En reposo", "En actividad física"])
        p_input = st.number_input("Presión interna (Pa)", value=13032.64 if tipo == "En reposo" else 13272.02)
        r_i_input = float(st.text_input("Radio interno (m)", value=1.67e-3))
        r_o_input = float(st.text_input("Radio externo (m)", value=1.78e-3))
        limite_elastico_input = st.number_input("Límite elástico (MPa)", value=500)
        FS_input = st.number_input("Factor de seguridad (FS)", value=2)

    with col2:
        st.subheader("Resultado del Análisis Mecánico")
        # Cálculo de tensiones
        p = p_input
        r_i = r_i_input
        r_o = r_o_input
        thickness = r_o - r_i
        limite_elastico = limite_elastico_input*1e6
        FS = FS_input
        
        sigma_theta, sigma_r, sigma_vm, limite_admisible = tension(p, r_i, r_o, limite_elastico, FS, False)

        if 0.1>thickness/r_i:
            st.write(f"**- Uso de la teoria de los cilindros delgados**")
        else:
            st.write(f"**- Uso de la teoria de los cilindros gruesos**")
        # Mostrar resultados
        st.write(f"**- Tensión circunferencial:** {sigma_theta / 1e6:.3f} MPa")
        st.write(f"**- Tensión radial:** {sigma_r / 1e6:.3f} MPa")
        st.write(f"**- Tensión de von Mises:** {sigma_vm / 1e6:.3f} MPa")
        st.write(f"**- Límite máximo admisible:** {limite_admisible / 1e6:.0f} MPa")

        if 0.1>thickness/r_i:
            r_values = np.linspace(r_i, r_o, 400)
            sigma_theta_f = np.full_like(r_values, sigma_theta / 1e6)  # MPa
            sigma_vm_f = sigma_theta_f  # es igual
            sigma_r_f = np.zeros_like(r_values)
        else:
            # Gráfica de la distribución de tensiones
            r_values = np.linspace(r_i, r_o, 200)
            sigma_theta_f = (p * r_values**2) / (r_o**2 - r_i**2)
            sigma_r_f = -(p * r_values**2) / (r_o**2 - r_i**2)
            sigma_vm_f = np.sqrt(0.5 * ((sigma_theta_f - sigma_r_f)**2 + sigma_theta_f**2 + sigma_r_f**2))
            
        # Conversión a MPa y mm para graficar
        sigma_theta_f /= 1e6
        sigma_r_f /= 1e6
        sigma_vm_f /= 1e6
        r_mm = r_values * 1000  # mm
        
        # Gráfica
        plt.figure(figsize=(8, 5))
        plt.plot(r_mm, sigma_theta_f, label='Tensión circunferencial (MPa)', color='blue')
        plt.plot(r_mm, sigma_r_f, label='Tensión radial (MPa)', color='red')
        plt.plot(r_mm, sigma_vm_f, linestyle='--', color='green', label='Tensión von Mises (MPa)')
        plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
        
        plt.title('Distribución de tensiones a través del espesor del stent')
        plt.xlabel('Radio desde el interior hacia el exterior (mm)')
        plt.ylabel('Tensión (MPa)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        st.pyplot(plt)
        
     # Interpretación visual
    if sigma_vm < limite_elastico and sigma_vm < limite_admisible:
        interpretacion = " ✅ <strong> El diseño es MECÁNICAMENTE SEGURO, cumple con el límite elástico y el factor de seguridad."
        color = "#5cb85c"
    else:
        interpretacion = "❌ <strong> PELIGRO: El diseño supera el límite elástico o no cumple con el factor de seguridad."
        color = "#d9534f"

    st.markdown(
        f"<div style='background-color:{color}; padding:15px; border-radius:10px; color:white;'>"
        f"<b>Interpretación final:</b> {interpretacion}</div>",
        unsafe_allow_html=True
    )
