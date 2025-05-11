

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
    uploaded_files = st.file_uploader("Sube uno o varios archivos STL", type=["stl"], accept_multiple_files=True)
    
    if uploaded_files:
        # Si se suben archivos, mostrar solo esos archivos
        for uploaded_file in uploaded_files:
            # Cargar y procesar cada archivo STL
            mesh = cargar_y_procesar_stl(uploaded_file)
            mostrar_modelo_stl(uploaded_file.name, mesh)
    else:
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

# Crear gráfico interactivo con Plotly
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=d_range * 1e6,  # Pasar de metros a micrómetros para el eje X
    y=f_resonancia_MHz_array,
    mode='lines',
    name='Frecuencia vs Distancia',
    line=dict(color='royalblue', width=2)
))

# Marcar el punto correspondiente a una distancia y capacitancia concreta (opcional)
# Aquí podrías usar valores como d_real y C_real si los tienes
# Ejemplo:
# d_real = 5e-6
# C_real = 2 * num_pares_electrodos * epsilon_0 * epsilon_r_poliamida * A_electrodo_m2 / d_real
# f_real = calcular_frecuencia_resonancia(L_total, C_real) / 1e6
# fig.add_trace(go.Scatter(...))

# Configurar el diseño del gráfico
fig.update_layout(
    title='Relación entre Distancia entre Electrodos y Frecuencia de Resonancia',
    xaxis_title='Distancia entre electrodos (µm)',
    yaxis_title='Frecuencia de Resonancia (MHz)',
    template='plotly_white',
    font=dict(family="Arial", size=14),
    xaxis=dict(
        showline=True,
        showgrid=True,
        zeroline=True
    ),
    yaxis=dict(
        showline=True,
        showgrid=True,
        zeroline=True
    )
)

st.plotly_chart(fig)

