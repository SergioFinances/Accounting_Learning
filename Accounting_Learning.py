import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import openai
import random
from fpdf import FPDF
import io
import os
from dotenv import load_dotenv

load_dotenv()

# Configuración segura de OpenRouter
api_key = os.getenv("OPENROUTER_API_KEY")
openai.api_key = api_key
openai.api_base = "https://openrouter.ai/api/v1"

# Función para dar formato español de números
def fmt(v):
    s = f"{v:,.1f}".replace(',', 'X').replace('.', ',').replace('X', '.')
    return s

# Callbacks para navegación de slides
def inv_go_prev():
    if st.session_state.inv_slide_index > 0:
        st.session_state.inv_slide_index -= 1

def inv_go_next(slides):
    if st.session_state.inv_slide_index < len(slides) - 1:
        st.session_state.inv_slide_index += 1

def dep_go_prev():
    if st.session_state.slide_index > 0:
        st.session_state.slide_index -= 1

def dep_go_next(total):
    if st.session_state.slide_index < total - 1:
        st.session_state.slide_index += 1

# Callback para Portada "Entrar"
def enter_app():
    st.session_state.show_portada = False

# Callback para Login "Ingresar"
def do_login():
    user = st.session_state.login_raw_user.strip().lower()
    pwd = st.session_state.login_password
    if not user or not pwd:
        st.session_state.login_error = "Por favor, ingresa usuario y contraseña."
    elif user in st.session_state.users and st.session_state.users[user]["password"] == pwd:
        st.session_state.authenticated = True
        st.session_state.username = user
        st.session_state.show_portada = True
        st.session_state.login_error = ""
    else:
        st.session_state.login_error = "Credenciales incorrectas."

# HTML de tabla combinada para simulación Promedio Ponderado
def get_table_html():
    return f"""
    <table style="width:100%; border-collapse: collapse;">
      <thead>
        <tr>
          <th rowspan="2" style="border:1px solid #ddd; padding:8px; text-align:center;">Detalle<br>Concepto</th>
          <th colspan="3" style="border:1px solid #ddd; padding:8px; text-align:center;">Compras</th>
          <th colspan="3" style="border:1px solid #ddd; padding:8px; text-align:center;">Ventas</th>
          <th colspan="3" style="border:1px solid #ddd; padding:8px; text-align:center;">Saldo</th>
        </tr>
        <tr>
          <th style="border:1px solid #ddd; padding:8px; text-align:center;">Cantidad</th>
          <th style="border:1px solid #ddd; padding:8px; text-align:center;">Valor unitario</th>
          <th style="border:1px solid #ddd; padding:8px; text-align:center;">Total</th>
          <th style="border:1px solid #ddd; padding:8px; text-align:center;">Cantidad</th>
          <th style="border:1px solid #ddd; padding:8px; text-align:center;">Valor unitario</th>
          <th style="border:1px solid #ddd; padding:8px; text-align:center;">Total</th>
          <th style="border:1px solid #ddd; padding:8px; text-align:center;">Cantidad</th>
          <th style="border:1px solid #ddd; padding:8px; text-align:center;">Valor unitario</th>
          <th style="border:1px solid #ddd; padding:8px; text-align:center;">Total</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td style="border:1px solid #ddd; padding:8px; text-align:center;">Inventario inicial</td>
          <td colspan="3" style="border:1px solid #ddd; padding:8px; text-align:center;"></td>
          <td colspan="3" style="border:1px solid #ddd; padding:8px; text-align:center;"></td>
          <td style="border:1px solid #ddd; padding:8px; text-align:center;">{fmt(100)}</td>
          <td style="border:1px solid #ddd; padding:8px; text-align:center;">{fmt(15)}</td>
          <td style="border:1px solid #ddd; padding:8px; text-align:center;">{fmt(1500)}</td>
        </tr>
        <tr>
          <td style="border:1px solid #ddd; padding:8px; text-align:center;">Compra</td>
          <td style="border:1px solid #ddd; padding:8px; text-align:center;">{fmt(150)}</td>
          <td style="border:1px solid #ddd; padding:8px; text-align:center;">{fmt(18)}</td>
          <td style="border:1px solid #ddd; padding:8px; text-align:center;">{fmt(2700)}</td>
          <td colspan="3" style="border:1px solid #ddd; padding:8px; text-align:center;"></td>
          <td style="border:1px solid #ddd; padding:8px; text-align:center;">{fmt(250)}</td>
          <td style="border:1px solid #ddd; padding:8px; text-align:center;">{fmt(16.8)}</td>
          <td style="border:1px solid #ddd; padding:8px; text-align:center;">{fmt(4200)}</td>
        </tr>
        <tr>
          <td style="border:1px solid #ddd; padding:8px; text-align:center;">Venta</td>
          <td colspan="3" style="border:1px solid #ddd; padding:8px; text-align:center;"></td>
          <td style="border:1px solid #ddd; padding:8px; text-align:center;">{fmt(150)}</td>
          <td style="border:1px solid #ddd; padding:8px; text-align:center;">{fmt(16.8)}</td>
          <td style="border:1px solid #ddd; padding:8px; text-align:center;">{fmt(2520)}</td>
          <td style="border:1px solid #ddd; padding:8px; text-align:center;">{fmt(100)}</td>
          <td style="border:1px solid #ddd; padding:8px; text-align:center;">{fmt(16.8)}</td>
          <td style="border:1px solid #ddd; padding:8px; text-align:center;">{fmt(1680)}</td>
        </tr>
      </tbody>
    </table>
    """

# HTML de tabla combinada para simulación PEPS personalizada
def get_peps_table_html():
    return f"""
    <table style="width:100%; border-collapse: collapse;">
      <thead>
        <tr>
          <th rowspan="2" style="border:1px solid #ddd; padding:8px; text-align:center;">Detalle<br>Concepto</th>
          <th colspan="3" style="border:1px solid #ddd; padding:8px; text-align:center;">Compras</th>
          <th colspan="3" style="border:1px solid #ddd; padding:8px; text-align:center;">Ventas</th>
          <th colspan="3" style="border:1px solid #ddd; padding:8px; text-align:center;">Saldo</th>
        </tr>
        <tr>
          <th style="border:1px solid #ddd; padding:8px; text-align:center;">Cantidad</th>
          <th style="border:1px solid #ddd; padding:8px; text-align:center;">Valor unitario</th>
          <th style="border:1px solid #ddd; padding:8px; text-align:center;">Total</th>
          <th style="border:1px solid #ddd; padding:8px; text-align:center;">Cantidad</th>
          <th style="border:1px solid #ddd; padding:8px; text-align:center;">Valor unitario</th>
          <th style="border:1px solid #ddd; padding:8px; text-align:center;">Total</th>
          <th style="border:1px solid #ddd; padding:8px; text-align:center;">Cantidad</th>
          <th style="border:1px solid #ddd; padding:8px; text-align:center;">Valor unitario</th>
          <th style="border:1px solid #ddd; padding:8px; text-align:center;">Total</th>
        </tr>
      </thead>
      <tbody>
        <!-- Inventario inicial -->
        <tr>
          <td style="border:1px solid #ddd; padding:8px; text-align:center;">Inventario inicial</td>
          <td colspan="3" style="border:1px solid #ddd; padding:8px; text-align:center;"></td>
          <td colspan="3" style="border:1px solid #ddd; padding:8px; text-align:center;"></td>
          <td style="border:1px solid #ddd; padding:8px; text-align:center;">{fmt(100)}</td>
          <td style="border:1px solid #ddd; padding:8px; text-align:center;">{fmt(15)}</td>
          <td style="border:1px solid #ddd; padding:8px; text-align:center;">{fmt(1500)}</td>
        </tr>
        <!-- Compra -->
        <tr>
          <td style="border:1px solid #ddd; padding:8px; text-align:center;">Compra</td>
          <td style="border:1px solid #ddd; padding:8px; text-align:center;">{fmt(150)}</td>
          <td style="border:1px solid #ddd; padding:8px; text-align:center;">{fmt(18)}</td>
          <td style="border:1px solid #ddd; padding:8px; text-align:center;">{fmt(2700)}</td>
          <td colspan="3" style="border:1px solid #ddd; padding:8px; text-align:center;"></td>
          <td style="border:1px solid #ddd; padding:8px; text-align:center;">{fmt(150)}</td>
          <td style="border:1px solid #ddd; padding:8px; text-align:center;">{fmt(18)}</td>
          <td style="border:1px solid #ddd; padding:8px; text-align:center;">{fmt(2700)}</td>
        </tr>
        <!-- Venta con celdas combinadas -->
        <tr>
          <td rowspan="2" style="border:1px solid #ddd; padding:8px; text-align:center;">Venta</td>
          <td colspan="3" rowspan="2" style="border:1px solid #ddd; padding:8px; text-align:center;"></td>
          <td rowspan="2" style="border:1px solid #ddd; padding:8px; text-align:center;">{fmt(50)}</td>
          <td rowspan="2" style="border:1px solid #ddd; padding:8px; text-align:center;">{fmt(15)}</td>
          <td rowspan="2" style="border:1px solid #ddd; padding:8px; text-align:center;">{fmt(750)}</td>
          <td style="border:1px solid #ddd; padding:8px; text-align:center;">{fmt(50)}</td>
          <td style="border:1px solid #ddd; padding:8px; text-align:center;">{fmt(15)}</td>
          <td style="border:1px solid #ddd; padding:8px; text-align:center;">{fmt(750)}</td>
        </tr>
        <tr>
          <td style="border:1px solid #ddd; padding:8px; text-align:center;">{fmt(150)}</td>
          <td style="border:1px solid #ddd; padding:8px; text-align:center;">{fmt(18)}</td>
          <td style="border:1px solid #ddd; padding:8px; text-align:center;">{fmt(2700)}</td>
        </tr>
      </tbody>
    </table>
    """


# Inicialización de estado de sesión
def init_session():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "show_portada" not in st.session_state:
        st.session_state.show_portada = False
    if "users" not in st.session_state:
        st.session_state.users = {
            "admin": {"password": "admin123", "role": "admin"},
            "user":  {"password": "user123",  "role": "user"}
        }
    # AQUI: inicializar login_error para evitar KeyError
    if "login_error" not in st.session_state:
        st.session_state.login_error = ""

def check_credentials(user, password):
    return user in st.session_state.users and st.session_state.users[user]["password"] == password

def go_prev():
    if st.session_state.inv_slide_index > 0:
        st.session_state.inv_slide_index -= 1

def go_next(slides):
    if st.session_state.inv_slide_index < len(slides) - 1:
        st.session_state.inv_slide_index += 1

# Login y portada
def login():
    st.title("Iniciar Sesión")
    with st.form("login_form"):
        raw_user = st.text_input("Usuario")
        password = st.text_input("Contraseña", type="password")
        if st.form_submit_button("Ingresar"):
            user = raw_user.strip().lower()
            if not user or not password:
                st.error("Por favor, ingresa usuario y contraseña.")
            elif check_credentials(user, password):
                st.session_state.authenticated = True
                st.session_state.username = user
                st.session_state.show_portada = True
            else:
                st.error("Credenciales incorrectas.")

def logout():
    st.session_state.authenticated = False
    st.session_state.username = ""
    st.session_state.show_portada = False

def mostrar_portada():
    st.image("https://i.ibb.co/MDwk0bmw/Gemini-Generated-Image-kdwslvkdwslvkdws.png", use_container_width=True)
    if st.button("Entrar", key="btn_enter_portada"):
        st.session_state.show_portada = False

# Sección Teoría: Valoración de Inventarios
def mostrar_valoracion_inventarios():
    st.header("Valoración de Inventarios")
    slides = [
        {"video": "https://youtu.be/usCCpByy_Tk"},
        {"content": "**¿Qué es un sistema de inventarios?**\n\n Un **sistema de inventarios** es el conjunto de procedimientos y registros que una empresa utiliza para controlar la cantidad, el costo y el movimiento de sus bienes (mercancías, materias primas, productos terminados).\n\n**Objetivos:**\n- Mantener niveles óptimos (ni exceso ni desabastecimiento).\n- Valorar correctamente el costo de ventas y el inventario final.\n- Facilitar la toma de decisiones de compra y producción.\n\n**Importancia:** \n- Impacta directamente en el ejercicio del reconocimiento de activos y en la elaboración de estados financieros fiables. \n- Permite cumplir con Normas Internacionales de Información Financiera (NIIF) y requisitos tributarios."},
        {"content": "**Inventario Periódico**\n\n En el sistema periódico, el conteo físico de inventario se realiza al final de un periodo contable (mensual, trimestral o anual) y los movimientos de compras y ventas NO se registran inmediatamente en la cuenta de “Inventarios”.\n\n**Ventajas:**\n- Simplicidad en pequeñas empresas. \n- Menor control administrativo diario. \n\n**Desventajas:**\n- Menor precisión en el costo de ventas durante el periodo.\n- Dificultad para detectar robos o pérdidas a tiempo.\n\n**Juego de Inventarios Periódico:**\n\n El juego de inventarios es una simulación interactiva donde registras compras, ventas y conteos para calcular el costo de ventas y el valor final de tu inventario de forma práctica y lúdica. \n\n **Fórmula:** \n\n $$Costo~de~Ventas = Inventario~Inicial + Compras - Devoluciones - Inventario~Final$$"},
        {"simulate_periodic": True},
        {"content": "**¿Qué es el método de promedio ponderado?**\n\nEl **método de promedio ponderado** es una técnica de valoración de inventarios que asigna un costo uniforme a cada unidad, calculando un promedio del valor total de las existencias dividido por la cantidad total.\n\n**Pasos:**\n- Suma las unidades y valores de cada lote (inventario inicial + compras).\n- Divide el valor total entre la cantidad total para obtener el costo unitario promedio.\n- Valora las salidas (ventas o consumos) con ese costo promedio hasta la siguiente compra.\n- Recalcula el promedio cuando ingresa un nuevo lote.\n\n**Ejemplo práctico:**\n- Inventario inicial: 100 unidades con un valor unitario de \\$10 (valor total \\$1.000).\n- Compra: 50 unidades con un valor unitario de \\$12 (valor total \\$600).\n- Costo promedio = (1.000 + 600) / (100 + 50) = $10,67.\n- Cada unidad vendida se valora a $10,67 hasta la próxima compra.\n\n**Importancia:**\n- Suaviza variaciones de precios entre lotes.\n- Facilita la contabilidad continua y el registro automático.\n- Reduce la necesidad de seguimiento detallado de cada lote."},
        {"sim_weighted": True},
        {"content": "**¿Qué es el método PEPS (FIFO)?**\n\nEl **método PEPS** (Primeras en Entrar, Primeras en salir) asume que las unidades más antiguas del inventario son las primeras en venderse o consumirse.\n\n**Pasos:**\n- Organiza los lotes de inventario en orden cronológico (de más antiguo a más reciente).\n- Al registrar una salida (venta o consumo), asigna el costo de las unidades más antiguas disponibles.\n- Resta esas unidades de los lotes iniciales hasta cubrir la cantidad vendida.\n- El inventario final estará compuesto por los lotes más recientes.\n\n**Ejemplo práctico:**\n- Inventario inicial: 100 unidades con un valor unitario de \\$10 (valor total \\$1.000).\n- Compra: 50 unidades con un valor unitario de \\$12 (valor total \\$600).\n- Venta de 120 unidades:\n  - 100 unidades salen del lote inicial a \\$10 → Costo de ventas = \\$1.000. \n  - 20 unidades salen del lote 2 a \\$12 → Costo de ventas = \\$240. \n  - Costo de venta total: \\$1.240 \n- Inventario final:\n  - 30 unidades a \\$12 = \\$360.\n  - Total inventario = \\$360.\n\n**Importancia:**\n- Refleja el flujo físico de productos perecederos o con rotación constante.\n- Bajo inflación, tiende a mostrar un costo de ventas menor y utilidades mayores.\n- Cumple con normas NIIF y facilita la trazabilidad de lotes antiguos."},
        {"sim_peps": True}
    ]
    if "inv_slide_index" not in st.session_state:
        st.session_state.inv_slide_index = 0
    slide = slides[st.session_state.inv_slide_index]

    if "video" in slide:
        st.video(slide["video"])
    elif "content" in slide:
        st.markdown(slide["content"], unsafe_allow_html=True)
    elif slide.get("simulate_periodic"):
        st.subheader("Simulación de inventario periódico")

        # Función para generar valores aleatorios lógicos
        def randomize():
            inv0 = random.randint(1000, 50000)
            compras = random.randint(100, 50000)
            devoluciones = random.randint(0, compras)
            max_invf = inv0 + compras - devoluciones
            invf = random.randint(0, max_invf)
            st.session_state.inv0 = inv0
            st.session_state.compras = compras
            st.session_state.devoluciones = devoluciones
            st.session_state.invf = invf

        if st.button("Valores aleatorios"):
            randomize()

        inv0 = st.number_input(
            "Inventario inicial", min_value=0,
            value=st.session_state.get("inv0", 100),
            key="inv0"
        )
        compras = st.number_input(
            "Compras", min_value=0,
            value=st.session_state.get("compras", 50),
            key="compras"
        )
        devoluciones = st.number_input(
            "Devoluciones", min_value=0, max_value=compras,
            value=st.session_state.get("devoluciones", 0),
            key="devoluciones"
        )
        invf = st.number_input(
            "Inventario final", min_value=0,
            max_value=inv0 + compras - devoluciones,
            value=st.session_state.get("invf", 60),
            key="invf"
        )

        # Cálculo correcto
        correct_cost = inv0 + compras - devoluciones - invf

        # Entrada de la respuesta del estudiante
        user_cost = st.number_input(
            "Ingresa tu resultado de Costo de Ventas",
            value=0.0,
            key="user_cost"
        )
        if st.button("Validar Respuesta"):
            if np.isclose(user_cost, correct_cost):
                st.success(f"¡Correcto! El costo de ventas es {correct_cost}")
            else:
                st.error(f"Incorrecto. No es el valor esperado")
                st.info("Recuerda la fórmula: Inventario inicial + Compras - Devoluciones - Inventario final")


    elif slide.get("sim_weighted"):
        st.subheader("Ejemplo Promedio Ponderado")
        st.markdown(get_table_html(), unsafe_allow_html=True)
        st.markdown("""
**Transacciones:**
1. Compra a la empresa Proveedora S.A.: 150 u. a 18,0 c/u → 2 700,0  
2. Venta a la empresa Cliente S.A.: 150 u. a costo promedio 16,8 c/u → 2 520,0  
3. Inventario final remanente: 100 u. a 16,8 c/u → 1 680,0  

**Cómo se llegó a esos valores:**
- Se parte de un inventario inicial de 100 u. a 15,0 c/u (1 500,0).  
- Tras comprar 150 u. a 18,0 c/u, el valor total es 2 700,0.  
- Costo promedio = (1 500,0 + 2 700,0) / (100 + 150) = 16,8 c/u.  
- Al vender 150 u. se aplicó ese promedio, y el saldo final de 100 u. se valora igual.
        """)
    elif slide.get("sim_peps"):
        st.subheader("Ejemplo PEPS (FIFO)")
        st.markdown(get_peps_table_html(), unsafe_allow_html=True)
        st.markdown("""
**Transacciones:**
1. Compra a la empresa Proveedora S.A.: 150 unidades a un valor de 18,0 cada unidad → 2 700,0  
2. Venta a la empresa Cliente S.A.: 50 unidades 

**Cómo se llegó a esos valores:**
- Inventario inicial: 100 unidades a un valor de 15,0 cada unidad (1 500,0).
- Compra: se regitra las cantidades y el valor unitario y el saldo sería la multiplicación de estos dos valores.
- Con PEPS, primero salen las 50 unidades del inventario inicial (las más antiguas con valor de 15,0 cada uno). 
- Al final, se observan dos valores en el saldo, 50 que quedaron del lote inicial y los 150 de la compra.
        """)

    # Navegación con un solo clic
    col1, col2 = st.columns(2)
    with col1:
        st.button("Anterior", on_click=inv_go_prev, key="inv_prev_btn")
    with col2:
        if not slide.get("sim_peps"):
            st.button("Siguiente", on_click=inv_go_next, args=(slides,), key="inv_next_btn")

# Sección Depreciaciones
def mostrar_depreciaciones_teoria():
    st.header("Depreciaciones")

    # Definir las secciones
    slides = [
        "Video introductorio",
        "Definición de Depreciación",
        "Método Línea Recta",
        "Ejercicio Línea Recta",
        "Método Saldo Decreciente",
        "Ejercicio Saldo Decreciente",
        "Método Suma de Dígitos de los Años",
        "Ejercicio SYD",
        "Método Unidades de Producción",
        "Ejercicio Unidades de Producción"
    ]

    total = len(slides)

    if "slide_index" not in st.session_state:
        st.session_state.slide_index = 0
    idx = st.session_state.slide_index
    slide = slides[idx]

    # Contenido de cada slide
    if slide == "Video introductorio":
        st.video("https://www.youtube.com/watch?v=LO-Zil21tKw")
    elif slide == "Definición de Depreciación":
        st.markdown(
            "**¿Qué es la Depreciación?**\n\nLa **depreciación** es la pérdida de valor que sufre un activo fijo con el paso del tiempo, debido al uso, el desgaste, el envejecimiento o la obsolescencia.\n\n**Importancia contable:**\n- Permite distribuir el costo de un activo a lo largo de su vida útil.\n- Refleja el valor real de los activos en los estados financieros.\n- Afecta el cálculo de las utilidades y los impuestos.\n\n**Ejemplos de activos que se deprecian:**\n- Vehículos\n- Maquinaria\n- Equipos de oficina\n- Muebles y enseres\n\n**Dato clave:**\n\n📌 El **terreno no se deprecia**, ya que no pierde valor con el tiempo en condiciones normales.\n\n**Juego de Depreciación - Concepto Básico:**\n\nExplora ejemplos donde decidirás si un bien se deprecia o no, y cuál sería su vida útil aproximada. Esto te prepara para aplicar los métodos de cálculo más adelante."
        )
    elif slide == "Método Línea Recta":
        st.subheader("Método Línea Recta")
        st.write("**Método de Línea Recta**\n\nEste método distribuye el valor depreciable del activo de manera uniforme durante su vida útil. Es el más utilizado por su simplicidad.\n\n**Ventajas:**\n- Fácil de aplicar.\n- Ideal para activos con uso constante.\n\n**Desventajas:**\n- No refleja variaciones en el uso o productividad del activo.\n\n**Juego de Depreciación - Línea Recta:**\n\nEn esta simulación, registrarás el costo del activo, su vida útil y el valor residual para calcular la depreciación anual constante.\n\n**Fórmula:**\n\n$$Depreciación~Anual = \\frac{Costo~del~Activo - Valor~Residual}{Vida~Útil~(años)}$$")
    elif slide == "Ejercicio Línea Recta":
        st.subheader("Ejercicio: Línea Recta")
        def rnd_linea():
            st.session_state.costo = random.randint(10000, 50000)
            st.session_state.salvamento = random.randint(1000, int(st.session_state.costo * 0.3))
            st.session_state.vida = random.randint(3, 10)
        if st.button("Generar valores aleatorios"): rnd_linea()
        costo = st.number_input("Costo", value=st.session_state.get("costo", 20000))
        salv = st.number_input("Valor de Salvamento", value=st.session_state.get("salvamento", 2000))
        vida = st.number_input("Vida útil (años)", min_value=1, value=st.session_state.get("vida", 5), step=1)
        user = st.number_input("Tu Depreciación Anual", value=0.0, format="%.2f")
        if st.button("Validar Respuesta Línea Recta"):
            correcto = (costo - salv) / vida
            if np.isclose(user, correcto):
                st.success(f"¡Correcto! La depreciación anual es {correcto:.2f}")
            else:
                st.error("Incorrecto.")
                st.info("Recuerda: (Costo - Salvamento) / Vida útil")
    elif slide == "Método Saldo Decreciente":
        st.subheader("Método Saldo Decreciente Doble")
        st.write("**Método de Saldo Decreciente Doble**\n\nEste método acelera la depreciación en los primeros años de vida útil del activo. Se calcula aplicando el doble de la tasa de línea recta sobre el valor en libros del activo al inicio de cada año.\n\n**Ventajas:**\n- Refleja mejor la pérdida de valor de activos que se usan más al principio.\n\n**Desventajas:**\n- Más complejo que el método de línea recta.\n- El valor residual no se garantiza al final del período.\n\n**Juego de Depreciación - Saldo Decreciente Doble:**\n\nSimularás varios años de vida útil calculando la depreciación acelerada año a año.\n\n**Fórmula:**\n\n$$Depreciación~Anual = 2 \\times \\frac{1}{Vida~Útil} \\times Valor~en~Libros~al~Inicio~del~Año$$")
    elif slide == "Ejercicio Saldo Decreciente":
        st.subheader("Ejercicio: Saldo Decreciente")
        def rnd_dd():
            st.session_state.costo_dd = random.randint(10000, 50000)
            st.session_state.vida_dd = random.randint(3, 10)
        if st.button("Generar valores aleatorios Doble Decline"): rnd_dd()
        costo_dd = st.number_input("Costo", value=st.session_state.get("costo_dd", 25000), key="costo_dd")
        vida_dd = st.number_input("Vida útil (años)", min_value=1, value=st.session_state.get("vida_dd", 5), step=1, key="vida_dd")
        user_dd = st.number_input("Tu Depreciación Primer Año", value=0.0, format="%.2f", key="user_dd")
        if st.button("Validar Respuesta Saldo Decreciente"):
            tasa = 2 / vida_dd
            correcto = tasa * costo_dd
            if np.isclose(user_dd, correcto):
                st.success(f"¡Correcto! Depreciación primer año = {correcto:.2f}")
            else:
                st.error("Incorrecto.")
                st.info("Recuerda: 2 / Vida útil * Costo inicial")
    elif slide == "Método Suma de Dígitos de los Años":
        st.subheader("Suma de Dígitos de los Años (SYD)")
        st.write("**Método Suma de Dígitos de los Años (SYD)**\n\nEste método también proporciona una depreciación acelerada, asignando una fracción del valor depreciable según los años restantes de vida útil.\n\n**Ventajas:**\n- Refleja una mayor depreciación en los primeros años.\n- Mejora la relación gasto/beneficio en activos que se desgastan más rápido al inicio.\n\n**Desventajas:**\n- Cálculo más complejo.\n\n**Juego de Depreciación - SYD:**\n\nVas a usar la suma de los dígitos de la vida útil para distribuir la depreciación cada año, con base en los años restantes.\n\n**Fórmulas:**\n\n- Suma de dígitos = $$1 + 2 + ... + n = \\frac{n(n+1)}{2}$$\n- Depreciación del año t = $$\\frac{Años~Restantes}{Suma~de~los~Años} \\times (Costo - Valor~Residual)$$")
    elif slide == "Ejercicio SYD":
        st.subheader("Ejercicio: SYD")
        def rnd_syd():
            st.session_state.costo_syd = random.randint(10000, 50000)
            st.session_state.salv_syd = random.randint(1000, int(st.session_state.costo_syd * 0.3))
            st.session_state.vida_syd = random.randint(3, 10)
            st.session_state.periodo_syd = random.randint(1, st.session_state.vida_syd)
        if st.button("Generar valores aleatorios SYD"): rnd_syd()
        costo_syd = st.number_input("Costo", value=st.session_state.get("costo_syd", 20000), key="costo_syd")
        salv_syd = st.number_input("Salvamento", value=st.session_state.get("salv_syd", 2000), key="salv_syd")
        vida_syd = st.number_input("Vida útil (años)", min_value=1, value=st.session_state.get("vida_syd", 5), step=1, key="vida_syd")
        periodo_syd = st.number_input("¿Para qué año? (t)", min_value=1, max_value=vida_syd, value=st.session_state.get("periodo_syd", 1), step=1, key="periodo_syd")
        user_syd = st.number_input("Tu Depreciación SYD", value=0.0, format="%.2f", key="user_syd")
        if st.button("Validar Respuesta SYD"):
            n = vida_syd
            sum_dig = n * (n + 1) / 2
            correcto = (costo_syd - salv_syd) * (n - periodo_syd + 1) / sum_dig
            if np.isclose(user_syd, correcto):
                st.success(f"¡Correcto! Depreciación año {periodo_syd} = {correcto:.2f}")
            else:
                st.error("Incorrecto.")
                st.info("Recuerda: (Costo - Salvamento) * (vida - t + 1) / [n(n+1)/2]")
    elif slide == "Método Unidades de Producción":
        st.subheader("Método Unidades de Producción")
        st.write("**Método de Unidades de Producción**\n\nEste método calcula la depreciación según el uso real del activo, medido en unidades producidas, horas trabajadas u otro indicador.\n\n**Ventajas:**\n- Muy preciso para activos cuya vida depende del uso.\n\n**Desventajas:**\n- Requiere llevar un registro detallado del uso.\n\n**Juego de Depreciación - Unidades de Producción:**\n\nSimularás el uso del activo en cada periodo y calcularás la depreciación basada en la producción real registrada.\n\n**Fórmulas:**\n\n- Depreciación por unidad: \n$$\\frac{Costo - Valor~Residual}{Total~Unidades~Estimadas}$$\n\n- Depreciación del período:\n$$Depreciación~=~Unidades~Producidas~en~el~Periodo \\times Depreciación~por~Unidad$$")
    elif slide == "Ejercicio Unidades de Producción":
        st.subheader("Ejercicio: Unidades de Producción")
        def rnd_up():
            st.session_state.costo_up = random.randint(10000, 50000)
            st.session_state.salv_up = random.randint(1000, int(st.session_state.costo_up * 0.3))
            st.session_state.unid_up = random.randint(1000, 10000)
            st.session_state.prod_up = random.randint(1, st.session_state.unid_up)
        if st.button("Generar valores aleatorios UP"): rnd_up()
        costo_up = st.number_input("Costo", value=st.session_state.get("costo_up", 20000), key="costo_up")
        salv_up = st.number_input("Salvamento", value=st.session_state.get("salv_up", 2000), key="salv_up")
        unid_up = st.number_input("Unidades Totales Estimadas", min_value=1, value=st.session_state.get("unid_up", 5000), key="unid_up")
        prod_up = st.number_input("Unidades Producidas este período", min_value=0, max_value=unid_up, value=st.session_state.get("prod_up", 1000), key="prod_up")
        user_up = st.number_input("Tu Depreciación Período", value=0.0, format="%.2f", key="user_up")
        if st.button("Validar Respuesta UP"):
            dep_unit = (costo_up - salv_up) / unid_up
            correcto = dep_unit * prod_up
            if np.isclose(user_up, correcto):
                st.success(f"¡Correcto! Depreciación período = {correcto:.2f}")
            else:
                st.error("Incorrecto.")
                st.info("Recuerda: (Costo - Salvamento)/Unidades totales * Unidades producidas")

    # Navegación con un solo clic
    col1, col2 = st.columns(2)
    with col1:
        if idx > 0:
            st.button("Anterior", on_click=dep_go_prev, key="dep_prev_btn")
        else:
            col1.write("")
    with col2:
        if idx < total - 1:
            st.button("Siguiente", on_click=dep_go_next, args=(total,), key="dep_next_btn")
        else:
            col2.write("")

# Práctica Inventarios
def ejercicios_valoracion_inventarios():
    st.header("Ejercicios de Valoración de Inventarios")
    st.write("...")

# Práctica Depreciaciones Gráfico
def ejercicios_depreciaciones_grafico():
    st.header("Ejercicios de Depreciaciones - Gráfico")
    costo    = st.number_input("Costo total", value=10000)
    residual = st.number_input("Valor residual", value=1000)
    vida     = st.number_input("Vida útil (años)", value=5)
    if vida > 0:
        dep  = (costo - residual) / vida
        vals = costo - dep * np.arange(vida + 1)
        fig, ax = plt.subplots()
        ax.plot(np.arange(vida + 1), vals, marker='o')
        st.pyplot(fig)
    else:
        st.error("Vida útil debe ser >0")

# Chat Contable
def chat_contable():
    st.title("Chat Contable")
    pregunta = st.text_input("Escribe tu pregunta relacionada con contabilidad:", key="chat_pregunta")
    if st.button("Enviar pregunta", key="btn_chat"):
        if pregunta.strip() == "":
            st.error("Por favor, ingresa una pregunta.")
        else:
            st.info("Esto puede tardar un momento...")

            prompt = (
                "Actúa como un experto en contabilidad. "
                "Si la pregunta del usuario no está relacionada con contabilidad, responde: "
                "'El chat contable es solo para preguntas de contabilidad'. "
                "De lo contrario, responde la pregunta de manera clara y detallada. "
                f"Pregunta del usuario: '{pregunta}'."
            )
            response = openai.ChatCompletion.create(
                extra_headers={},
                extra_body={},
                model="deepseek/deepseek-chat-v3-0324:free",
                messages=[{"role": "user", "content": prompt}]
            )
            respuesta = response.choices[0].message.content.strip()
            st.info(respuesta)

            # Generar el PDF en memoria y manejar caracteres no Latin-1
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.multi_cell(0, 10, f"Pregunta:\n{pregunta}\n\nRespuesta:\n{respuesta}")

            # Dest="S" devuelve el PDF como str; luego lo codificamos ignorando caracteres no Latin-1
            pdf_str = pdf.output(dest="S")
            pdf_bytes = pdf_str.encode("latin-1", errors="ignore")

            st.download_button(
                label="📄 Descargar chat en PDF",
                data=pdf_bytes,
                file_name="chat_contable.pdf",
                mime="application/pdf"
            )

# Panel Admin completo
def admin_panel():
    st.header("Administrador de Usuarios")
    users = st.session_state.users

    # Mostrar usuarios actuales
    df = pd.DataFrame([
        {"Usuario": u, "Contraseña": info["password"], "Rol": info["role"]}
        for u, info in users.items()
    ])
    st.subheader("Usuarios actuales")
    st.dataframe(df)

    st.markdown("---")
    # Crear usuario
    st.subheader("Crear nuevo usuario")
    new_user = st.text_input("Nombre de usuario", key="admin_new_user")
    new_pass = st.text_input("Contraseña", type="password", key="admin_new_pass")
    new_role = st.selectbox("Rol", ["user", "admin"], key="admin_new_role")
    if st.button("Agregar usuario"):
        if not new_user or not new_pass:
            st.error("Completa todos los campos.")
        elif new_user in users:
            st.error("El usuario ya existe.")
        else:
            st.session_state.users[new_user] = {"password": new_pass, "role": new_role}
            st.success(f"Usuario '{new_user}' agregado.")

    st.markdown("---")
    # Editar usuario
    st.subheader("Editar usuario")
    edit_user = st.selectbox("Selecciona usuario", list(users.keys()), key="admin_edit_select")
    if edit_user:
        edit_pass = st.text_input("Nueva contraseña", value=users[edit_user]["password"], key="admin_edit_pass")
        edit_role = st.selectbox("Nuevo rol", ["user", "admin"],
                                 index=0 if users[edit_user]["role"] == "user" else 1,
                                 key="admin_edit_role")
        if st.button("Actualizar usuario"):
            st.session_state.users[edit_user] = {"password": edit_pass, "role": edit_role}
            st.success(f"Usuario '{edit_user}' actualizado.")

    st.markdown("---")
    # Eliminar usuario
    st.subheader("Eliminar usuario")
    del_user = st.selectbox("Selecciona usuario a eliminar", list(users.keys()), key="admin_del_select")
    if st.button("Eliminar usuario"):
        if del_user == st.session_state.username:
            st.error("No puedes eliminar la cuenta con la que estás conectado.")
        else:
            del st.session_state.users[del_user]
            st.success(f"Usuario '{del_user}' eliminado.")

# App principal
def main_app():
    if st.session_state.show_portada:
        mostrar_portada()
        return
    st.sidebar.title("Menú")
    opts=["Teoría","Práctica","Chat Contable"]
    if st.session_state.users[st.session_state.username]["role"]=="admin":
        opts.append("Administrador")
    sel = st.sidebar.radio("Categoría", opts)
    if sel=="Teoría":
        sub = st.sidebar.radio("Opciones", ["Valoración de inventarios","Depreciaciones"])
        if sub=="Valoración de inventarios":
            mostrar_valoracion_inventarios()
        else:
            mostrar_depreciaciones_teoria()
    elif sel=="Práctica":
        sub = st.sidebar.radio("Opciones", ["Ejercicios inventarios","Ejercicios depreciaciones"])
        if sub=="Ejercicios inventarios":
            ejercicios_valoracion_inventarios()
        else:
            ejercicios_depreciaciones_grafico()
    elif sel=="Chat Contable":
        chat_contable()
    else:
        admin_panel()

    # AQUI: Botón Cerrar Sesión sin doble clic
    if st.sidebar.button("Cerrar Sesión", key="btn_logout"):
        logout()

# Entry
def main():
    init_session()  # AQUI: asegura inicialización de login_error
    if st.session_state.authenticated:
        main_app()
    else:
        st.title("Iniciar Sesión")
        with st.form(key="login_form"):
            st.text_input("Usuario", key="login_raw_user")
            st.text_input("Contraseña", type="password", key="login_password")
            submitted = st.form_submit_button("Ingresar")
            if submitted:
                do_login()
        if st.session_state.login_error:
            st.error(st.session_state.login_error)

if __name__ == '__main__':
    main()
