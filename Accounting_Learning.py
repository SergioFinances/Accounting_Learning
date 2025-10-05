# -*- coding: utf-8 -*-
# =========================================================
#   Herramienta Contable - Inventarios Gamificados (sin Mongo)
#   Niveles 1-4 con TTS y práctica + feedback IA
#   Auto-desbloqueo y salto entre niveles al aprobar
#   OpenRouter + DeepSeek (v3.1:free)
#   Fecha: 2025-10-05
# =========================================================

import os
import random
from datetime import datetime

import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

# ===========================
# Configuración Streamlit
# ===========================
st.set_page_config(
    page_title="Herramienta Contable - Inventarios",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===========================
# Carga variables de entorno
# ===========================
load_dotenv()
SURVEY_URL = os.getenv("SURVEY_URL", "https://example.com/encuesta-final")

# ===========================
# IA (DeepSeek vía OpenRouter)
# ===========================
from openai import OpenAI

api_key = os.getenv("OPENROUTER_API_KEY")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
    default_headers={
        "HTTP-Referer": "http://localhost:8501",   # Si despliegas, pon tu dominio
        "X-Title": "Herramienta Contable"
    }
)

DEEPSEEK_MODEL = "deepseek/deepseek-chat-v3.1:free"

def ia_feedback(prompt_user: str, role_desc: str = "tutor") -> str:
    """
    Llama a DeepSeek (OpenRouter) para dar feedback educativo.
    - role_desc: "tutor", "corrector", "coach", "mentor" para matizar el tono.
    - Devuelve texto corto en español (máx 6 líneas).
    """
    if not api_key:
        return "Feedback IA no disponible (falta OPENROUTER_API_KEY). Tus resultados se validaron localmente."
    try:
        completion = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Eres un " + role_desc + " de contabilidad empático y claro. "
                        "Responde SIEMPRE en español y en máximo 6 líneas, "
                        "indicando: (1) qué está bien/mal, (2) por qué, (3) 1 truco útil. "
                        "Si faltan datos, di qué falta antes de concluir."
                    )
                },
                {"role": "user", "content": prompt_user}
            ],
            temperature=0.3,
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        return f"No pude generar feedback con IA ahora. (Detalle: {e})"

# ===========================
# Utilidades UI
# ===========================
def fmt(v, dec=1):
    """Formato ES para miles y coma decimal."""
    if isinstance(v, (int, np.integer)) or (isinstance(v, float) and abs(v - int(v)) < 1e-12):
        try:
            s = f"{int(round(v)):,}".replace(",", ".")
            return s
        except Exception:
            return str(v)
    try:
        s = f"{v:,.{dec}f}".replace(",", "X").replace(".", ",").replace("X", ".")
        return s
    except Exception:
        return str(v)

def peso(v):
    return f"${fmt(v,2)}"

def celebracion_confeti():
    st.balloons()

def speak_block(texto: str, key_prefix: str, lang_hint="es"):
    """
    Control TTS del navegador con selector de voz + velocidad + tono (Web Speech API).
    """
    escaped = (
        texto.replace("\\", "\\\\")
             .replace("`", "\\`")
             .replace("\n", "\\n")
             .replace('"', '\\"')
    )
    html = f"""
    <div style="padding:8px;border:1px solid #eee;border-radius:10px;">
      <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;">
        <label for="{key_prefix}-voice">Voz:</label>
        <select id="{key_prefix}-voice"></select>

        <label for="{key_prefix}-rate">Velocidad:</label>
        <input id="{key_prefix}-rate" type="range" min="0.7" max="1.3" step="0.05" value="1.0" />

        <label for="{key_prefix}-pitch">Tono:</label>
        <input id="{key_prefix}-pitch" type="range" min="0.7" max="1.3" step="0.05" value="1.0" />

        <button id="{key_prefix}-play">🔊 Escuchar</button>
        <button id="{key_prefix}-stop">⏹️ Detener</button>
      </div>
      <small>Tip: en Chrome suelen aparecer voces como <em>Google español</em> o <em>Microsoft Sabina</em>.</small>
    </div>
    <script>
      (function() {{
        const text = "{escaped}";
        const langHint = "{lang_hint}".toLowerCase();
        const sel = document.getElementById("{key_prefix}-voice");
        const rate = document.getElementById("{key_prefix}-rate");
        const pitch = document.getElementById("{key_prefix}-pitch");
        const btnPlay = document.getElementById("{key_prefix}-play");
        const btnStop = document.getElementById("{key_prefix}-stop");

        function populateVoices() {{
          const voices = window.speechSynthesis.getVoices();
          sel.innerHTML = "";
          const score = (v) => {{
            const n = (v.name + " " + v.lang).toLowerCase();
            let s = 0;
            if (n.includes("es")) s += 5;
            if (n.includes("spanish")) s += 4;
            if (n.includes("mex")) s += 3;
            if (n.includes("col")) s += 3;
            if (n.includes("sabina")) s += 3;
            if (n.includes("google")) s += 2;
            if (n.includes(langHint)) s += 2;
            return s;
          }};
          voices.sort((a,b)=>score(b)-score(a));
          voices.forEach((v, i) => {{
            const opt = document.createElement("option");
            opt.value = i;
            opt.textContent = v.name + " (" + v.lang + ")";
            sel.appendChild(opt);
          }});
        }}

        populateVoices();
        if (typeof speechSynthesis !== "undefined") {{
          speechSynthesis.onvoiceschanged = populateVoices;
        }}

        btnPlay.onclick = () => {{
          try {{
            if (speechSynthesis.speaking) speechSynthesis.cancel();
            const voices = window.speechSynthesis.getVoices();
            const u = new SpeechSynthesisUtterance(text);
            const chosen = voices[sel.value] || voices[0];
            u.voice = chosen;
            u.rate = parseFloat(rate.value);
            u.pitch = parseFloat(pitch.value);
            speechSynthesis.speak(u);
          }} catch (e) {{}}
        }};
        btnStop.onclick = () => speechSynthesis.cancel();
      }})();
    </script>
    """
    components.html(html, height=120)

def section_title(icon, title):
    st.markdown(f"### {icon} {title}")

# ===========================
# Login en memoria (sin Mongo)
# ===========================
DEFAULT_USERS = {
    "admin": {"password": "AdminSeguro#2025", "role": "admin"},
    "estudiante": {"password": "1234", "role": "user"},
}

def init_session():
    st.session_state.setdefault("authenticated", False)
    st.session_state.setdefault("login_error", "")
    st.session_state.setdefault("username", "")
    st.session_state.setdefault("users", DEFAULT_USERS.copy())
    st.session_state.setdefault("all_progress", {})  # username -> progress
    st.session_state.setdefault("force_go_level2", False)
    st.session_state.setdefault("force_go_level3", False)
    st.session_state.setdefault("force_go_level4", False)

def check_credentials(user, password):
    users = st.session_state.users
    return user in users and users[user]["password"] == password

def do_login():
    user = st.session_state.login_raw_user.strip().lower()
    pwd  = st.session_state.login_password
    if not user or not pwd:
        st.session_state.login_error = "Por favor, ingresa usuario y contraseña."
        return
    if check_credentials(user, pwd):
        st.session_state.authenticated = True
        st.session_state.username      = user
        st.session_state.login_error   = ""
        if user not in st.session_state.all_progress:
            st.session_state.all_progress[user] = default_progress()
    else:
        st.session_state.login_error = "Credenciales incorrectas."

def logout():
    st.session_state.authenticated = False
    st.session_state.username = ""
    st.session_state.login_error = ""

# ===========================
# Progreso en memoria
# ===========================
def default_progress():
    return {
        "level1": {"passed": False, "date": None, "score": None},
        "level2": {"passed": False, "date": None, "score": None},
        "level3": {"passed": False, "date": None, "score": None},
        "level4": {"passed": False, "date": None, "score": None},
        "completed_survey": False
    }

def get_progress(username):
    allp = st.session_state.all_progress
    if username not in allp:
        allp[username] = default_progress()
    return allp[username]

def save_progress(username, level_key, passed: bool, score=None):
    prog = get_progress(username)
    prog[level_key] = {"passed": passed, "date": datetime.utcnow(), "score": score}
    st.session_state.all_progress[username] = prog

# ===========================
# Sidebar navegación por nivel
# ===========================
def sidebar_nav(username):
    prog = get_progress(username)
    st.sidebar.title("Niveles")

    options = ["Nivel 1: Introducción a Inventarios"]  # solo este al inicio
    if prog["level1"]["passed"]:
        options.append("Nivel 2: Métodos (PP/PEPS/UEPS)")
    if prog["level2"]["passed"]:
        options.append("Nivel 3: Devoluciones")
    if prog["level3"]["passed"]:
        options.append("Nivel 4: Estado de Resultados")

    # Auto-saltos al aprobar
    if prog["level1"]["passed"] and st.session_state.get("force_go_level2"):
        st.session_state["sidebar_level_select"] = "Nivel 2: Métodos (PP/PEPS/UEPS)"
        st.session_state["force_go_level2"] = False
    if prog["level2"]["passed"] and st.session_state.get("force_go_level3"):
        st.session_state["sidebar_level_select"] = "Nivel 3: Devoluciones"
        st.session_state["force_go_level3"] = False
    if prog["level3"]["passed"] and st.session_state.get("force_go_level4"):
        st.session_state["sidebar_level_select"] = "Nivel 4: Estado de Resultados"
        st.session_state["force_go_level4"] = False

    sel = st.sidebar.radio("Ir a:", options, key="sidebar_level_select")
    st.sidebar.markdown("---")
    def badge(ok): return "✅" if ok else "🔒"
    st.sidebar.caption("Progreso:")
    st.sidebar.write(f"{badge(prog['level1']['passed'])} Nivel 1")
    st.sidebar.write(f"{badge(prog['level2']['passed'])} Nivel 2")
    st.sidebar.write(f"{badge(prog['level3']['passed'])} Nivel 3")
    st.sidebar.write(f"{badge(prog['level4']['passed'])} Nivel 4")
    st.sidebar.button("Cerrar Sesión", on_click=logout, key="logout_btn")

    st.sidebar.markdown("---")
    if st.sidebar.button("🔍 Probar conexión IA"):
        fb = ia_feedback("Di 'OK' si recibiste este mensaje.", role_desc="asistente")
        st.sidebar.info("Respuesta IA: " + fb)

    if not api_key:
        st.sidebar.warning("⚠️ Falta OPENROUTER_API_KEY en tu entorno. El feedback IA caerá a local.")

    # Si todos los niveles pasados, mostrar encuesta
    if prog["level1"]["passed"] and prog["level2"]["passed"] and prog["level3"]["passed"] and prog["level4"]["passed"]:
        st.sidebar.markdown("---")
        st.sidebar.success("🎯 ¡Completaste los 4 niveles!")
        st.sidebar.markdown(f"[📝 Responder encuesta final]({SURVEY_URL})", unsafe_allow_html=True)

    return sel

# ===========================
# Helpers de métodos (Nivel 2 & 3)
# ===========================
def cogs_promedio_ponderado(inv_inicial_qty, inv_inicial_cost,
                            compras_qty, compras_cost, venta_qty):
    """
    PP perpetuo simplificado con una compra antes de la venta.
    """
    total_qty = inv_inicial_qty + compras_qty
    total_val = inv_inicial_qty * inv_inicial_cost + compras_qty * compras_cost
    if total_qty <= 0:
        return 0.0, 0.0, 0.0
    costo_prom = total_val / total_qty
    cogs = min(venta_qty, total_qty) * costo_prom
    inv_final_qty = max(total_qty - venta_qty, 0)
    inv_final_val = inv_final_qty * costo_prom
    return cogs, inv_final_qty, inv_final_val

def cogs_peps(inv_inicial_qty, inv_inicial_cost,
              compras_qty, compras_cost, venta_qty):
    """
    FIFO: salen primero las unidades más antiguas (inv inicial).
    """
    cogs = 0.0
    venta_rest = venta_qty

    # Consumir inventario inicial
    tomar = min(inv_inicial_qty, venta_rest)
    cogs += tomar * inv_inicial_cost
    venta_rest -= tomar

    # Consumir compras si falta
    if venta_rest > 0:
        tomar = min(compras_qty, venta_rest)
        cogs += tomar * compras_cost
        venta_rest -= tomar

    # Capas remanentes
    total_ini_usado = min(inv_inicial_qty, venta_qty)
    inv_ini_rem = inv_inicial_qty - total_ini_usado
    total_comp_usado = max(0, venta_qty - inv_inicial_qty)
    total_comp_usado = min(compras_qty, total_comp_usado)
    inv_comp_rem = compras_qty - total_comp_usado

    inv_final_val = inv_ini_rem * inv_inicial_cost + inv_comp_rem * compras_cost
    inv_final_qty = inv_ini_rem + inv_comp_rem

    return cogs, inv_final_qty, inv_final_val

def cogs_ueps(inv_inicial_qty, inv_inicial_cost,
              compras_qty, compras_cost, venta_qty):
    """
    LIFO (solo educativo): salen primero las unidades más recientes (compras).
    """
    cogs = 0.0
    venta_rest = venta_qty

    # Consumir compras primero
    tomar = min(compras_qty, venta_rest)
    cogs += tomar * compras_cost
    venta_rest -= tomar

    # Luego inventario inicial
    if venta_rest > 0:
        tomar = min(inv_inicial_qty, venta_rest)
        cogs += tomar * inv_inicial_cost
        venta_rest -= tomar

    # Remanentes
    total_comp_usado = min(compras_qty, venta_qty)
    inv_comp_rem = compras_qty - total_comp_usado
    total_ini_usado = max(0, venta_qty - compras_qty)
    total_ini_usado = min(inv_inicial_qty, total_ini_usado)
    inv_ini_rem = inv_inicial_qty - total_ini_usado

    inv_final_val = inv_ini_rem * inv_inicial_cost + inv_comp_rem * compras_cost
    inv_final_qty = inv_ini_rem + inv_comp_rem
    return cogs, inv_final_qty, inv_final_val

# ======== Nivel 3 helpers (devoluciones) ========
def aplicar_devoluciones_periodico(method, inv_q, inv_c, cmp_q, cmp_c, vta_q,
                                   dev_comp_q, dev_venta_q):
    """
    Versión pedagógica (simplificada) para devoluciones en sistema periódico:
    - Devolución de compra (a proveedor): reduce compras (qty) y su costo (misma capa de compras).
    - Devolución de venta (cliente devuelve): consideramos que reduce ventas del período
      y, a efectos de costo, es como si hubieras vendido menos unidades (vta neta).
      => vta_neta = max(vta_q - dev_venta_q, 0).
    - Calculamos COGS e InvF sobre: compras_net = max(cmp_q - dev_comp_q, 0), ventas_net = vta_neta.
    - Esto aproxima el efecto en el periódico (sin registros perpetuos detallados).
    """
    cmp_net_q = max(cmp_q - dev_comp_q, 0)
    vta_net_q = max(vta_q - dev_venta_q, 0)

    if method == "PP":
        return cogs_promedio_ponderado(inv_q, inv_c, cmp_net_q, cmp_c, vta_net_q)
    elif method == "PEPS":
        return cogs_peps(inv_q, inv_c, cmp_net_q, cmp_c, vta_net_q)
    else:  # UEPS educativo
        return cogs_ueps(inv_q, inv_c, cmp_net_q, cmp_c, vta_net_q)

# ===========================
# NIVEL 1
# ===========================
def page_level1(username):
    st.title("Nivel 1 · Introducción a la valoración de inventarios")

    tabs = st.tabs(["🎧 Teoría profunda", "🛠 Ejemplo guiado", "🎮 Práctica interactiva (IA)", "🏁 Evaluación para aprobar"])

    # TEORÍA
    with tabs[0]:
        st.subheader("¿Qué es valorar inventarios y por qué impacta tu utilidad?")
        teoria = (
            "Valorar inventarios es asignar un **costo monetario** a las existencias que mantiene una empresa para vender. "
            "Ese costo aparece como **activo** (Inventarios) y determina el **Costo de Ventas (COGS)** en el estado de resultados, "
            "afectando la **utilidad bruta**. En un **sistema periódico**, no actualizas inventarios con cada venta: "
            "acumulas durante el período y cierras con la fórmula base:\n\n"
            "  **COGS = Inventario Inicial + Compras - Devoluciones - Inventario Final**\n\n"
            "- **InvI:** lo que tenías al empezar.\n"
            "- **Compras:** adquisiciones del período (pueden incluir costos necesarios para poner el inventario disponible).\n"
            "- **Devoluciones:** típicamente restan a Compras cuando devuelves a proveedor.\n"
            "- **InvF:** lo que queda al cierre; su **valoración** depende del método que uses (verás PP/PEPS/UEPS en el Nivel 2).\n\n"
            "Regla mental: imagina una **mochila de costo**. Entra InvI y Compras; si devuelves, sacas una parte (Devoluciones). "
            "Al final miras qué queda dentro (InvF). **Lo que salió** para vender es el **COGS**."
        )
        st.write(teoria)
        speak_block(teoria, key_prefix="teo-n1", lang_hint="es")

        st.markdown("---")
        duda = st.text_area("¿Tienes una duda sobre la teoría? Escríbela y la IA te contesta:", key="n1_teo_duda")
        if st.button("💬 Resolver duda con IA", key="n1_teo_duda_btn"):
            prompt = (
                "Responde de forma breve y clara a esta duda sobre valoración de inventarios "
                "en sistema periódico usando la fórmula COGS = InvI + Compras - Devol - InvF. "
                f"Duda del estudiante: {duda}"
            )
            fb = ia_feedback(prompt, role_desc="mentor")
            st.info(fb)

        with st.expander("📌 Nota contable/NIIF"):
            st.markdown(
                "Bajo NIIF, debes usar un método de costo razonable y **consistente**. "
                "En aprendizaje verás UEPS como referencia, aunque **no es aceptado por NIIF plenas**."
            )

    # EJEMPLO
    with tabs[1]:
        st.subheader("Ejemplo guiado · paso a paso")
        colL, colR = st.columns([1,2], gap="large")
        with colL:
            st.caption("Ingresa/ajusta datos")
            inv0 = st.number_input("Inventario Inicial (InvI)", min_value=0.0, value=1500.0, step=100.0, key="n1_ex_inv0")
            compras = st.number_input("Compras del período", min_value=0.0, value=2700.0, step=100.0, key="n1_ex_compras")
            devol = st.number_input("Devoluciones (a proveedor)", min_value=0.0, value=200.0, step=50.0, key="n1_ex_devol")
            invf = st.number_input("Inventario Final (InvF)", min_value=0.0, value=1300.0, step=100.0, key="n1_ex_invf")

        with colR:
            st.caption("Desglose y explicación")
            st.write(f"**1) InvI + Compras** → {peso(inv0)} + {peso(compras)} = **{peso(inv0+compras)}**")
            st.write(f"**2) − Devoluciones**  → {peso(inv0+compras)} − {peso(devol)} = **{peso(inv0+compras-devol)}**")
            st.write(f"**3) − InvF**          → {peso(inv0+compras-devol)} − {peso(invf)} = **{peso(inv0+compras-devol-invf)}**")
            cogs = inv0 + compras - devol - invf
            st.success(f"**COGS (Costo de Ventas)** = {peso(cogs)}")
            st.caption("Interpretación: la ‘mochila de costo’ se llenó con InvI y Compras; devolviste parte (Devoluciones) "
                       "y lo que quedó al cierre (InvF) no salió a ventas. El resto es COGS.")

        st.markdown("—")
        st.write("**Mini reto**: explica qué pasaría con el COGS si **no hubiera devoluciones** y el **Inventario Final fuera muy pequeño**.")
        razonamiento = st.text_area("Tu razonamiento (la IA te comenta):", key="n1_ex_raz")

        if st.button("💬 Comentar con IA", key="n1_ex_fb"):
            prompt = (
                "Evalúa si el razonamiento es coherente con COGS = InvI + Compras - Devoluciones - InvF. "
                f"Datos: InvI={inv0}, Compras={compras}, Devoluciones={devol}, InvF={invf}. "
                f"Texto del estudiante: {razonamiento}"
            )
            fb = ia_feedback(prompt, role_desc="corrector")
            st.info(fb)

    # PRÁCTICA
    with tabs[2]:
        st.subheader("Práctica interactiva · escenarios aleatorios")
        st.caption("Completa el cálculo. Puedes generar otro escenario y validar con IA.")

        def new_case():
            inv0 = random.randint(500, 4000)
            compras = random.randint(800, 5000)
            devol = random.randint(0, int(compras*0.3))
            invf = random.randint(0, inv0 + compras - devol)
            st.session_state.n1p_inv0 = float(inv0)
            st.session_state.n1p_compras = float(compras)
            st.session_state.n1p_devol = float(devol)
            st.session_state.n1p_invf = float(invf)

        if "n1p_inv0" not in st.session_state:
            new_case()

        cols = st.columns(4)
        with cols[0]:
            st.metric("Inv. Inicial", peso(st.session_state.n1p_inv0))
        with cols[1]:
            st.metric("Compras", peso(st.session_state.n1p_compras))
        with cols[2]:
            st.metric("Devoluciones", peso(st.session_state.n1p_devol))
        with cols[3]:
            st.metric("Inv. Final", peso(st.session_state.n1p_invf))

        st.button("🔄 Nuevo escenario", on_click=new_case, key="n1_practice_new")

        user_cogs = st.number_input("Tu COGS ($)", min_value=0.0, value=0.0, step=10.0, key="n1_practice_user_cogs")
        user_comment = st.text_area("Explica tu cálculo (la IA te corrige):", key="n1_practice_comment")

        if st.button("✅ Validar práctica", key="n1_practice_validate"):
            inv0 = st.session_state.n1p_inv0
            compras = st.session_state.n1p_compras
            devol = st.session_state.n1p_devol
            invf = st.session_state.n1p_invf
            correct = inv0 + compras - devol - invf
            if abs(user_cogs - correct) <= 0.5:
                st.success(f"¡Correcto! COGS = {peso(correct)}")
            else:
                st.error(f"No coincide. El COGS esperado era {peso(correct)}")
            prompt = (
                f"Valida el cálculo del estudiante: COGS_est={user_cogs:.2f}. "
                f"Datos: InvI={inv0:.2f}, Compras={compras:.2f}, Devol={devol:.2f}, InvF={invf:.2f}. "
                f"COGS_correcto={correct:.2f}. Comentario del estudiante: {user_comment}"
            )
            fb = ia_feedback(prompt, role_desc="tutor")
            with st.expander("💬 Feedback de la IA"):
                st.write(fb)

    # EVALUACIÓN
    with tabs[3]:
        st.subheader("Evaluación final del Nivel 1")
        st.caption("Necesitas acertar **2 de 3** para aprobar y desbloquear el Nivel 2.")

        q1 = st.radio("1) En sistema periódico, ¿cuándo conoces con certeza el COGS?",
                      ["En cada venta", "Al cierre del período"], index=None, key="n1_eval_q1")
        q2 = st.radio("2) ¿Cuál de estos **disminuye** el COGS en la fórmula periódica?",
                      ["Devoluciones de compra", "Compras"], index=None, key="n1_eval_q2")
        q3 = st.radio("3) Selecciona la fórmula correcta:",
                      ["InvI + Compras + Devoluciones - InvF",
                       "InvI + Compras - Devoluciones - InvF",
                       "InvI - Compras + Devoluciones + InvF"], index=None, key="n1_eval_q3")

        if st.button("🧪 Validar evaluación", key="n1_eval_btn"):
            correct = {
                "n1_eval_q1": "Al cierre del período",
                "n1_eval_q2": "Devoluciones de compra",
                "n1_eval_q3": "InvI + Compras - Devoluciones - InvF"
            }
            answers = {"n1_eval_q1": q1, "n1_eval_q2": q2, "n1_eval_q3": q3}
            score = sum(1 for k,v in answers.items() if v == correct[k])
            passed = score >= 2

            prompt = (
                f"Nivel 1 evaluación. Respuestas estudiante: {answers}. Correctas: {correct}. "
                f"Aciertos: {score}/3. Da un feedback amable (≤6 líneas) y sugiere 1 repaso si falló."
            )
            fb = ia_feedback(prompt, role_desc="coach")

            if passed:
                st.success(f"¡Aprobado! Aciertos {score}/3 🎉 Se habilitará el Nivel 2 en el menú.")
                celebracion_confeti()
            else:
                st.error(f"No aprobado. Aciertos {score}/3. Repasa la teoría y vuelve a intentar.")
            with st.expander("💬 Feedback de la IA"):
                st.write(fb)

            save_progress(username, "level1", passed, score=score)

            if passed:
                st.session_state["force_go_level2"] = True
                st.rerun()

# ===========================
# NIVEL 2
# ===========================
def page_level2(username):
    st.title("Nivel 2 · Métodos de valoración: Promedio Ponderado, PEPS (FIFO) y UEPS (LIFO)")

    tabs = st.tabs(["🎧 Teoría", "🛠 Ejemplo comparativo", "🎮 Práctica con IA", "🏁 Evaluación para aprobar"])

    # TEORÍA
    with tabs[0]:
        teoria = (
            "**Objetivo:** asignar costo a las salidas (ventas) y al inventario final según una regla.\n\n"
            "1) **Promedio Ponderado (PP)**: mezcla costos en un solo **costo promedio**. "
            "Suaviza variaciones; es común en procesos continuos. \n\n"
            "2) **PEPS (FIFO)**: salen primero las **primeras capas** (las más antiguas). "
            "En **inflación**, tiende a **COGS menor** y **inventario final mayor**. \n\n"
            "3) **UEPS (LIFO)** (solo educativo): salen primero las **últimas capas** (las más recientes). "
            "En **inflación**, tiende a **COGS mayor** y **inventario final menor**. *No admitido por NIIF plenas.*\n\n"
            "👉 Todos respetan: **COGS + InvF = InvI + Compras (netas)**, pero **distribuyen** ese total distinto."
        )
        st.write(teoria)
        speak_block(teoria, key_prefix="teo-n2", lang_hint="es")

        with st.expander("ℹ️ Nota rápida"):
            st.markdown(
                "- En PP, recalculas el costo promedio cuando entra un nuevo lote (versión perpetua). "
                "Aquí usamos un caso único de compras y luego una venta.\n"
                "- En FIFO y LIFO, piensa en **capas** (lotes) y vas consumiendo en orden."
            )

    # EJEMPLO
    with tabs[1]:
        st.subheader("Caso base: compara PP, PEPS y UEPS")
        c1, c2 = st.columns(2)
        with c1:
            inv_q = st.number_input("Inv. inicial (unid)", min_value=0, value=100, step=10, key="n2_e_invq")
            inv_c = st.number_input("Costo unitario Inv. inicial", min_value=0.0, value=10.0, step=0.5, key="n2_e_invc")
            cmp_q = st.number_input("Compras (unid)", min_value=0, value=120, step=10, key="n2_e_cmpq")
            cmp_c = st.number_input("Costo unitario Compras", min_value=0.0, value=12.0, step=0.5, key="n2_e_cmpc")
            vta_q = st.number_input("Venta (unid)", min_value=0, value=150, step=10, key="n2_e_vtaq")
        with c2:
            st.info("💡 Sugerencia: sube el costo de compras por encima del inicial para simular **inflación**.")

        pp_cogs, pp_if_q, pp_if_val = cogs_promedio_ponderado(inv_q, inv_c, cmp_q, cmp_c, vta_q)
        fifo_cogs, fifo_if_q, fifo_if_val = cogs_peps(inv_q, inv_c, cmp_q, cmp_c, vta_q)
        lifo_cogs, lifo_if_q, lifo_if_val = cogs_ueps(inv_q, inv_c, cmp_q, cmp_c, vta_q)

        st.markdown("#### Resultados")
        st.write(f"**PP** → COGS: {peso(pp_cogs)} · InvF: {pp_if_q} u. ({peso(pp_if_val)})")
        st.write(f"**PEPS (FIFO)** → COGS: {peso(fifo_cogs)} · InvF: {fifo_if_q} u. ({peso(fifo_if_val)})")
        st.write(f"**UEPS (LIFO)** → COGS: {peso(lifo_cogs)} · InvF: {lifo_if_q} u. ({peso(lifo_if_val)})")

        st.markdown("---")
        comentario = st.text_area("Explica con tus palabras por qué cambian COGS e InvF entre métodos:", key="n2_e_comment")
        if st.button("💬 Comentar con IA", key="n2_e_fb"):
            prompt = (
                "Evalúa el comentario del estudiante sobre las diferencias PP/FIFO/LIFO. "
                f"Datos: InvI={inv_q} u @ {inv_c}, Compras={cmp_q} u @ {cmp_c}, Venta={vta_q} u. "
                f"Resultados: PP_COGS={pp_cogs:.2f}, FIFO_COGS={fifo_cogs:.2f}, LIFO_COGS={lifo_cogs:.2f}. "
                f"PP_InvF={pp_if_val:.2f}, FIFO_InvF={fifo_if_val:.2f}, LIFO_InvF={lifo_if_val:.2f}. "
                f"Comentario: {comentario}"
            )
            fb = ia_feedback(prompt, role_desc="corrector")
            st.info(fb)

    # PRÁCTICA
    with tabs[2]:
        st.subheader("Práctica: elige método y calcula COGS e InvF")

        def nuevo_caso():
            inv_q = random.randint(50, 180)
            inv_c = random.choice([8, 9, 10, 11, 12])
            cmp_q = random.randint(60, 200)
            cmp_c = random.choice([inv_c - 1, inv_c, inv_c + 1, inv_c + 2])
            cmp_c = max(1, float(cmp_c))
            vta_q = random.randint(30, inv_q + cmp_q)
            st.session_state.n2p = dict(inv_q=inv_q, inv_c=float(inv_c),
                                        cmp_q=cmp_q, cmp_c=float(cmp_c), vta_q=vta_q)

        if "n2p" not in st.session_state:
            nuevo_caso()

        meta = st.session_state.n2p
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("InvI (u.)", meta["inv_q"])
        m2.metric("InvI ($/u)", peso(meta["inv_c"]))
        m3.metric("Compras (u.)", meta["cmp_q"])
        m4.metric("Compras ($/u)", peso(meta["cmp_c"]))
        m5.metric("Venta (u.)", meta["vta_q"])

        st.button("🔄 Nuevo caso", on_click=nuevo_caso, key="n2_p_new")

        metodo = st.selectbox("Selecciona el método", ["PP", "PEPS (FIFO)", "UEPS (LIFO)"], key="n2_p_m")
        user_cogs = st.number_input("Tu COGS ($)", min_value=0.0, value=0.0, step=10.0, key="n2_p_cogs")
        user_invf = st.number_input("Tu InvF ($)", min_value=0.0, value=0.0, step=10.0, key="n2_p_invf")
        explic = st.text_area("Explica tu procedimiento (la IA te corrige):", key="n2_p_explic")

        if st.button("✅ Validar práctica", key="n2_p_val"):
            inv_q, inv_c, cmp_q, cmp_c, vta_q = meta["inv_q"], meta["inv_c"], meta["cmp_q"], meta["cmp_c"], meta["vta_q"]
            if metodo.startswith("PP"):
                cogs, if_q, if_val = cogs_promedio_ponderado(inv_q, inv_c, cmp_q, cmp_c, vta_q)
            elif metodo.startswith("PEPS"):
                cogs, if_q, if_val = cogs_peps(inv_q, inv_c, cmp_q, cmp_c, vta_q)
            else:  # UEPS
                cogs, if_q, if_val = cogs_ueps(inv_q, inv_c, cmp_q, cmp_c, vta_q)

            ok_cogs = abs(user_cogs - cogs) <= 0.5
            ok_invf = abs(user_invf - if_val) <= 0.5

            if ok_cogs and ok_invf:
                st.success(f"¡Perfecto! COGS={peso(cogs)} · InvF={peso(if_val)}")
            elif ok_cogs:
                st.warning(f"COGS correcto ({peso(cogs)}), pero InvF esperado era {peso(if_val)}")
            elif ok_invf:
                st.warning(f"InvF correcto ({peso(if_val)}), pero COGS esperado era {peso(cogs)}")
            else:
                st.error(f"No coincide. COGS esperado {peso(cogs)} · InvF esperado {peso(if_val)}")

            prompt = (
                f"Corrige el cálculo del estudiante para el método {metodo}. "
                f"Datos: InvI={inv_q}u @ {inv_c}, Compras={cmp_q}u @ {cmp_c}, Venta={vta_q}u. "
                f"Respuesta estudiante: COGS={user_cogs:.2f}, InvF={user_invf:.2f}. "
                f"Correcto: COGS={cogs:.2f}, InvF={if_val:.2f}. Explicación del estudiante: {explic}"
            )
            fb = ia_feedback(prompt, role_desc="tutor")
            with st.expander("💬 Feedback de la IA"):
                st.write(fb)

    # EVALUACIÓN
    with tabs[3]:
        st.subheader("Evaluación final del Nivel 2")
        st.caption("Necesitas acertar **2 de 3** para aprobar y desbloquear el Nivel 3.")

        e1 = st.radio(
            "1) Con **inflación** (los costos de compra suben), ¿cuál tiende a mostrar **mayor COGS**?",
            ["Promedio Ponderado", "PEPS (FIFO)", "UEPS (LIFO)"],
            index=None, key="n2_q1"
        )
        e2 = st.radio(
            "2) En **PEPS (FIFO)**, el inventario final está compuesto por las unidades...",
            ["más recientes", "más antiguas"],
            index=None, key="n2_q2"
        )
        e3 = st.radio(
            "3) Bajo **inflación**, ¿cuál suele dar **mayor valor del inventario final**?",
            ["Promedio Ponderado", "PEPS (FIFO)", "UEPS (LIFO)"],
            index=None, key="n2_q3"
        )

        if st.button("🧪 Validar evaluación (Nivel 2)", key="n2_eval_btn"):
            correct = {
                "n2_q1": "UEPS (LIFO)",
                "n2_q2": "más recientes",
                "n2_q3": "PEPS (FIFO)",
            }
            answers = {"n2_q1": e1, "n2_q2": e2, "n2_q3": e3}
            score = sum(1 for k,v in answers.items() if v == correct[k])
            passed = score >= 2

            prompt = (
                f"Nivel 2 evaluación. Respuestas estudiante: {answers}. Correctas: {correct}. "
                f"Aciertos: {score}/3. Da feedback amable (≤6 líneas) y un tip por método."
            )
            fb = ia_feedback(prompt, role_desc="coach")

            if passed:
                st.success(f"¡Aprobado! Aciertos {score}/3 🎉 Se habilitará el Nivel 3 en el menú.")
                celebracion_confeti()
            else:
                st.error(f"No aprobado. Aciertos {score}/3. Repasa y vuelve a intentar.")
            with st.expander("💬 Feedback de la IA"):
                st.write(fb)

            save_progress(username, "level2", passed, score=score)

            if passed:
                st.session_state["force_go_level3"] = True
                st.rerun()

# ===========================
# NIVEL 3 · Devoluciones (compra y venta)
# ===========================
def page_level3(username):
    st.title("Nivel 3 · Casos avanzados: Devoluciones de compra y de venta")

    tabs = st.tabs(["🎧 Teoría", "🛠 Ejemplo con devoluciones", "🎮 Práctica con IA", "🏁 Evaluación para aprobar"])

    # TEORÍA
    with tabs[0]:
        teoria = (
            "**Contexto (periódico):**\n\n"
            "- **Devoluciones de compra** (a proveedor): reducen el costo de las compras del período. "
            "En fórmula, suelen presentarse como **Compras netas = Compras - Devoluciones**.\n"
            "- **Devoluciones de venta** (cliente devuelve): reducen ventas. Si evaluamos el **costo**, "
            "para simplificar en sistema periódico asumimos que es como si **hubieras vendido menos unidades**. "
            "Por tanto, para el flujo de costo, puedes trabajar con **ventas netas = Ventas - Devoluciones de venta**.\n\n"
            "En este nivel verás cómo cambian **COGS** e **Inventario Final** aplicando PP/FIFO/UEPS bajo estas devoluciones."
        )
        st.write(teoria)
        speak_block(teoria, key_prefix="teo-n3", lang_hint="es")

        with st.expander("Nota didáctica"):
            st.markdown(
                "Para fines pedagógicos en periódico, modelamos las devoluciones de venta como si disminuyeran las unidades vendidas netas. "
                "En la práctica, la imputación detallada del costo puede requerir registros perpetuos o supuestos consistentes."
            )

    # EJEMPLO
    with tabs[1]:
        st.subheader("Ejemplo guiado con devoluciones")
        c1, c2 = st.columns(2)
        with c1:
            inv_q = st.number_input("Inv. inicial (u.)", min_value=0, value=100, step=10, key="n3_e_invq")
            inv_c = st.number_input("Costo unitario Inv. inicial", min_value=0.0, value=10.0, step=0.5, key="n3_e_invc")
            cmp_q = st.number_input("Compras (u.)", min_value=0, value=120, step=10, key="n3_e_cmpq")
            cmp_c = st.number_input("Costo unitario Compras", min_value=0.0, value=12.0, step=0.5, key="n3_e_cmpc")
            dev_cmp = st.number_input("Devolución de compra (u.)", min_value=0, value=10, step=5, key="n3_e_devc")
            vta_q = st.number_input("Ventas (u.)", min_value=0, value=150, step=10, key="n3_e_vtas")
            dev_vta = st.number_input("Devolución de venta (u.)", min_value=0, value=5, step=5, key="n3_e_devv")
        with c2:
            st.info("Calcularemos PP, FIFO y LIFO usando **Compras netas** y **Ventas netas** por simplicidad periódica.")

        # Cálculos
        res = {}
        for method in ["PP", "PEPS", "UEPS"]:
            cogs, if_q, if_val = aplicar_devoluciones_periodico(
                method, inv_q, inv_c, cmp_q, cmp_c, vta_q, dev_cmp, dev_vta
            )
            res[method] = (cogs, if_q, if_val)

        st.markdown("#### Resultados")
        for m in ["PP", "PEPS", "UEPS"]:
            c, qf, vf = res[m]
            st.write(f"**{m}** → COGS: {peso(c)} · InvF: {qf} u. ({peso(vf)})")

        st.markdown("---")
        comentario = st.text_area("¿Qué efecto observas de las devoluciones en COGS e InvF? (IA te comenta):", key="n3_e_cmt")
        if st.button("💬 Comentar con IA", key="n3_e_fb"):
            prompt = (
                "Explica el efecto de las devoluciones de compra y venta sobre COGS e Inventario Final "
                "bajo PP/FIFO/LIFO en sistema periódico, con el supuesto didáctico de ventas netas y compras netas. "
                f"Datos: InvI={inv_q}u @ {inv_c}, Compras={cmp_q}u @ {cmp_c}, DevCompras={dev_cmp}, Ventas={vta_q}, DevVentas={dev_vta}. "
                f"Resultados: { {k:(float(v[0]), v[1], float(v[2])) for k,v in res.items()} }. "
                f"Comentario del estudiante: {comentario}"
            )
            fb = ia_feedback(prompt, role_desc="corrector")
            st.info(fb)

    # PRÁCTICA
    with tabs[2]:
        st.subheader("Práctica: método + devoluciones")

        def n3_new_case():
            inv_q = random.randint(40, 160)
            inv_c = random.choice([8, 9, 10, 11, 12])
            cmp_q = random.randint(50, 180)
            cmp_c = random.choice([inv_c - 1, inv_c, inv_c + 1, inv_c + 2])
            cmp_c = max(1, float(cmp_c))
            dev_cmp = random.randint(0, max(0, cmp_q // 4))
            vta_q = random.randint(20, inv_q + cmp_q)
            dev_vta = random.randint(0, max(0, vta_q // 6))
            st.session_state.n3p = dict(inv_q=inv_q, inv_c=float(inv_c),
                                        cmp_q=cmp_q, cmp_c=float(cmp_c),
                                        dev_cmp=dev_cmp, vta_q=vta_q, dev_vta=dev_vta)

        if "n3p" not in st.session_state:
            n3_new_case()

        meta = st.session_state.n3p
        c0, c1, c2, c3, c4, c5 = st.columns(6)
        c0.metric("InvI (u.)", meta["inv_q"])
        c1.metric("InvI ($/u)", peso(meta["inv_c"]))
        c2.metric("Compras (u.)", meta["cmp_q"])
        c3.metric("Dev. Compras (u.)", meta["dev_cmp"])
        c4.metric("Ventas (u.)", meta["vta_q"])
        c5.metric("Dev. Ventas (u.)", meta["dev_vta"])

        st.button("🔄 Nuevo caso", on_click=n3_new_case, key="n3_p_new")

        metodo = st.selectbox("Método", ["PP", "PEPS", "UEPS"], key="n3_p_m")
        user_cogs = st.number_input("Tu COGS ($)", min_value=0.0, value=0.0, step=10.0, key="n3_p_cogs")
        user_invf = st.number_input("Tu InvF ($)", min_value=0.0, value=0.0, step=10.0, key="n3_p_invf")
        explic = st.text_area("Explica tu procedimiento (IA te corrige):", key="n3_p_explic")

        if st.button("✅ Validar práctica", key="n3_p_val"):
            inv_q, inv_c = meta["inv_q"], meta["inv_c"]
            cmp_q, cmp_c = meta["cmp_q"], meta["cmp_c"]
            dev_c, vta_q, dev_v = meta["dev_cmp"], meta["vta_q"], meta["dev_vta"]

            cogs, if_q, if_val = aplicar_devoluciones_periodico(
                metodo, inv_q, inv_c, cmp_q, cmp_c, vta_q, dev_c, dev_v
            )

            ok_cogs = abs(user_cogs - cogs) <= 0.5
            ok_invf = abs(user_invf - if_val) <= 0.5

            if ok_cogs and ok_invf:
                st.success(f"¡Perfecto! COGS={peso(cogs)} · InvF={peso(if_val)}")
            elif ok_cogs:
                st.warning(f"COGS correcto ({peso(cogs)}), pero InvF esperado era {peso(if_val)}")
            elif ok_invf:
                st.warning(f"InvF correcto ({peso(if_val)}), pero COGS esperado era {peso(cogs)}")
            else:
                st.error(f"No coincide. COGS esperado {peso(cogs)} · InvF esperado {peso(if_val)}")

            prompt = (
                f"Corrige el cálculo del estudiante con devoluciones usando {metodo}. "
                f"Datos: InvI={inv_q}u @ {inv_c}, Compras={cmp_q}u @ {cmp_c}, DevCompras={dev_c}, Ventas={vta_q}u, DevVentas={dev_v}. "
                f"Respuesta estudiante: COGS={user_cogs:.2f}, InvF={user_invf:.2f}. "
                f"Correcto: COGS={cogs:.2f}, InvF={if_val:.2f}. Explicación del estudiante: {explic}"
            )
            fb = ia_feedback(prompt, role_desc="tutor")
            with st.expander("💬 Feedback de la IA"):
                st.write(fb)

    # EVALUACIÓN
    with tabs[3]:
        st.subheader("Evaluación final del Nivel 3")
        st.caption("Apruebas con **2 de 3**.")

        q1 = st.radio(
            "1) En sistema periódico, las **devoluciones de compra** suelen presentarse como:",
            ["Aumento de compras", "Disminución de compras (compras netas)"],
            index=None, key="n3_q1"
        )
        q2 = st.radio(
            "2) Para modelar el **costo** de devoluciones de venta en este curso, asumimos que:",
            ["Se registra como mayor COGS sin afectar ventas",
             "Es como si se hubieran vendido menos unidades (ventas netas)"],
            index=None, key="n3_q2"
        )
        q3 = st.radio(
            "3) Si hay muchas devoluciones de venta al final del período, el COGS resultante (bajo esta simplificación) tenderá a:",
            ["Aumentar", "Disminuir"],
            index=None, key="n3_q3"
        )

        if st.button("🧪 Validar evaluación (Nivel 3)", key="n3_eval_btn"):
            correct = {
                "n3_q1": "Disminución de compras (compras netas)",
                "n3_q2": "Es como si se hubieran vendido menos unidades (ventas netas)",
                "n3_q3": "Disminuir",
            }
            answers = {"n3_q1": q1, "n3_q2": q2, "n3_q3": q3}
            score = sum(1 for k,v in answers.items() if v == correct[k])
            passed = score >= 2

            prompt = (
                f"Nivel 3 evaluación. Respuestas estudiante: {answers}. Correctas: {correct}. "
                f"Aciertos: {score}/3. Escribe feedback breve (≤6 líneas) y una recomendación."
            )
            fb = ia_feedback(prompt, role_desc="coach")

            if passed:
                st.success(f"¡Aprobado! Aciertos {score}/3 🎉 Desbloquearás el Nivel 4.")
                celebracion_confeti()
            else:
                st.error(f"No aprobado. Aciertos {score}/3. Repasa y vuelve a intentar.")
            with st.expander("💬 Feedback de la IA"):
                st.write(fb)

            save_progress(username, "level3", passed, score=score)

            if passed:
                st.session_state["force_go_level4"] = True
                st.rerun()

# ===========================
# NIVEL 4 · Estado de Resultados
# ===========================
def page_level4(username):
    st.title("Nivel 4 · Construcción del Estado de Resultados (básico)")

    tabs = st.tabs(["🎧 Teoría", "🛠 Ejemplo guiado", "🎮 Práctica con IA", "🏁 Evaluación final & Encuesta"])

    # TEORÍA
    with tabs[0]:
        teoria = (
            "Un **Estado de Resultados** (formato simple) resume ingresos y costos/gastos del período:\n\n"
            "1) **Ventas netas** = Ventas brutas − Devoluciones de venta (y descuentos, si aplica)\n"
            "2) **COGS (Costo de Ventas)** → viene de la valoración de inventarios\n"
            "3) **Utilidad bruta** = Ventas netas − COGS\n"
            "4) **Gastos de operación** (adm., ventas, etc.)\n"
            "5) **Utilidad operativa** = Utilidad bruta − Gastos operativos\n"
            "6) **Impuestos** (p.ej., % sobre utilidad operativa positiva)\n"
            "7) **Utilidad neta** = Utilidad operativa − Impuestos\n\n"
            "En este nivel armarás el ER a partir de COGS y devoluciones vistas en los niveles anteriores."
        )
        st.write(teoria)
        speak_block(teoria, key_prefix="teo-n4", lang_hint="es")

    # EJEMPLO
    with tabs[1]:
        st.subheader("Ejemplo guiado")
        c1, c2 = st.columns(2)
        with c1:
            ventas = st.number_input("Ventas brutas ($)", min_value=0.0, value=20000.0, step=500.0, key="n4_e_vtas")
            dev_venta = st.number_input("Devoluciones de venta ($)", min_value=0.0, value=500.0, step=100.0, key="n4_e_dev_v")
            cogs = st.number_input("COGS ($)", min_value=0.0, value=12000.0, step=500.0, key="n4_e_cogs")
            gastos = st.number_input("Gastos operativos ($)", min_value=0.0, value=3000.0, step=200.0, key="n4_e_gop")
            tasa = st.number_input("Tasa de impuestos (%)", min_value=0.0, value=30.0, step=1.0, key="n4_e_tax")
        with c2:
            vnet = max(ventas - dev_venta, 0.0)
            ub = vnet - cogs
            uop = ub - gastos
            imp = max(uop, 0.0) * (tasa/100.0)
            uneta = uop - imp

            st.markdown("#### Estado de Resultados")
            st.write(f"Ventas netas: **{peso(vnet)}**")
            st.write(f"COGS: **{peso(cogs)}**")
            st.write(f"Utilidad bruta: **{peso(ub)}**")
            st.write(f"Gastos operativos: **{peso(gastos)}**")
            st.write(f"Utilidad operativa: **{peso(uop)}**")
            st.write(f"Impuestos ({tasa:.0f}%): **{peso(imp)}**")
            st.success(f"Utilidad neta: **{peso(uneta)}**")

        st.markdown("---")
        comentario = st.text_area("Comenta el efecto de subir o bajar el COGS en la utilidad (IA responde):", key="n4_e_cmt")
        if st.button("💬 Comentar con IA", key="n4_e_fb"):
            prompt = (
                "Explica brevemente cómo impacta el COGS en utilidad bruta, operativa y neta. "
                f"Datos ejemplo: Ventas={ventas}, DevVentas={dev_venta}, COGS={cogs}, Gastos={gastos}, Tasa={tasa}. "
                f"Comentario: {comentario}"
            )
            fb = ia_feedback(prompt, role_desc="tutor")
            st.info(fb)

    # PRÁCTICA
    with tabs[2]:
        st.subheader("Práctica guiada (con IA)")
        def n4_new_case():
            ventas = random.randint(12000, 35000)
            dev_v = random.randint(0, 1500)
            cogs = random.randint(7000, 24000)
            gastos = random.randint(1500, 6000)
            tasa = random.choice([25, 30, 33])
            st.session_state.n4p = dict(ventas=ventas, dev_v=dev_v, cogs=cogs, gastos=gastos, tasa=tasa)

        if "n4p" not in st.session_state:
            n4_new_case()

        meta = st.session_state.n4p
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Ventas brutas", peso(meta["ventas"]))
        m2.metric("Dev. venta", peso(meta["dev_v"]))
        m3.metric("COGS", peso(meta["cogs"]))
        m4.metric("Gastos Op.", peso(meta["gastos"]))
        m5.metric("Tasa imp. (%)", meta["tasa"])

        st.button("🔄 Nuevo caso", on_click=n4_new_case, key="n4_p_new")

        user_vnet = st.number_input("Tus Ventas netas ($)", min_value=0.0, value=0.0, step=100.0, key="n4_vnet")
        user_ub = st.number_input("Tu Utilidad bruta ($)", min_value=0.0, value=0.0, step=100.0, key="n4_ub")
        user_uneta = st.number_input("Tu Utilidad neta ($)", min_value=-100000.0, value=0.0, step=100.0, key="n4_uneta")
        explic = st.text_area("Explica tu procedimiento (IA te corrige):", key="n4_explic")

        if st.button("✅ Validar práctica", key="n4_p_val"):
            ventas, dev_v, cogs, gastos, tasa = meta["ventas"], meta["dev_v"], meta["cogs"], meta["gastos"], meta["tasa"]
            vnet = max(ventas - dev_v, 0.0)
            ub = vnet - cogs
            uop = ub - gastos
            imp = max(uop, 0.0) * (tasa/100.0)
            uneta = uop - imp

            ok_vnet = abs(user_vnet - vnet) <= 0.5
            ok_ub = abs(user_ub - ub) <= 0.5
            ok_uneta = abs(user_uneta - uneta) <= 0.5

            if ok_vnet and ok_ub and ok_uneta:
                st.success("¡Perfecto! Coinciden Ventas netas, Utilidad bruta y Utilidad neta.")
            else:
                if not ok_vnet:
                    st.error(f"Ventas netas esperadas: {peso(vnet)}")
                if not ok_ub:
                    st.error(f"Utilidad bruta esperada: {peso(ub)}")
                if not ok_uneta:
                    st.error(f"Utilidad neta esperada: {peso(uneta)}")

            prompt = (
                f"Corrige el armado del ER del estudiante. Datos: Ventas={ventas}, DevVentas={dev_v}, "
                f"COGS={cogs}, Gastos={gastos}, Tasa={tasa}. "
                f"Respuestas estudiante: VentasNetas={user_vnet:.2f}, UtilidadBruta={user_ub:.2f}, UtilidadNeta={user_uneta:.2f}. "
                f"Correcto: VentasNetas={vnet:.2f}, UtilidadBruta={ub:.2f}, UtilidadNeta={uneta:.2f}. "
                f"Explicación del estudiante: {explic}"
            )
            fb = ia_feedback(prompt, role_desc="corrector")
            with st.expander("💬 Feedback de la IA"):
                st.write(fb)

    # EVALUACIÓN FINAL + ENCUESTA
    with tabs[3]:
        st.subheader("Evaluación final del Nivel 4")
        st.caption("Apruebas con **2 de 3** y habilitas la encuesta final.")

        q1 = st.radio(
            "1) ¿Cómo se calcula la **utilidad bruta**?",
            ["Ventas netas − COGS", "Ventas brutas − Gastos operativos"],
            index=None, key="n4_q1"
        )
        q2 = st.radio(
            "2) La **utilidad operativa** es:",
            ["Utilidad bruta − Gastos operativos", "Ventas netas − (COGS + Impuestos)"],
            index=None, key="n4_q2"
        )
        q3 = st.radio(
            "3) Si la utilidad operativa es negativa, los **impuestos** en este modelo:",
            ["Se calculan igual (siempre positivos)", "No se calculan (se toma 0)"],
            index=None, key="n4_q3"
        )

        if st.button("🧪 Validar evaluación (Nivel 4)", key="n4_eval_btn"):
            correct = {
                "n4_q1": "Ventas netas − COGS",
                "n4_q2": "Utilidad bruta − Gastos operativos",
                "n4_q3": "No se calculan (se toma 0)",
            }
            answers = {"n4_q1": q1, "n4_q2": q2, "n4_q3": q3}
            score = sum(1 for k,v in answers.items() if v == correct[k])
            passed = score >= 2

            prompt = (
                f"Nivel 4 evaluación. Respuestas estudiante: {answers}. Correctas: {correct}. "
                f"Aciertos: {score}/3. Feedback breve (≤6 líneas) y una recomendación final."
            )
            fb = ia_feedback(prompt, role_desc="coach")

            if passed:
                st.success(f"¡Aprobado! Aciertos {score}/3 🎉 Has completado los 4 niveles.")
                celebracion_confeti()
            else:
                st.error(f"No aprobado. Aciertos {score}/3. Ajusta y vuelve a intentar.")
            with st.expander("💬 Feedback de la IA"):
                st.write(fb)

            save_progress(username, "level4", passed, score=score)

        st.markdown("---")
        prog = get_progress(username)
        if prog["level1"]["passed"] and prog["level2"]["passed"] and prog["level3"]["passed"] and prog["level4"]["passed"]:
            st.success("🎯 ¡Completaste los 4 niveles! Por favor responde la encuesta final.")
            st.markdown(f"[📝 Abrir encuesta]({SURVEY_URL})", unsafe_allow_html=True)
            done = st.checkbox("✅ Marcar encuesta como completada", value=prog.get("completed_survey", False), key="survey_done")
            if done and not prog.get("completed_survey", False):
                prog["completed_survey"] = True
                st.session_state.all_progress[username] = prog
                st.toast("¡Gracias por completar la encuesta!", icon="🎉")

# ===========================
# Pantalla Login
# ===========================
def login_screen():
    st.header("Iniciar Sesión")
    with st.form("login_form"):
        st.text_input("Usuario", key="login_raw_user")
        st.text_input("Contraseña", type="password", key="login_password")
        st.form_submit_button("Ingresar", on_click=do_login)
    if st.session_state.login_error:
        st.error(st.session_state.login_error)
    st.markdown("---")
    st.caption("Usuarios demo → **admin / AdminSeguro#2025** · **estudiante / 1234**")

# ===========================
# Router principal
# ===========================
def main_app():
    username = st.session_state.username
    current = sidebar_nav(username)

    if current.startswith("Nivel 1"):
        page_level1(username)
    elif current.startswith("Nivel 2"):
        page_level2(username)
    elif current.startswith("Nivel 3"):
        page_level3(username)
    elif current.startswith("Nivel 4"):
        page_level4(username)
    else:
        page_level1(username)

# ===========================
# Entry
# ===========================
def main():
    init_session()
    if not st.session_state.authenticated:
        login_screen()
    else:
        main_app()

if __name__ == "__main__":
    main()
