# -*- coding: utf-8 -*-
# =========================================================
#   Herramienta Contable - Inventarios Gamificados (sin Mongo)
#   Niveles por página · Nivel 1 con TTS y práctica + feedback IA
#   Auto-desbloqueo y salto a Nivel 2 al aprobar Nivel 1
#   OpenRouter + DeepSeek (v3.1:free) con headers recomendados
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

    # >>> Auto-selección de Nivel 2 al aprobar Nivel 1
    if prog["level1"]["passed"] and st.session_state.get("force_go_level2"):
        st.session_state["sidebar_level_select"] = "Nivel 2: Métodos (PP/PEPS/UEPS)"
        st.session_state["force_go_level2"] = False

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

    return sel

# ===========================
# NIVEL 1 (con retroalimentaciones IA)
# ===========================
def page_level1(username):
    st.title("Nivel 1 · Introducción a la valoración de inventarios")

    tabs = st.tabs(["🎧 Teoría profunda", "🛠 Ejemplo guiado", "🎮 Práctica interactiva (IA)", "🏁 Evaluación para aprobar"])

    # ----- TEORÍA PROFUNDA -----
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

    # ----- EJEMPLO GUIADO -----
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

    # ----- PRÁCTICA INTERACTIVA (IA) -----
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

    # ----- EVALUACIÓN PARA APROBAR -----
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
                f"Aciertos: {score}/3. Da un feedback amable en máximo 6 líneas y sugiere 1 repaso si falló."
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

            # >>> Fuerza desbloqueo visible y salto a Nivel 2
            if passed:
                st.session_state["force_go_level2"] = True
                st.rerun()

# ===========================
# Placeholders niveles 2–4
# ===========================
def page_placeholder(title):
    st.title(title)
    st.info("Este nivel se habilitará y lo mejoraremos después de que termines el nivel anterior. 👍")

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
        page_placeholder("Nivel 2 · Métodos (Promedio Ponderado, PEPS, UEPS)")
    elif current.startswith("Nivel 3"):
        page_placeholder("Nivel 3 · Devoluciones")
    elif current.startswith("Nivel 4"):
        page_placeholder("Nivel 4 · Estado de Resultados")
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
