# -*- coding: utf-8 -*-
# =========================================================
#   Herramienta Contable - Inventarios Gamificados (con Mongo)
#   Optimizada: cache de Mongo, formularios (sin reruns), IA opcional,
#   escenarios aleatorios estables, confeti/TTS livianos,
#   Admin con CRUD + Estadísticas (aggregations)
#   Fecha: 2025-10-08
# =========================================================

import os
import random
import ssl
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

# ===========================
# Constantes
# ===========================
ADMIN_OPTION = "⚙️ Administrador de Usuarios"

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

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)
DEEPSEEK_MODEL = "deepseek/deepseek-chat-v3.1:free"

def ia_feedback(prompt_user: str) -> str:
    """
    Usa OpenRouter con el modelo DeepSeek para dar feedback educativo breve.
    Si no hay API key o hay fallo de red/modelo, devuelve mensaje local.
    """
    if not OPENROUTER_API_KEY:
        return "Feedback IA no disponible. Tus resultados se validaron localmente."
    try:
        completion = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Eres un tutor de contabilidad empático y claro. "
                        "Explica en máximo 6 líneas el acierto/error del estudiante, "
                        "resalta la fórmula clave o el concepto y ofrece 1 truco memotécnico."
                    )
                },
                {"role": "user", "content": prompt_user}
            ],
            temperature=0.3,
            extra_body={}
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        return f"No pude generar feedback con IA ahora. ({e})"

def n1_eval_open_ai(respuesta_estudiante: str) -> tuple[bool, str, str]:
    """
    Valida la respuesta abierta con IA.
    Devuelve (aprobado_bool, feedback_corto, retroalimentacion_formativa).
    Si la IA no está disponible, usa una heurística local simple.
    """
    texto = (respuesta_estudiante or "").strip().lower()

    # Heurística local mínima (fallback)
    def fallback_check(t: str) -> bool:
        return ("aumenta" in t) and ("inventario final" in t or "fórmula" in t or "se resta menos" in t)

    if not OPENROUTER_API_KEY:
        ok = fallback_check(texto)
        fb = "Aprobado por validación local." if ok else "No aprobado por validación local."
        retro = (
            "Recuerda que cuando el inventario final disminuye, se resta menos en la fórmula, "
            "y por eso el costo de la mercancía vendida aumenta."
            if not ok else
            "Excelente, identificaste correctamente la relación entre inventario final y costo."
        )
        return ok, fb, retro

    # Prompt a la IA
    prompt = (
        "Evalúa la respuesta del estudiante sobre la relación entre el inventario final y el costo de la mercancía vendida. "
        "Primero, indica si está correcta o no. Luego, ofrece retroalimentación formativa breve. "
        "Responde SOLO en este formato JSON:\n\n"
        "{\n"
        "  \"aprobado\": true o false,\n"
        "  \"comentario_corto\": \"...\",\n"
        "  \"retroalimentacion\": \"...\"\n"
        "}\n\n"
        "Criterio: debe decir que al disminuir el inventario final, el costo de la mercancía vendida aumenta, "
        "y explicar brevemente por qué.\n\n"
        f"Respuesta del estudiante: \"{respuesta_estudiante}\""
    )

    try:
        fb = ia_feedback(prompt)
        fb_json = fb.strip()

        # Intentar leer JSON devuelto por la IA
        import json
        try:
            data = json.loads(fb_json)
            ok = bool(data.get("aprobado"))
            corto = data.get("comentario_corto", "").strip()
            retro = data.get("retroalimentacion", "").strip()
            return ok, corto, retro
        except Exception:
            # Si no devolvió JSON, procesar texto plano
            aprobado = fb_json.lower().startswith("aprobado")
            return aprobado, fb_json, "No se pudo leer retroalimentación estructurada."

    except Exception as e:
        ok = fallback_check(texto)
        fb = f"Error IA ({e}). Validación local aplicada."
        retro = (
            "Cuando el inventario final baja, el costo de la mercancía vendida sube, "
            "porque se resta menos al total de costos disponibles."
        )
        return ok, fb, retro

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

def speak_block(texto: str, key_prefix: str, lang_hint="es"):
    """
    Control TTS del navegador con selector de voz + velocidad + tono.
    (Web Speech API del navegador) — Montaje perezoso vía expander.
    """
    escaped = (
        texto.replace("\\", "\\\\")
             .replace("`", "\\`")
             .replace("\n", "\\n")
             .replace('"', '\\"')
    )
    html = f"""
    <div style="padding:8px;border:1px solid #eee;border-radius:10px;margin-bottom:8px;">
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
      <small>Tip: prueba voces como <em>Google español</em> o <em>Microsoft Sabina</em>. Algunas respetan mejor velocidad y tono.</small>
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
          const sorted = voices.slice().sort((a,b)=>score(b)-score(a));
          sorted.forEach((v, i) => {{
            const opt = document.createElement("option");
            opt.value = voices.indexOf(v);
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
            const idx = parseInt(sel.value, 10);
            const u = new SpeechSynthesisUtterance(text);
            if (!isNaN(idx) && voices[idx]) {{
              u.voice = voices[idx];
            }}
            u.rate = parseFloat(rate.value);
            u.pitch = parseFloat(pitch.value);
            speechSynthesis.speak(u);
          }} catch (e) {{}}
        }};
        btnStop.onclick = () => speechSynthesis.cancel();
      }})();
    </script>
    """
    components.html(html, height=140)

# ===========================
# Config encuesta
# ===========================
SURVEY_URL = os.getenv("SURVEY_URL", "https://forms.gle/pSxXp78LR3gqRzeR6")

# ===========================
# Pantalla de Celebración (aparte) — liviana
# ===========================
def confetti_block(duration_ms: int = 3500, height_px: int = 320):
    """
    Confeti y 'globos' simples 100% inline (sin CDNs) — parámetros reducidos.
    """
    try:
        st.balloons()
    except Exception:
        pass

    html = f"""
    <div id="confetti-wrapper" style="position:relative;width:100%;height:{height_px-10}px;overflow:hidden;border-radius:12px;border:1px solid #eee;background:transparent;">
      <canvas id="confetti-canvas" style="position:absolute;inset:0;width:100%;height:100%;"></canvas>
    </div>
    <script>
    (function() {{
      const wrapper = document.getElementById('confetti-wrapper');
      const canvas = document.getElementById('confetti-canvas');
      const ctx = canvas.getContext('2d');

      function resize() {{
        const r = wrapper.getBoundingClientRect();
        canvas.width = Math.max(200, r.width);
        canvas.height = Math.max(120, r.height);
      }}
      resize();
      if (typeof ResizeObserver !== 'undefined') {{
        new ResizeObserver(resize).observe(wrapper);
      }} else {{
        window.addEventListener('resize', resize);
      }}

      const colors = ['#ff6b6b','#ffd93d','#6BCB77','#4D96FF','#845EC2','#FF9671','#FFC75F'];
      const rand = (a,b)=>a+Math.random()*(b-a);
      const pick = (arr)=>arr[Math.floor(Math.random()*arr.length)];

      const pieces = [];
      const N = 90;  // antes 180
      for (let i=0;i<N;i++) {{
        pieces.push({{
          type: Math.random()<0.4 ? 'tri' : 'rect',
          x: Math.random()*canvas.width,
          y: rand(-canvas.height, 0),
          w: rand(6, 12),
          h: rand(8, 18),
          r: rand(0, Math.PI*2),
          vr: rand(-0.1, 0.1),
          vx: rand(-0.6, 0.6),
          vy: rand(1.8, 3.2),
          color: pick(colors),
          alpha: rand(0.85, 1)
        }});
      }}

      const balloons = [];
      for (let i=0;i<4;i++) {{
        balloons.push({{
          x: Math.random()*canvas.width,
          y: canvas.height + rand(20, 120),
          r: rand(14, 22),
          vy: rand(0.4, 0.8),
          color: pick(colors)
        }});
      }}

      function burst(x, y, count=22) {{
        for (let i=0;i<count;i++) {{
          pieces.push({{
            type: Math.random()<0.5 ? 'tri' : 'rect',
            x, y,
            w: rand(5, 10),
            h: rand(6, 14),
            r: rand(0, Math.PI*2),
            vr: rand(-0.2, 0.2),
            vx: rand(-3, 3),
            vy: rand(-3, 1),
            color: pick(colors),
            alpha: 1
          }});
        }}
      }}
      burst(canvas.width*0.5, canvas.height*0.3);
      burst(canvas.width*0.2, canvas.height*0.2);
      burst(canvas.width*0.8, canvas.height*0.25);

      const start = performance.now();
      (function draw(now){{
        const elapsed = now - start;
        ctx.clearRect(0,0,canvas.width,canvas.height);

        for (const p of pieces) {{
          p.x += p.vx + Math.sin(p.y*0.02)*0.2;
          p.y += p.vy;
          p.r += p.vr;

          if (p.y > canvas.height + 20) {{
            p.y = -20;
            p.x = Math.random()*canvas.width;
            p.vx = rand(-0.6, 0.6);
            p.vy = rand(1.8, 3.2);
            p.r = rand(0, Math.PI*2);
            p.color = pick(colors);
            p.alpha = rand(0.85, 1);
          }}

          ctx.save();
          ctx.globalAlpha = p.alpha;
          ctx.translate(p.x, p.y);
          ctx.rotate(p.r);
          ctx.fillStyle = p.color;

          if (p.type === 'rect') {{
            ctx.fillRect(-p.w/2, -p.h/2, p.w, p.h);
          }} else {{
            ctx.beginPath();
            ctx.moveTo(0, -p.h/2);
            ctx.lineTo(-p.w/2, p.h/2);
            ctx.lineTo(p.w/2, p.h/2);
            ctx.closePath();
            ctx.fill();
          }}
          ctx.restore();
        }}

        for (const b of balloons) {{
          b.y -= b.vy;
          if (b.y + b.r < -30) {{
            b.y = canvas.height + rand(30, 120);
            b.x = Math.random()*canvas.width;
            b.vy = rand(0.4, 0.8);
            b.color = pick(colors);
          }}
          ctx.beginPath();
          ctx.fillStyle = b.color;
          ctx.arc(b.x, b.y, b.r, 0, Math.PI*2);
          ctx.fill();
          ctx.beginPath();
          ctx.strokeStyle = '#888';
          ctx.moveTo(b.x, b.y + b.r);
          ctx.lineTo(b.x, b.y + b.r + 26);
          ctx.stroke();
        }}

        if (elapsed < {duration_ms}) {{
          requestAnimationFrame(draw);
        }}
      }})(performance.now());
    }})();
    </script>
    """
    components.html(html, height=height_px)

def start_celebration(message_md: str, next_label: str, next_key_value: str):
    st.session_state["celebrate_active"] = True
    st.session_state["celebrate_message"] = message_md
    st.session_state["celebrate_next_label"] = next_label
    st.session_state["celebrate_next_key"] = next_key_value
    st.rerun()

def celebration_screen():
    if not st.session_state.get("celebrate_active"):
        return False

    st.markdown("# 🎉 ¡Lo lograste!")
    confetti_block(duration_ms=3500, height_px=320)

    msg = st.session_state.get("celebrate_message", "¡Felicidades!")
    st.markdown(
        f"""
        <div style="margin-top:10px;margin-bottom:16px;padding:16px;border:1px solid #eee;border-radius:12px;background:#fffaf0">
          <div style="font-size:1.1rem;line-height:1.6">{msg}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        label = st.session_state.get("celebrate_next_label", "siguiente nivel")
        if st.button(f"➡️ Ir al {label}", key="celebrate_go_next_btn", use_container_width=True):
            next_key = st.session_state.get("celebrate_next_key")
            if next_key:
                st.session_state["sidebar_level_select"] = next_key
            st.session_state["celebrate_active"] = False
            st.session_state["celebrate_message"] = ""
            st.session_state["celebrate_next_label"] = ""
            st.session_state["celebrate_next_key"] = ""
            st.rerun()
    return True

# ===========================
# --- REPO / MONGO HELPERS ---
# ===========================
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from pymongo import WriteConcern
import certifi
import hashlib

# Fuerza el bundle de certificados de certifi
os.environ.setdefault("SSL_CERT_FILE", certifi.where())

# Hashing simple
def hash_password(p: str) -> str:
    return hashlib.sha256(("pepper123::" + p).encode("utf-8")).hexdigest()

def verify_password(p: str, h: str) -> bool:
    return hash_password(p) == h

def _connect_mongo(uri: str, insecure: bool = False):
    """
    Crea un MongoClient con TLS y CA de certifi.
    Si insecure=True, permite certificados inválidos (solo desarrollo).
    """
    kwargs = dict(
        server_api=ServerApi('1'),
        tls=True,
        tlsCAFile=certifi.where(),
        socketTimeoutMS=30000,
        connectTimeoutMS=30000,
        retryReads=True,
        retryWrites=True,
    )
    if insecure:
        kwargs["tlsAllowInvalidCertificates"] = True
    return MongoClient(uri, **kwargs)

# ---------- Versión cacheada de la conexión ----------
@st.cache_resource(show_spinner=False)
def repo_init_cached(uri: str, admin_user: str, admin_pass: str):
    try:
        client = _connect_mongo(uri, insecure=False)
        client.admin.command('ping')
    except Exception:
        client = _connect_mongo(uri, insecure=True)
        client.admin.command('ping')

    db = client["accounting_app"]
    users_col    = db["users"]
    progress_col = db["progress"]
    # attempts con write concern w=0 para evitar bloqueo de UI
    attempts_col = db.get_collection("attempts").with_options(write_concern=WriteConcern(w=0))

    if users_col.count_documents({"username": admin_user}) == 0:
        users_col.insert_one({
            "username": admin_user,
            "password_hash": hash_password(admin_pass),
            "role": "admin",
            "created_at": datetime.now(timezone.utc)
        })

    try:
        users_col.create_index("username", unique=True)
        progress_col.create_index("username", unique=True)
        attempts_col.create_index([("username", 1), ("level", 1), ("created_at", -1)])
    except Exception:
        pass

    return db, users_col, progress_col, attempts_col

def repo_init():
    """
    Crea el cliente Mongo cacheado y retorna (db, users_col, progress_col, attempts_col).
    """
    uri = None
    try:
        uri = st.secrets["mongodb"]["uri"]
    except Exception:
        uri = os.getenv("MONGODB_URI")

    if not uri:
        raise RuntimeError("No encuentro la URI de MongoDB. Define [mongodb].uri en secrets.toml o MONGODB_URI en el entorno.")

    try:
        admin_user = st.secrets["admin"]["username"]
        admin_pass = st.secrets["admin"]["password"]
    except Exception:
        admin_user = "admin"
        admin_pass = "AdminSeguro#2025"

    return repo_init_cached(uri, admin_user, admin_pass)

# --------- PROGRESO (Gamificación) ----------
def _default_progress_doc(username: str) -> dict:
    return {
        "username": username,
        "levels": {
            "level1": {"passed": False, "date": None, "score": None},
            "level2": {"passed": False, "date": None, "score": None},
            "level3": {"passed": False, "date": None, "score": None},
            "level4": {"passed": False, "date": None, "score": None},
        },
        "completed_survey": False,
        "updated_at": datetime.now(timezone.utc),
        "created_at": datetime.now(timezone.utc),
    }

def ensure_progress(progress_col, username: str) -> dict:
    doc = progress_col.find_one({"username": username})
    if doc is None:
        doc = _default_progress_doc(username)
        progress_col.insert_one(doc)
    return doc

def load_progress(progress_col, username: str) -> dict:
    return ensure_progress(progress_col, username)

def set_level_passed(progress_col, username: str, level_key: str, score: int | None):
    now = datetime.now(timezone.utc)
    progress_col.update_one(
        {"username": username},
        {
            "$set": {
                f"levels.{level_key}.passed": True,
                f"levels.{level_key}.date": now,
                f"levels.{level_key}.score": score,
                "updated_at": now,
            }
        },
        upsert=True
    )

def set_completed_survey(progress_col, username: str, value: bool = True):
    now = datetime.now(timezone.utc)
    progress_col.update_one(
        {"username": username},
        {"$set": {"completed_survey": value, "updated_at": now}},
        upsert=True
    )

# --------- ATTEMPTS (Estadísticas) ----------
def record_attempt(username: str, level: int, score: int | None, passed: bool):
    """
    Registra cada validación de evaluación que haga el estudiante.
    level: 1..4
    score: aciertos (p.ej. 0..3)
    passed: True/False
    """
    attempts_col = st.session_state.get("attempts_col")

    # ⬅️ Importante: NO usar la colección en contexto booleano.
    if attempts_col is None or not username:
        return

    try:
        attempts_col.insert_one({
            "username": username,
            "level": int(level),
            "score": int(score) if score is not None else None,
            "passed": bool(passed),
            "created_at": datetime.now(timezone.utc)
        })
    except Exception:
        # No bloquear UI si falla el log
        pass


# --------- USERS (CRUD) ----------
def verify_credentials(users_col, username: str, password: str):
    if users_col is None:
        return None
    doc = users_col.find_one({"username": username})
    if not doc:
        return None
    if verify_password(password, doc.get("password_hash", "")):
        return doc
    return None

def create_user(users_col, progress_col, username: str, password: str, role: str = "user"):
    users_col.insert_one({
        "username": username.strip().lower(),
        "password_hash": hash_password(password),
        "role": role,
        "created_at": datetime.now(timezone.utc)
    })
    if progress_col is not None:
        try:
            progress_col.insert_one(_default_progress_doc(username.strip().lower()))
        except Exception:
            pass

def update_user(users_col, username: str, new_password: str | None, new_role: str | None):
    update = {}
    if new_password:
        update["password_hash"] = hash_password(new_password)
    if new_role:
        update["role"] = new_role
    if update:
        users_col.update_one({"username": username}, {"$set": update})

def delete_user(users_col, progress_col, username: str):
    users_col.delete_one({"username": username})
    if progress_col is not None:
        progress_col.delete_many({"username": username})

# ===========================
# Estado de sesión mínimo
# ===========================
def init_session():
    st.session_state.setdefault("authenticated", False)
    st.session_state.setdefault("login_error", "")
    st.session_state.setdefault("username", "")
    # colecciones se setean en main()

# ===========================
# Login / Logout
# ===========================
def do_login():
    user = st.session_state.login_raw_user.strip().lower()
    pwd  = st.session_state.login_password
    if not user or not pwd:
        st.session_state.login_error = "Por favor, ingresa usuario y contraseña."
        return

    users_col = st.session_state.get("users_col")
    progress_col = st.session_state.get("progress_col")

    doc = verify_credentials(users_col, user, pwd)
    if doc:
        st.session_state.authenticated = True
        st.session_state.username = user
        st.session_state.login_error = ""
        # 🔑 Asegura documento de progreso al iniciar sesión
        ensure_progress(progress_col, user)
    else:
        st.session_state.login_error = "Credenciales incorrectas."

def logout():
    st.session_state.authenticated = False
    st.session_state.username = ""
    st.session_state.login_error = ""

# ===========================
# Sidebar navegación por nivel (Admin solo si rol=admin)
# ===========================
def sidebar_nav(username):
    st.sidebar.title("Niveles")

    users_col = st.session_state.get("users_col")
    progress_col = st.session_state.get("progress_col")
    # rol
    current_user_role = "user"
    if users_col is not None and username:
        doc = users_col.find_one({"username": username}, {"role": 1, "_id": 0}) or {}
        current_user_role = doc.get("role", "user")

    # progreso
    prog = load_progress(progress_col, username)
    lv = prog.get("levels", {})
    l1 = lv.get("level1", {}).get("passed", False)
    l2 = lv.get("level2", {}).get("passed", False)
    l3 = lv.get("level3", {}).get("passed", False)
    l4 = lv.get("level4", {}).get("passed", False)

    # 🚦 Opciones bloqueadas/desbloqueadas
    options = ["Nivel 1: Introducción a Inventarios"]
    if l1:
        options.append("Nivel 2: Métodos (PP/PEPS/UEPS)")
    if l2:
        options.append("Nivel 3: Devoluciones")
    if l3:
        options.append("Nivel 4: Estado de Resultados")
    if l4:
        options.append("Encuesta de satisfacción")
    if current_user_role == "admin":
        options.append(ADMIN_OPTION)

    # --- PRESELECCIÓN SEGURA ANTES DE CREAR EL RADIO ---
    pending = st.session_state.pop("sidebar_next_select", None)
    index_arg = 0
    if pending in options:
        index_arg = options.index(pending)
    else:
        prev = st.session_state.get("sidebar_level_select")
        if prev in options:
            index_arg = options.index(prev)
        else:
            index_arg = 0

    sel = st.sidebar.radio("Ir a:", options, key="sidebar_level_select", index=index_arg)

    st.sidebar.markdown("---")
    def badge(ok): return "✅" if ok else "🔒"
    st.sidebar.caption(f"Usuario: **{username}** · Rol: **{current_user_role}**")
    st.sidebar.write(f"{badge(l1)} Nivel 1")
    st.sidebar.write(f"{badge(l2)} Nivel 2")
    st.sidebar.write(f"{badge(l3)} Nivel 3")
    st.sidebar.write(f"{badge(l4)} Nivel 4")
    st.sidebar.markdown("---")
    st.sidebar.button("Cerrar Sesión", on_click=logout, key="logout_btn")

    return sel

# ===========================
# Helpers de escenarios aleatorios estables
# ===========================
def n1_new_case():
    inv0 = random.randint(500, 4000)
    compras = random.randint(800, 5000)
    devol = random.randint(0, int(compras*0.3))
    invf = random.randint(0, inv0 + compras - devol)
    st.session_state.n1p_inv0 = float(inv0)
    st.session_state.n1p_compras = float(compras)
    st.session_state.n1p_devol = float(devol)
    st.session_state.n1p_invf = float(invf)

def n2_new_case():
    st.session_state.n2_inv0_u  = random.randint(50, 150)
    st.session_state.n2_inv0_pu = random.choice([10.0, 11.0, 12.0])
    st.session_state.n2_comp_u  = random.randint(50, 200)
    st.session_state.n2_comp_pu = random.choice([12.0, 13.0, 14.0])
    st.session_state.n2_venta_u = random.randint(60, st.session_state.n2_inv0_u + st.session_state.n2_comp_u)

def n3_new_case():
    st.session_state.n3_inv0    = random.randint(500, 1500)
    st.session_state.n3_prom0   = random.choice([15.0, 16.0, 17.0])
    st.session_state.n3_comp    = random.randint(500, 2000)
    st.session_state.n3_comp_pu = random.choice([17.0, 18.0, 19.0])
    st.session_state.n3_dev_comp= random.randint(0, int(st.session_state.n3_comp*0.2))
    st.session_state.n3_venta_u = random.randint(200, st.session_state.n3_inv0 + st.session_state.n3_comp)
    st.session_state.n3_dev_v_u = random.randint(0, int(st.session_state.n3_venta_u*0.2))

def n4_new_case():
    st.session_state.n4_ventas   = random.randint(8000, 20000)
    st.session_state.n4_dev_vtas = random.randint(0, 1200)
    st.session_state.n4_cogs     = random.randint(4000, 12000)
    st.session_state.n4_gastos   = random.randint(1000, 5000)

# ===========================
# NIVEL 1
# ===========================
def page_level1(username):
    st.title("Nivel 1 · Introducción a la valoración de inventarios")

    tabs = st.tabs(["🎧 Teoría profunda", "🛠 Ejemplo guiado", "🎮 Práctica interactiva (IA)", "🏁 Evaluación para aprobar"])

    # Teoría
    with tabs[0]:
        st.subheader("¿Qué es valorar inventarios y por qué impacta tu utilidad?")
        teoria = (
            "Valorar inventarios significa asignar un **costo monetario** a las unidades que una empresa mantiene "
            "para vender. Este valor aparece como **activo** en el balance y, al mismo tiempo, determina el "
            "**costo de la mercancía vendida** en el estado de resultados, por lo que impacta directamente la utilidad.\n\n"
            "En un **sistema periódico**, los movimientos del inventario no se registran con detalle en cada venta; "
            "durante el período se acumulan las compras y los ajustes, y al cierre se calcula el costo de la mercancía vendida "
            "con la fórmula base:\n\n"
            "  **Costo de la mercancía vendida = Inventario inicial + Compras − Devoluciones en compras − Inventario final**\n\n"
            "• **Inventario inicial**: el costo de las unidades disponibles al comenzar el período.\n"
            "• **Compras**: el costo de las unidades adquiridas durante el período (incluye desembolsos necesarios para dejarlas disponibles para la venta).\n"
            "• **Devoluciones en compras**: restan las compras cuando se regresan unidades al proveedor.\n"
            "• **Inventario final**: el costo de las unidades que permanecen al cierre; su valoración depende del método (promedio ponderado, primeras en entrar primeras en salir, últimas en entrar primeras en salir) que estudiarás en el siguiente nivel.\n\n"
            "💡 **Analogía de la mochila de costos**: imagina que cargas una mochila donde ingresan el inventario inicial y las compras. "
            "Si devuelves mercancía al proveedor, sacas parte de esa mochila. Al finalizar, miras lo que queda adentro (inventario final). "
            "Lo que salió para atender las ventas del período es, precisamente, el **costo de la mercancía vendida**."
        )
        st.write(teoria)
        with st.expander("🔊 Escuchar explicación"):
            speak_block(teoria, key_prefix="teo-n1", lang_hint="es")

        with st.expander("📌 Nota contable/NIIF"):
            st.markdown(
                "Bajo NIIF, debes usar un método de costo razonable y **consistente**. "
                "UEPS no es aceptado por NIIF plenas (se usa aquí con fines educativos/comparativos)."
            )

    # Ejemplo guiado
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
            st.markdown(
                f"""
                <div style='line-height:1.8; font-size:1.05rem;'>
                    <b>1)</b> Inventario inicial + Compras → {peso(inv0)} + {peso(compras)} = <b>{peso(inv0+compras)}</b><br>
                    <b>2)</b> Menos devoluciones en compras → {peso(inv0+compras)} − {peso(devol)} = <b>{peso(inv0+compras-devol)}</b><br>
                    <b>3)</b> Menos inventario final → {peso(inv0+compras-devol)} − {peso(invf)} = <b>{peso(inv0+compras-devol-invf)}</b>
                </div>
                """,
                unsafe_allow_html=True
            )
            cogs = inv0 + compras - devol - invf
            st.success(f"**Costo de la mercancía vendida** = {peso(cogs)}")
            st.caption(
                "Interpretación: en la ‘mochila de costos’ entran el inventario inicial y las compras. "
                "Las devoluciones en compras restan costo disponible. Al final, el inventario que queda en la mochila "
                "(inventario final) **no** corresponde a ventas. La diferencia es el **costo de la mercancía vendida**."
            )

        st.write("**Mini reto**: explica qué ocurriría con el costo de la mercancía vendida si **no hubiera devoluciones en compras** y el **inventario final fuera muy pequeño**.")
        razonamiento = st.text_area("Tu razonamiento (opcional, la inteligencia artificial te comenta):", key="n1_ex_raz")

        # IA opcional
        ask_ai = st.checkbox("💬 Pedir feedback de IA (opcional)", key="n1_ex_ai", value=False)
        if st.button("💬 Comentar", key="n1_ex_fb"):
            if ask_ai:
                with st.spinner("Generando feedback con IA..."):
                    prompt = (
                        "Evalúa si el razonamiento es coherente con COGS = InvI + Compras - Devoluciones - InvF. "
                        f"Datos: InvI={inv0}, Compras={compras}, Devol={devol}, InvF={invf}. "
                        f"Texto del estudiante: {razonamiento}"
                    )
                    fb = ia_feedback(prompt)
                with st.expander("💬 Feedback de la IA"):
                    st.write(fb)
        else:
            st.info(
                "Validación local: si disminuye el inventario final, el costo de la mercancía vendida aumenta; "
                "si no hay devoluciones en compras, el valor de las compras no se reduce."
            )

    # Práctica interactiva (IA) — escenarios estables
    with tabs[2]:
        st.subheader("Práctica interactiva")
        st.caption("Completa el cálculo. Puedes generar otro escenario y validar (IA opcional).")

        if "n1p_inv0" not in st.session_state:
            n1_new_case()

        cols = st.columns(4)
        with cols[0]:
            st.metric("Inventario Inicial", peso(st.session_state.n1p_inv0))
        with cols[1]:
            st.metric("Compras", peso(st.session_state.n1p_compras))
        with cols[2]:
            st.metric("Devoluciones", peso(st.session_state.n1p_devol))
        with cols[3]:
            st.metric("Inventario Final", peso(st.session_state.n1p_invf))

        st.button("🔄 Nuevo escenario", on_click=n1_new_case, key="n1_practice_new")

        with st.form("n1_practice_form"):
            user_cogs = st.number_input("El costo de mercancía vendida es:", min_value=0.0, value=0.0, step=10.0, key="n1_practice_user_cogs")
            user_comment = st.text_area("Justifica brevemente (opcional):", key="n1_practice_comment")
            ask_ai = st.checkbox("💬 Pedir feedback de IA (opcional)", key="n1_practice_ai", value=False)
            submitted = st.form_submit_button("✅ Validar práctica")

        if submitted:
            st.toast("✅ Respuesta recibida, validando...", icon="✅")
            inv0 = st.session_state.n1p_inv0
            compras = st.session_state.n1p_compras
            devol = st.session_state.n1p_devol
            invf = st.session_state.n1p_invf
            correct = inv0 + compras - devol - invf
            if abs(user_cogs - correct) <= 0.5:
                st.success(f"¡Correcto! El **costo de la mercancía vendida** es {peso(correct)}")
            else:
                st.error(f"No coincide. El **costo de la mercancía vendida** esperado era {peso(correct)}")

            if ask_ai:
                with st.spinner("Generando feedback con IA..."):
                    prompt = (
                        f"Valida el cálculo del estudiante: costo_estimado={user_cogs:.2f}. "
                        f"Datos: Inventario_inicial={inv0:.2f}, Compras={compras:.2f}, Devoluciones_en_compras={devol:.2f}, Inventario_final={invf:.2f}. "
                        f"Costo_correcto={correct:.2f}. Comentario del estudiante: {user_comment}"
                    )
                    fb = ia_feedback(prompt)
                with st.expander("💬 Feedback de la IA"):
                    st.write(fb)

    # Evaluación final — 5 preguntas (2 selección múltiple, 2 cálculo, 1 abierta IA)
    with tabs[3]:
        st.subheader("Evaluación final del Nivel 1")
        st.caption("Son 5 preguntas. Apruebas con **4 de 5**.")

        # ---------- Claves correctas / datos de cálculo ----------
        # P1: Fórmula correcta
        P1_CORRECTA = "Inventario inicial + Compras − Devoluciones en compras − Inventario final"
        # P2: Afirmación verdadera
        P2_CORRECTA = "Un inventario final más alto, manteniendo todo lo demás igual, reduce el costo de la mercancía vendida."

        # P3: Cálculo directo
        P3_invI, P3_comp, P3_dev, P3_invF = 1800.0, 4500.0, 300.0, 1200.0
        P3_CORRECTO = P3_invI + P3_comp - P3_dev - P3_invF  # 4800.0

        # P4: Cálculo inverso (despejar Inventario final)
        P4_invI, P4_comp, P4_dev, P4_cmv = 2400.0, 3600.0, 200.0, 4000.0
        P4_CORRECTO = P4_invI + P4_comp - P4_dev - P4_cmv    # 1800.0

        TOL = 0.5  # tolerancia para respuestas numéricas
        TOTAL_ITEMS = 5
        PASS_MIN = 4

        with st.form("n1_eval_form_v2", clear_on_submit=False):
            # ---------- Pregunta 1 (Selección múltiple) ----------
            st.markdown("**1) En un sistema periódico, ¿cuál es la fórmula correcta para calcular el costo de la mercancía vendida?**")
            p1_opts = [
                "Inventario inicial + Compras + Devoluciones en compras − Inventario final",
                P1_CORRECTA,
                "Inventario final + Compras − Devoluciones en compras − Ventas",
                "Inventario inicial + Ventas − Devoluciones en compras − Inventario final",
            ]
            q1 = st.radio("Selecciona una opción:", p1_opts, index=None, key="n1v2_q1")

            st.markdown("---")

            # ---------- Pregunta 2 (Selección múltiple) ----------
            st.markdown("**2) En un sistema periódico, selecciona la afirmación verdadera:**")
            p2_opts = [
                "El costo de la mercancía vendida se conoce con certeza en cada venta.",
                "Las devoluciones en compras aumentan el costo de la mercancía vendida.",
                P2_CORRECTA,
                "El costo de la mercancía vendida no se ve afectado por las devoluciones en compras.",
            ]
            q2 = st.radio("Selecciona una opción:", p2_opts, index=None, key="n1v2_q2")

            st.markdown("---")

            # ---------- Pregunta 3 (Cálculo directo) ----------
            st.markdown("**3) Con base en esta información. Calcule el costo de la mercancía vendida**")
            st.markdown(
                f"""
                <div style='line-height:1.8; font-size:1.05rem;'>
                <b>Datos:</b><br>
                • Inventario inicial = <b>{peso(P3_invI)}</b><br>
                • Compras = <b>{peso(P3_comp)}</b><br>
                • Devoluciones en compras = <b>{peso(P3_dev)}</b><br>
                • Inventario final = <b>{peso(P3_invF)}</b>
                </div>
                """,
                unsafe_allow_html=True
            )
            q3 = st.number_input("Escribe tu resultado ($):", min_value=0.0, value=0.0, step=10.0, key="n1v2_q3")

            st.markdown("---")

            # ---------- Pregunta 4 (Cálculo inverso) ----------
            st.markdown("**4) Con base en esta información. Calcule el valor del inventario final**")
            st.markdown(
                f"""
                <div style='line-height:1.8; font-size:1.05rem;'>
                <b>Datos:</b><br>
                • Inventario inicial = <b>{peso(P4_invI)}</b><br>
                • Compras = <b>{peso(P4_comp)}</b><br>
                • Devoluciones en compras = <b>{peso(P4_dev)}</b><br>
                • Costo de la mercancía vendida = <b>{peso(P4_cmv)}</b>
                </div>
                """,
                unsafe_allow_html=True
            )
            q4 = st.number_input("Inventario final ($):", min_value=0.0, value=0.0, step=10.0, key="n1v2_q4")

            st.markdown("---")

            # ---------- Pregunta 5 (Abierta, validación con IA) ----------
            st.markdown("**5) Respuesta abierta (validada con IA)**")
            st.write(
                "Explica **qué ocurre con el costo de la mercancía vendida** cuando **disminuye el inventario final**, "
                "manteniendo constantes el inventario inicial, las compras y las devoluciones en compras."
            )
            q5_text = st.text_area("Tu explicación en 2–4 líneas:", key="n1v2_q5")

            ask_ai = st.checkbox("Mostrar el resultado detallado de la validación por IA", value=True, key="n1v2_ai_show")

            submitted = st.form_submit_button("🧪 Validar evaluación")

        if submitted:
            st.toast("✅ Respuestas recibidas, validando...", icon="✅")

            score = 0
            details = []

            # P1
            ok1 = (q1 == P1_CORRECTA)
            if ok1: score += 1
            details.append(("1) Fórmula correcta", ok1))

            # P2
            ok2 = (q2 == P2_CORRECTA)
            if ok2: score += 1
            details.append(("2) Afirmación verdadera", ok2))

            # P3
            ok3 = (abs((q3 or 0.0) - P3_CORRECTO) <= TOL)
            if ok3: score += 1
            details.append(("3) Cálculo directo", ok3))

            # P4
            ok4 = (abs((q4 or 0.0) - P4_CORRECTO) <= TOL)
            if ok4: score += 1
            details.append(("4) Cálculo inverso", ok4))

            # P5 — validación con IA
            ok5, fb5_short, fb5_retro = n1_eval_open_ai(q5_text)
            if ok5: score += 1
            details.append(("5) Respuesta abierta (IA)", ok5))

            # Resultado
            passed = (score >= PASS_MIN)
            record_attempt(username, level=1, score=score, passed=passed)  # Mongo: sin cambios

            # Feedback al estudiante
            cols = st.columns([1,1,1])
            cols[0].metric("Aciertos", f"{score}/{TOTAL_ITEMS}")
            cols[1].metric("Regla de aprobación", f"{PASS_MIN} de {TOTAL_ITEMS}")
            cols[2].metric("Estado", "✅ Aprobado" if passed else "❌ No aprobado")

            with st.expander("Detalle por pregunta"):
                for label, ok in details:
                    st.write(f"{'✅' if ok else '❌'} {label}")

            if ask_ai:
                with st.expander("💬 Resultado de la IA (pregunta 5)"):
                    st.markdown(f"**Resultado:** {'✅ Aprobado' if ok5 else '❌ No aprobado'}")
                    st.write(fb5_short)
                    st.markdown("---")
                    st.info(f"**Retroalimentación formativa:** {fb5_retro}")

            if passed:
                # Guarda progreso y navega al Nivel 2
                set_level_passed(st.session_state["progress_col"], username, "level1", score)
                st.session_state["sidebar_next_select"] = "Nivel 2: Métodos (PP/PEPS/UEPS)"
                start_celebration(
                    message_md=(
                        "<b>¡Nivel 1 superado!</b> 🏆<br><br>"
                        "Dominaste la fórmula del <b>costo de la mercancía vendida</b> y el sistema periódico. "
                        "Ahora sí: pasemos a los <b>métodos de valoración</b>."
                    ),
                    next_label="Nivel 2",
                    next_key_value="Nivel 2: Métodos (PP/PEPS/UEPS)"
                )
            else:
                st.error(f"No aprobado. Aciertos {score}/{TOTAL_ITEMS}. Repasa la teoría y vuelve a intentar.")


# ===========================
# NIVEL 2 (Métodos PP/PEPS/UEPS)
# ===========================
def page_level2(username):
    st.title("Nivel 2 · Métodos de valoración: Promedio Ponderado, PEPS (FIFO) y UEPS")

    tabs = st.tabs(["🎧 Teoría", "🛠 Ejemplos guiados", "🎮 Práctica (IA)", "🏁 Evaluación para aprobar"])

    with tabs[0]:
        theory = (
            "**Promedio Ponderado (PP):** mezcla lotes y asigna un costo promedio a cada unidad.\n\n"
            "**PEPS (FIFO):** las primeras unidades en entrar son las primeras en salir; el inventario final queda con los costos más recientes.\n\n"
            "**UEPS (LIFO):** las últimas en entrar son las primeras en salir; el inventario final queda con los costos más antiguos.\n\n"
            "Bajo inflación:\n"
            "- **PEPS** → COGS menor, utilidades mayores (inventario final más alto).\n"
            "- **UEPS** → COGS mayor, utilidades menores (inventario final más bajo). *No aceptado por NIIF plenas*.\n"
            "- **PP** suaviza la volatilidad de precios."
        )
        st.write(theory)
        with st.expander("🔊 Escuchar explicación"):
            speak_block(theory, key_prefix="teo-n2", lang_hint="es")

    with tabs[1]:
        st.subheader("Ejemplo de PP dinámico")
        colA, colB = st.columns([1,1])
        with colA:
            inv0_u = st.number_input("Inv. inicial (u)", min_value=0, value=100, step=10, key="n2_pp_inv_u")
            inv0_pu = st.number_input("Inv. inicial $/u", min_value=0.0, value=15.0, step=0.5, key="n2_pp_inv_pu")
            comp_u = st.number_input("Compra (u)", min_value=0, value=150, step=10, key="n2_pp_comp_u")
            comp_pu = st.number_input("Compra $/u", min_value=0.0, value=18.0, step=0.5, key="n2_pp_comp_pu")
            venta_u = st.number_input("Venta (u)", min_value=0, value=150, step=10, key="n2_pp_venta_u")
        with colB:
            inv0_val = inv0_u * inv0_pu
            comp_val = comp_u * comp_pu
            total_u = inv0_u + comp_u
            total_val = inv0_val + comp_val
            prom = (total_val / total_u) if total_u > 0 else 0
            cogs = min(venta_u, total_u) * prom
            saldo_u = max(total_u - venta_u, 0)
            saldo_val = saldo_u * prom

            st.markdown(
                "\n".join([
                    f"**Costo Promedio** = ({peso(inv0_val)} + {peso(comp_val)}) / ({inv0_u} + {comp_u}) = **{peso(prom)}**/u",
                    f"**COGS** por venta de {venta_u} u = {venta_u} × {peso(prom)} = **{peso(cogs)}**",
                    f"**Saldo final**: {saldo_u} u × {peso(prom)} = **{peso(saldo_val)}**"
                ])
            )

        st.markdown("---")
        st.subheader("Ejemplo FIFO vs LIFO (comparación rápida)")
        inv = [(100, 10.0), (50, 12.0)]
        venta = 120
        fifo_cogs = 0.0; remaining = venta; inv_fifo = inv.copy()
        for u, pu in inv_fifo:
            use = min(remaining, u)
            fifo_cogs += use * pu
            remaining -= use
            if remaining <= 0: break
        lifo_cogs = 0.0; remaining = venta; inv_lifo = inv.copy()[::-1]
        for u, pu in inv_lifo:
            use = min(remaining, u)
            lifo_cogs += use * pu
            remaining -= use
            if remaining <= 0: break

        st.info(f"Venta: {venta} u. Inventario: 100u @10; 50u @12")
        st.success(f"**FIFO COGS** ≈ {peso(fifo_cogs)} · **LIFO COGS** ≈ {peso(lifo_cogs)}  → (LIFO mayor COGS con precios al alza)")

    with tabs[2]:
        st.subheader("Práctica: elige el método correcto")
        st.caption("Completa el cálculo según el método seleccionado.")

        if "n2_inv0_u" not in st.session_state:
            n2_new_case()
        st.button("🔄 Nuevo escenario", on_click=n2_new_case, key="n2_new_case_btn")

        inv0_u  = st.session_state.n2_inv0_u
        inv0_pu = st.session_state.n2_inv0_pu
        comp_u  = st.session_state.n2_comp_u
        comp_pu = st.session_state.n2_comp_pu
        venta_u = st.session_state.n2_venta_u

        st.write(f"Inv0: {inv0_u} u @ {peso(inv0_pu)} | Compra: {comp_u} u @ {peso(comp_pu)} | Venta: {venta_u} u")

        with st.form("n2_prac_form"):
            metodo = st.selectbox("Método", ["Promedio Ponderado", "PEPS (FIFO)", "UEPS (LIFO)"], key="n2_pract_met")
            ans_cogs = st.number_input("Tu COGS", min_value=0.0, value=0.0, step=10.0, key="n2_prac_cogs")
            ask_ai = st.checkbox("💬 Pedir feedback de IA (opcional)", key="n2_prac_ai", value=False)
            submitted = st.form_submit_button("✅ Validar práctica N2")

        if submitted:
            st.toast("✅ Respuesta recibida, validando...", icon="✅")
            total_u = inv0_u + comp_u
            inv0_val = inv0_u * inv0_pu
            comp_val = comp_u * comp_pu

            if metodo == "Promedio Ponderado":
                prom = (inv0_val + comp_val) / total_u
                correct = min(venta_u, total_u) * prom
            elif metodo == "PEPS (FIFO)":
                remaining = venta_u
                correct = 0.0
                use = min(remaining, inv0_u)
                correct += use * inv0_pu
                remaining -= use
                if remaining > 0:
                    use2 = min(remaining, comp_u)
                    correct += use2 * comp_pu
            else:  # UEPS
                remaining = venta_u
                correct = 0.0
                use = min(remaining, comp_u)
                correct += use * comp_pu
                remaining -= use
                if remaining > 0:
                    use2 = min(remaining, inv0_u)
                    correct += use2 * inv0_pu

            ok = abs(ans_cogs - correct) <= 0.5
            if ok:
                st.success(f"¡Bien! COGS esperado ≈ {peso(correct)}")
            else:
                st.error(f"COGS esperado ≈ {peso(correct)}")

            if ask_ai:
                with st.spinner("Generando feedback con IA..."):
                    fb = ia_feedback(
                        f"Práctica N2 con {metodo}. Datos: Inv0={inv0_u}@{inv0_pu}, Comp={comp_u}@{comp_pu}, Venta={venta_u}. "
                        f"COGS_est={ans_cogs}, COGS_correcto={correct}. "
                        f"Explica el porqué del cálculo en máximo 6 líneas con un truco memotécnico."
                    )
                with st.expander("💬 Feedback de la IA"):
                    st.write(fb)

    with tabs[3]:
        st.subheader("Evaluación final del Nivel 2")
        st.caption("Necesitas acertar **2 de 3**.")

        with st.form("n2_eval_form"):
            q1 = st.radio("1) En inflación, ¿cuál suele dar mayor COGS?",
                          ["PEPS", "UEPS", "Promedio Ponderado"], index=None, key="n2_eval_q1")
            q2 = st.radio("2) En PEPS, ¿con qué costos se valora el inventario final?",
                          ["Con los más antiguos", "Con los más recientes"], index=None, key="n2_eval_q2")
            q3 = st.radio("3) El Promedio Ponderado:",
                          ["Usa costo del último lote", "Mezcla costos para un único costo unitario"], index=None, key="n2_eval_q3")
            ask_ai = st.checkbox("💬 Pedir feedback de IA (opcional)", key="n2_eval_ai", value=False)
            submitted = st.form_submit_button("🧪 Validar evaluación N2")

        if submitted:
            st.toast("✅ Respuesta recibida, validando...", icon="✅")
            correct = {
                "n2_eval_q1": "UEPS",
                "n2_eval_q2": "Con los más recientes",
                "n2_eval_q3": "Mezcla costos para un único costo unitario"
            }
            answers = {"n2_eval_q1": q1, "n2_eval_q2": q2, "n2_eval_q3": q3}
            score = sum(1 for k,v in answers.items() if v == correct[k])
            passed = score >= 2

            record_attempt(username, level=2, score=score, passed=passed)

            fb = None
            if ask_ai:
                with st.spinner("Generando feedback con IA..."):
                    fb = ia_feedback(
                        f"Nivel 2 evaluación. Respuestas estudiante: {answers}. Correctas: {correct}. "
                        f"Aciertos: {score}/3. Da feedback amable y breve."
                    )

            if passed:
                set_level_passed(st.session_state["progress_col"], username, "level2", score)
                st.session_state["sidebar_next_select"] = "Nivel 3: Devoluciones"
                start_celebration(
                    message_md=(
                        "<b>¡Nivel 2 completado!</b> 🧠✨<br><br>"
                        "Ya dominas <b>PP / PEPS / UEPS</b>. Vamos a meterle realismo: "
                        "<b>devoluciones</b> que ajustan compras y ventas."
                    ),
                    next_label="Nivel 3",
                    next_key_value="Nivel 3: Devoluciones"
                )

            else:
                st.error(f"No aprobado. Aciertos {score}/3. Repasa y vuelve a intentar.")
                if fb:
                    with st.expander("💬 Feedback de la IA"):
                        st.write(fb)

# ===========================
# NIVEL 3 (Devoluciones)
# ===========================
def page_level3(username):
    st.title("Nivel 3 · Casos con Devoluciones (compras y ventas)")

    tabs = st.tabs(["🎧 Teoría", "🛠 Ejemplos", "🎮 Práctica (IA)", "🏁 Evaluación para aprobar"])

    with tabs[0]:
        theory = (
            "**Devoluciones de compra**: restan a compras; reducen el pool de costo disponible.\n\n"
            "**Devoluciones de venta**: el cliente devuelve unidades → reingresan al inventario. "
            "Su valoración depende del método (PP, PEPS, UEPS). En **periódico**, se suele ajustar en las ventas netas "
            "y, si corresponde, reconocer el costo del reingreso a inventario.\n\n"
            "Idea clave: mantén consistencia con el método de inventario y registra contra la cuenta correcta."
        )
        st.write(theory)
        with st.expander("🔊 Escuchar explicación"):
            speak_block(theory, key_prefix="teo-n3", lang_hint="es")

    with tabs[1]:
        st.subheader("Devolución de compra (impacto directo en Compras)")
        compra = st.number_input("Compra bruta ($)", min_value=0.0, value=5000.0, step=100.0, key="n3_ej_compra")
        dev_comp = st.number_input("Devolución a proveedor ($)", min_value=0.0, value=600.0, step=50.0, key="n3_ej_devcomp")
        compras_net = compra - dev_comp
        st.info(f"**Compras netas = {peso(compra)} − {peso(dev_comp)} = {peso(compras_net)}")

        st.subheader("Devolución de venta (reingreso de unidades)")
        st.caption("Escenario simple PP: el costo reingresado es el costo promedio vigente.")
        prom = st.number_input("Costo promedio vigente ($/u)", min_value=0.0, value=16.8, step=0.1, key="n3_ej_prompp")
        dev_venta_u = st.number_input("Unidades devueltas por cliente", min_value=0, value=10, step=1, key="n3_ej_devventa_u")
        costo_reingreso = prom * dev_venta_u
        st.success(f"**Reingreso inventario**: {dev_venta_u} u × {peso(prom)} = {peso(costo_reingreso)}")

    with tabs[2]:
        st.subheader("Práctica: combina compras netas y devolución de venta (PP)")

        if "n3_inv0" not in st.session_state:
            n3_new_case()
        st.button("🔄 Nuevo escenario", on_click=n3_new_case, key="n3_new_case_btn")

        inv0    = st.session_state.n3_inv0
        prom0   = st.session_state.n3_prom0
        comp    = st.session_state.n3_comp
        comp_pu = st.session_state.n3_comp_pu
        dev_comp= st.session_state.n3_dev_comp
        venta_u = st.session_state.n3_venta_u
        dev_venta_u = st.session_state.n3_dev_v_u

        st.write(
            f"Inv0: {inv0} u @ {peso(prom0)} | Compra: {comp} u @ {peso(comp_pu)} | "
            f"Devol. compra: {peso(dev_comp)} (resta $) | Venta: {venta_u} u | Devol. venta: {dev_venta_u} u"
        )

        with st.form("n3_prac_form"):
            ans_cogs = st.number_input("Tu COGS estimado (PP)", min_value=0.0, value=0.0, step=10.0, key="n3_prac_cogs")
            ask_ai = st.checkbox("💬 Pedir feedback de IA (opcional)", key="n3_prac_ai", value=False)
            submitted = st.form_submit_button("✅ Validar práctica N3")

        if submitted:
            st.toast("✅ Respuesta recibida, validando...", icon="✅")
            inv0_val = inv0 * prom0
            comp_val = comp * comp_pu
            comp_net_val = comp_val - dev_comp
            total_val = inv0_val + comp_net_val
            total_u = inv0 + comp
            prom = total_val / total_u if total_u > 0 else 0.0
            venta_neta_u = max(venta_u - dev_venta_u, 0)
            correct = venta_neta_u * prom

            ok = abs(ans_cogs - correct) <= 0.5
            if ok:
                st.success(f"COGS (venta neta) ≈ {peso(correct)} con PP")
            else:
                st.error(f"COGS esperado ≈ {peso(correct)}")

            if ask_ai:
                with st.spinner("Generando feedback con IA..."):
                    fb = ia_feedback(
                        f"N3 práctica PP con devoluciones. Datos: Inv0={inv0}@{prom0}, Comp={comp}@{comp_pu}, "
                        f"DevCompra=${dev_comp}, Venta={venta_u}, DevVenta={dev_venta_u}. "
                        f"COGS_est={ans_cogs}, COGS_correcto={correct}. Explica el razonamiento."
                    )
                with st.expander("💬 Feedback de la IA"):
                    st.write(fb)

    with tabs[3]:
        st.subheader("Evaluación final del Nivel 3")
        st.caption("Necesitas acertar **2 de 3**.")

        with st.form("n3_eval_form"):
            q1 = st.radio("1) La devolución de compra...",
                          ["Aumenta las compras", "Disminuye las compras", "No afecta las compras"], index=None, key="n3_eval_q1")
            q2 = st.radio("2) La devolución de venta (PP) reingresa unidades con costo...",
                          ["Del último lote", "Promedio vigente", "Más antiguo"], index=None, key="n3_eval_q2")
            q3 = st.radio("3) En términos de COGS, una devolución de venta...",
                          ["Disminuye el COGS neto", "Aumenta el COGS neto", "No lo afecta"], index=None, key="n3_eval_q3")
            ask_ai = st.checkbox("💬 Pedir feedback de IA (opcional)", key="n3_eval_ai", value=False)
            submitted = st.form_submit_button("🧪 Validar evaluación N3")

        if submitted:
            st.toast("✅ Respuesta recibida, validando...", icon="✅")
            correct = {
                "n3_eval_q1": "Disminuye las compras",
                "n3_eval_q2": "Promedio vigente",
                "n3_eval_q3": "Disminuye el COGS neto"
            }
            answers = {"n3_eval_q1": q1, "n3_eval_q2": q2, "n3_eval_q3": q3}
            score = sum(1 for k,v in answers.items() if v == correct[k])
            passed = score >= 2

            record_attempt(username, level=3, score=score, passed=passed)

            fb = None
            if ask_ai:
                with st.spinner("Generando feedback con IA..."):
                    fb = ia_feedback(
                        f"Nivel 3 evaluación. Respuestas estudiante: {answers}. Correctas: {correct}. "
                        f"Aciertos: {score}/3. Da feedback breve y amable."
                    )

            if passed:
                set_level_passed(st.session_state["progress_col"], username, "level3", score)
                st.session_state["sidebar_next_select"] = "Nivel 4: Estado de Resultados"
                start_celebration(
                    message_md=(
                        "<b>¡Nivel 3 dominado!</b> 🔁📦<br><br>"
                        "Entendiste cómo ajustar por <b>devoluciones</b>. "
                        "Ahora a integrar todo en el <b>Estado de Resultados</b>."
                    ),
                    next_label="Nivel 4",
                    next_key_value="Nivel 4: Estado de Resultados"
                )

            else:
                st.error(f"No aprobado. Aciertos {score}/3. Repasa y vuelve a intentar.")
                if fb:
                    with st.expander("💬 Feedback de la IA"):
                        st.write(fb)

# ===========================
# NIVEL 4 (Estado de Resultados)
# ===========================
def page_level4(username):
    st.title("Nivel 4 · Construcción del Estado de Resultados (simplificado)")

    tabs = st.tabs(["🎧 Teoría", "🛠 Ejemplo guiado", "🎮 Práctica (IA)", "🏁 Evaluación final + Encuesta"])

    with tabs[0]:
        theory = (
            "El **Estado de Resultados** muestra ingresos y gastos del período, hasta la **utilidad neta**. "
            "En una empresa comercial sencilla:\n\n"
            "- **Ventas netas** = Ventas brutas − Devoluciones/Descuentos sobre ventas\n"
            "- **COGS** (costo de ventas) → de tus métodos de inventario\n"
            "- **Utilidad bruta** = Ventas netas − COGS\n"
            "- **Gastos operativos** (administrativos, ventas)\n"
            "- **Utilidad operativa** = Utilidad bruta − Gastos operativos"
        )
        st.write(theory)
        with st.expander("🔊 Escuchar explicación"):
            speak_block(theory, key_prefix="teo-n4", lang_hint="es")

    with tabs[1]:
        st.subheader("Ejemplo simple")
        colL, colR = st.columns(2)
        with colL:
            ventas = st.number_input("Ventas brutas", min_value=0.0, value=12000.0, step=100.0, key="n4_ex_ventas")
            dev_ventas = st.number_input("Devol. y Descuentos sobre ventas", min_value=0.0, value=500.0, step=50.0, key="n4_ex_dev_vtas")
            cogs = st.number_input("COGS", min_value=0.0, value=7000.0, step=100.0, key="n4_ex_cogs")
            gastos = st.number_input("Gastos operativos", min_value=0.0, value=2000.0, step=100.0, key="n4_ex_gastos")
        with colR:
            vtas_net = ventas - dev_ventas
            util_bruta = vtas_net - cogs
            util_oper = util_bruta - gastos
            st.info(f"**Ventas netas** = {peso(ventas)} − {peso(dev_ventas)} = **{peso(vtas_net)}**")
            st.info(f"**Utilidad bruta** = {peso(vtas_net)} − {peso(cogs)} = **{peso(util_bruta)}**")
            st.success(f"**Utilidad operativa** = {peso(util_bruta)} − {peso(gastos)} = **{peso(util_oper)}**")

    with tabs[2]:
        st.subheader("Práctica: arma tu Estado de Resultados")

        if "n4_ventas" not in st.session_state:
            n4_new_case()
        st.button("🔄 Nuevo escenario", on_click=n4_new_case, key="n4_new_case_btn")

        ventas   = st.session_state.n4_ventas
        dev_vtas = st.session_state.n4_dev_vtas
        cogs     = st.session_state.n4_cogs
        gastos   = st.session_state.n4_gastos

        st.write(
            f"Ventas brutas={peso(ventas)}, Devol/Desc Ventas={peso(dev_vtas)}, "
            f"COGS={peso(cogs)}, Gastos Op.={peso(gastos)}"
        )

        with st.form("n4_prac_form"):
            ans_util_oper = st.number_input("Tu Utilidad Operativa", min_value=-100000.0, value=0.0, step=50.0, key="n4_prac_uop")
            ask_ai = st.checkbox("💬 Pedir feedback de IA (opcional)", key="n4_prac_ai", value=False)
            submitted = st.form_submit_button("✅ Validar práctica N4")

        if submitted:
            st.toast("✅ Respuesta recibida, validando...", icon="✅")
            vtas_net = ventas - dev_vtas
            util_bruta = vtas_net - cogs
            correct = util_bruta - gastos
            if abs(ans_util_oper - correct) <= 0.5:
                st.success(f"¡Correcto! Utilidad operativa = {peso(correct)}")
            else:
                st.error(f"Utilidad operativa esperada = {peso(correct)}")

            if ask_ai:
                with st.spinner("Generando feedback con IA..."):
                    fb = ia_feedback(
                        f"N4 práctica EERR. Datos: Ventas={ventas}, DevVtas={dev_vtas}, COGS={cogs}, Gastos={gastos}. "
                        f"UO_est={ans_util_oper}, UO_correcta={correct}. Explica pasos y da truco memotécnico."
                    )
                with st.expander("💬 Feedback de la IA"):
                    st.write(fb)

    with tabs[3]:
        st.subheader("Evaluación final del Nivel 4")
        st.caption("Necesitas acertar **2 de 3** para terminar el curso.")

        with st.form("n4_eval_form"):
            q1 = st.radio("1) Ventas netas se calculan como:",
                          ["Ventas brutas + Devoluciones", "Ventas brutas − Devoluciones/Descuentos", "Ventas brutas"], index=None, key="n4_eval_q1")
            q2 = st.radio("2) Utilidad bruta =",
                          ["Ventas netas − COGS", "Ventas netas − Gastos operativos", "Ventas brutas − COGS"], index=None, key="n4_eval_q2")
            q3 = st.radio("3) Utilidad operativa =",
                          ["Utilidad bruta − Gastos operativos", "Ventas netas − COGS − Gastos financieros", "COGS − Gastos operativos"], index=None, key="n4_eval_q3")
            ask_ai = st.checkbox("💬 Pedir feedback de IA (opcional)", key="n4_eval_ai", value=False)
            submitted = st.form_submit_button("🧪 Validar evaluación N4")

        if submitted:
            st.toast("✅ Respuesta recibida, validando...", icon="✅")
            correct = {
                "n4_eval_q1": "Ventas brutas − Devoluciones/Descuentos",
                "n4_eval_q2": "Ventas netas − COGS",
                "n4_eval_q3": "Utilidad bruta − Gastos operativos"
            }
            answers = {"n4_eval_q1": q1, "n4_eval_q2": q2, "n4_eval_q3": q3}
            score = sum(1 for k,v in answers.items() if v == correct[k])
            passed = score >= 2

            record_attempt(username, level=4, score=score, passed=passed)

            fb = None
            if ask_ai:
                with st.spinner("Generando feedback con IA..."):
                    fb = ia_feedback(
                        f"Nivel 4 evaluación. Respuestas estudiante: {answers}. Correctas: {correct}. "
                        f"Aciertos: {score}/3. Feedback amable y breve."
                    )

            if passed:
                set_level_passed(st.session_state["progress_col"], username, "level4", score)
                set_completed_survey(st.session_state["progress_col"], username, True)
                st.session_state["sidebar_next_select"] = "Encuesta de satisfacción"
                start_celebration(
                    message_md=(
                        "<b>¡Curso completado!</b> 🎓🌟<br><br>"
                        "Has recorrido desde el COGS básico hasta el EERR. "
                        "Por favor responde la <b>Encuesta de satisfacción</b> para ayudarnos a mejorar."
                    ),
                    next_label="Encuesta de satisfacción",
                    next_key_value="Encuesta de satisfacción"
                )

            else:
                st.error(f"No aprobado. Aciertos {score}/3. Refuerza conceptos y vuelve a intentar.")
                if fb:
                    with st.expander("💬 Feedback de la IA"):
                        st.write(fb)

# ===========================
# Página: Encuesta de satisfacción
# ===========================
def page_survey():
    st.title("📝 Encuesta de satisfacción")
    st.write(
        "Tu opinión nos ayuda a mejorar esta herramienta. "
        "Por favor abre el siguiente enlace y completa el formulario:"
    )
    st.markdown(f"- 👉 **[Abrir formulario de encuesta]({SURVEY_URL})**")
    st.caption("Si el enlace no abre en tu navegador, copia y pégalo en otra pestaña.")

# ===========================
# Módulo: Administrador de Usuarios (Mongo) + Estadísticas
# ===========================

@st.cache_data(ttl=15, show_spinner=False)
def get_user_list(_users_col, _cache_key: str = "users:list"):
    # _users_col NO se usa para el hash del caché (por el guion bajo)
    return [u["username"] for u in _users_col.find({}, {"username":1, "_id":0}).sort("username",1)]


@st.cache_data(ttl=30, show_spinner=False)
def attempts_kpis(_attempts_col, _cache_key: str = "attempts:kpis"):
    # KPIs globales (igual que antes)
    agg_base = list(_attempts_col.aggregate([
        {"$group": {
            "_id": None,
            "total_intentos": {"$sum": 1},
            "usuarios_unicos": {"$addToSet": "$username"},
            "aprobados": {"$sum": {"$cond": ["$passed", 1, 0]}}
        }},
        {"$project": {
            "_id": 0,
            "total_intentos": 1,
            "total_usuarios": {"$size": "$usuarios_unicos"},
            "tasa_global": {"$cond": [
                {"$eq": ["$total_intentos", 0]},
                0,
                {"$multiply": [{"$divide": ["$aprobados", "$total_intentos"]}, 100]}
            ]}
        }}
    ]))
    kpis = agg_base[0] if agg_base else {"total_intentos":0,"total_usuarios":0,"tasa_global":0}

    lvl_rate = list(_attempts_col.aggregate([
        {"$group": {"_id": "$level", "aprobacion": {"$avg": {"$cond": ["$passed", 1, 0]}}}},
        {"$project": {"level": "$_id", "_id":0, "aprobacion_%": {"$multiply": ["$aprobacion", 100]}}},
        {"$sort": {"level": 1}}
    ]))

    lvl_score = list(_attempts_col.aggregate([
        {"$match": {"score": {"$ne": None}}},
        {"$group": {"_id": "$level", "prom_puntaje": {"$avg": "$score"}}},
        {"$project": {"level": "$_id", "_id":0, "prom_puntaje": 1}},
        {"$sort": {"level": 1}}
    ]))

    last25 = list(_attempts_col.find({}, {"_id":0}).sort("created_at",-1).limit(25))
    return kpis, lvl_rate, lvl_score, last25


def admin_page():
    st.title("⚙️ Administrador de Usuarios")

    users_col = st.session_state.get("users_col")
    progress_col = st.session_state.get("progress_col")
    attempts_col = st.session_state.get("attempts_col")
    if users_col is None:
        st.error("No hay conexión con MongoDB.")
        return

    tab_users, tab_stats = st.tabs(["👥 Usuarios", "📊 Estadísticas"])

    # ---------- TAB: USUARIOS ----------
    with tab_users:
        st.subheader("Usuarios actuales")
        data = list(users_col.find({}, {"_id": 0, "username": 1, "role": 1, "created_at": 1}))
        st.data_editor(pd.DataFrame(data), disabled=True, use_container_width=True)

        st.markdown("---")

        st.subheader("Crear nuevo usuario")
        with st.form("admin_create_user"):
            new_user = st.text_input("Nombre de usuario (min 3, sin espacios)").strip().lower()
            new_pass = st.text_input("Contraseña (min 4)", type="password")
            new_role = st.selectbox("Rol", ["user", "admin"])
            submitted = st.form_submit_button("➕ Crear usuario")

        if submitted:
            if not new_user or len(new_user) < 3 or " " in new_user:
                st.error("Usuario inválido. Debe tener al menos 3 caracteres y sin espacios.")
            elif not new_pass or len(new_pass) < 4:
                st.error("Contraseña demasiado corta (mínimo 4).")
            elif users_col.find_one({"username": new_user}):
                st.error("El usuario ya existe.")
            else:
                create_user(users_col, progress_col, new_user, new_pass, new_role)
                st.success(f"Usuario '{new_user}' creado como {new_role}.")
                st.cache_data.clear()  # refresca listados/estadísticas

        st.markdown("---")

        st.subheader("Editar usuario")
        usernames = get_user_list(users_col)
        if usernames:
            edit_user = st.selectbox("Selecciona el usuario a editar", usernames, key="admin_edit_select")
            if edit_user:
                curr = users_col.find_one({"username": edit_user}, {"role": 1, "_id": 0}) or {}
                curr_role = curr.get("role", "user")
                with st.form("admin_edit_user"):
                    new_pass_opt = st.text_input("Nueva contraseña (opcional: vacío = no cambiar)", type="password")
                    new_role_opt = st.selectbox("Nuevo rol", ["user", "admin"], index=0 if curr_role == "user" else 1)
                    submit_edit = st.form_submit_button("✏️ Guardar cambios")

                if submit_edit:
                    # Evitar quitar el último admin
                    if curr_role == "admin" and new_role_opt == "user":
                        other_admin = users_col.count_documents({"username": {"$ne": edit_user}, "role": "admin"}) > 0
                        if not other_admin:
                            st.error("No puedes quitar el último administrador del sistema.")
                        else:
                            update_user(users_col, edit_user, new_pass_opt or None, new_role_opt)
                            st.success(f"Usuario '{edit_user}' actualizado.")
                            st.cache_data.clear()
                    else:
                        update_user(users_col, edit_user, new_pass_opt or None, new_role_opt)
                        st.success(f"Usuario '{edit_user}' actualizado.")
                        st.cache_data.clear()
        else:
            st.info("No hay usuarios para editar.")

        st.markdown("---")

        st.subheader("Eliminar usuario")
        usernames = get_user_list(users_col)
        if usernames:
            del_user = st.selectbox("Selecciona el usuario a eliminar", usernames, key="admin_del_select")
            if st.button("🗑️ Eliminar usuario seleccionado"):
                if del_user == st.session_state.username:
                    st.error("No puedes eliminar tu propia cuenta en esta vista.")
                elif del_user == "admin":
                    st.error("Por seguridad no se permite eliminar la cuenta 'admin' por defecto.")
                else:
                    doc = users_col.find_one({"username": del_user}, {"role": 1, "_id": 0})
                    if doc and doc.get("role") == "admin":
                        other_admin = users_col.count_documents({"username": {"$ne": del_user}, "role": "admin"}) > 0
                        if not other_admin:
                            st.error("No puedes eliminar el último administrador del sistema.")
                        else:
                            delete_user(users_col, progress_col, del_user)
                            st.success(f"Usuario '{del_user}' eliminado.")
                            st.cache_data.clear()
                    else:
                        delete_user(users_col, progress_col, del_user)
                        st.success(f"Usuario '{del_user}' eliminado.")
                        st.cache_data.clear()
        else:
            st.info("No hay usuarios para eliminar.")

    # ---------- TAB: ESTADÍSTICAS ----------
    with tab_stats:
        st.subheader("Resumen de desempeño")
        if attempts_col is None:
            st.error("Colección de intentos no disponible.")
            return

        kpis, by_level, by_level_score, ult = attempts_kpis(attempts_col)

        c1, c2, c3 = st.columns(3)
        c1.metric("Intentos totales", f"{kpis['total_intentos']}")
        c2.metric("Usuarios únicos", f"{kpis['total_usuarios']}")
        c3.metric("Tasa aprobación global", f"{kpis['tasa_global']:.1f}%")

        st.markdown("---")
        st.subheader("Aprobación por nivel")
        df_lvl = pd.DataFrame(by_level).sort_values("level")
        if not df_lvl.empty:
            st.data_editor(df_lvl, disabled=True, use_container_width=True)
            st.bar_chart(df_lvl.set_index("level"))
        else:
            st.info("Sin datos por nivel aún.")

        st.markdown("---")
        st.subheader("Promedio de puntaje por nivel")
        df_lvl_score = pd.DataFrame(by_level_score).sort_values("level")
        if not df_lvl_score.empty:
            st.data_editor(df_lvl_score, disabled=True, use_container_width=True)
            st.bar_chart(df_lvl_score.set_index("level"))
        else:
            st.info("Sin puntajes aún.")

        st.markdown("---")
        st.subheader("Últimos 25 intentos")
        df_last = pd.DataFrame(ult)
        if not df_last.empty:
            if "created_at" in df_last.columns:
                df_last["created_at"] = pd.to_datetime(df_last["created_at"])
            if "passed" in df_last.columns:
                df_last["passed"] = df_last["passed"].map({True:"✅", False:"❌"})
            keep_cols = [c for c in ["created_at","username","level","score","passed"] if c in df_last.columns]
            st.data_editor(df_last.sort_values("created_at", ascending=False)[keep_cols], disabled=True, use_container_width=True)
        else:
            st.info("Aún no hay intentos registrados.")

# ===========================
# Pantalla Login
# ===========================
def login_screen():
    st.header("Iniciar Sesión")
    with st.form("login_form"):
        st.text_input("Usuario", key="login_raw_user")
        st.text_input("Contraseña", type="password", key="login_password")
        st.form_submit_button("Ingresar", on_click=do_login)
    if st.session_state.get("login_error"):
        st.error(st.session_state["login_error"])
    st.markdown("---")
    st.caption("Si es tu primera vez, inicia con tu usuario y contraseña asignados.")

# ===========================
# Router principal
# ===========================
def main_app():
    username = st.session_state.username

    if celebration_screen():
        return

    current = sidebar_nav(username)

    if current.startswith("Nivel 1"):
        page_level1(username)
    elif current == ADMIN_OPTION:
        users_col = st.session_state.get("users_col")
        role = "user"
        if users_col is not None:
            doc = users_col.find_one({"username": username}, {"role": 1, "_id": 0}) or {}
            role = doc.get("role", "user")
        if role != "admin":
            st.warning("No autorizado.")
            return
        admin_page()
    elif current.startswith("Nivel 2"):
        page_level2(username)
    elif current.startswith("Nivel 3"):
        page_level3(username)
    elif current.startswith("Nivel 4"):
        page_level4(username)
    elif current == "Encuesta de satisfacción":
        page_survey()
    else:
        page_level1(username)

# ===========================
# Entry
# ===========================
def main():
    init_session()

    # Inicializa conexión y colecciones (cache_resource)
    try:
        db, users_col, progress_col, attempts_col = repo_init()
        st.session_state["users_col"] = users_col
        st.session_state["progress_col"] = progress_col
        st.session_state["attempts_col"] = attempts_col
    except Exception as e:
        st.error(f"Error conectando a MongoDB: {e}")
        st.stop()

    # Flujo principal
    if not st.session_state.get("authenticated"):
        login_screen()
    else:
        main_app()

if __name__ == "__main__":
    main()

