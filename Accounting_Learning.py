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
import json, re

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

def ia_call(messages: list, temperature: float = 0.2) -> str:
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY ausente o inválida.")
    completion = client.chat.completions.create(
        model=DEEPSEEK_MODEL,  # definido arriba (recomendado sin :free)
        messages=messages,
        temperature=temperature,
        extra_body={}
    )
    text = (completion.choices[0].message.content or "").strip()
    return text

def _parse_first_json(s: str) -> dict:
    """
    Extrae y parsea SOLO el primer objeto JSON balanceado de la respuesta.
    Tolera ```json fences, tokens raros, 'aprobo', comillas simples, y texto extra después del JSON.
    """
    import json, re

    t = (s or "").strip()

    # Errores comunes (texto, no JSON)
    lower = t.lower()
    if ("no endpoints found" in lower or "unauthorized" in lower or
        "rate limit" in lower or "timeout" in lower):
        raise RuntimeError(t)

    # Limpieza básica
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.IGNORECASE|re.MULTILINE).strip()
    t = t.replace("<|begin_of_sentence|>", "").replace("<|end_of_sentence|>", "")
    t = t.replace("“","\"").replace("”","\"").replace("’","'")
    t = t.replace("\"aprobo\"", "\"aprobado\"").replace("'aprobo'", "'aprobado'")

    # --- Encontrar el PRIMER objeto JSON balanceado ---
    start = t.find("{")
    if start == -1:
        raise RuntimeError("No se encontró '{' en la respuesta del modelo.")

    brace = 0
    in_str = False
    esc = False
    end = None
    for i in range(start, len(t)):
        ch = t[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == "\"":
                in_str = False
        else:
            if ch == "\"":
                in_str = True
            elif ch == "{":
                brace += 1
            elif ch == "}":
                brace -= 1
                if brace == 0:
                    end = i
                    break
    if end is None:
        # No cerró, intento con recorte hasta último '}' por si vino basura
        last = t.rfind("}")
        if last == -1:
            raise RuntimeError("No se encontró cierre '}' en la respuesta del modelo.")
        end = last

    candidate = t[start:end+1].strip()

    # Intento 1: parseo directo
    try:
        return json.loads(candidate)
    except Exception:
        pass

    # Intento 2: normalizar comillas simples -> dobles (fuera de strings escapados)
    candidate2 = re.sub(r"(?<!\\)'", '"', candidate)
    try:
        return json.loads(candidate2)
    except Exception as e:
        # Último recurso: quitar comas finales antes de '}' si las hay
        candidate3 = re.sub(r",\s*}", "}", candidate2)
        return json.loads(candidate3)


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

def eval_ia_explicacion(pregunta: str, criterios: str, respuesta_estudiante: str) -> tuple[bool, str, str]:
    """
    Evalúa con IA una explicación abierta contra 'pregunta' y 'criterios'.
    Devuelve: (aprobado_bool, comentario_corto, retroalimentacion).
    """
    texto = (respuesta_estudiante or "").strip()

    system_msg = {
        "role": "system",
        "content": (
            "Eres un tutor de contabilidad empático, alentador y claro. "
            "Siempre retroalimentas con tono amable, sin regaños ni juicios. "
            "Estructura tu respuesta pedagógica así: "
            "1) Refuerzo positivo breve; "
            "2) Explicación clara y paso a paso; "
            "3) Analogía simple (por ejemplo: 'mochila de costos', 'balanza', 'caja que se vacía'); "
            "4) Un consejo práctico (‘siguiente paso’) para mejorar. "
            "Usa terminología local de Colombia: CMV (Costo de la Mercancía Vendida), no COGS. "
            "Evita tecnicismos innecesarios, sé conciso y motivador."
        )
    }

    user_msg = {
        "role": "user",
        "content": (
            f"Pregunta de evaluación:\n«{pregunta}»\n\n"
            f"Criterios de evaluación (deben cumplirse todos):\n{criterios}\n\n"
            "Responde EXCLUSIVAMENTE con este JSON (una sola línea, sin texto extra):\n"
            "{"
            "\"aprobado\": true|false, "
            "\"comentario_corto\": \"≤ 20 palabras (síntesis amable pero siempre diciendo que está correcto o incorrecto, es decir"
            "sin flexibilidad)\", "
            "\"retroalimentacion\": "
            "\"≤ 120 palabras, tono amable y pedagógico, inicia con refuerzo positivo, explica paso a paso, "
            "incluye una analogía sencilla (p.ej. 'mochila de costos' o 'balanza'), y cierra con un consejo práctico. "
            "Usa CMV (no COGS).\""
            "}\n\n"
            f"Respuesta del estudiante: \"{respuesta_estudiante.strip()}\""
        )
    }

    try:
        raw = ia_call([system_msg, user_msg], temperature=0.25)
        st.session_state["eval_raw"] = raw  # opcional diagnóstico
        data = _parse_first_json(raw)

        aprobado = bool(data.get("aprobado"))
        comentario = (data.get("comentario_corto") or "").strip()
        retro = (data.get("retroalimentacion") or "").strip()

        if not comentario:
            comentario = "IA: evaluación recibida."
        else:
            comentario = f"IA: {comentario}"
        if not retro:
            retro = ("Recuerda: al disminuir el inventario final se resta menos; por eso el CMV aumenta.")

        return aprobado, comentario, retro

    except Exception as e:
        # Fallback mínimo
        t = (texto or "").lower()
        inv_down = any(w in t for w in ["disminuye","disminuir","baja","menor","reduce","decrece","muy pequeño"])
        cost_up  = any(w in t for w in ["aumenta","sube","mayor","incrementa"])
        resta_menos = ("resta menos" in t) or ("se resta menos" in t)

        aprobado = inv_down and (cost_up or resta_menos)
        comentario = "Fallback: la IA no estuvo disponible."
        retro = ("Si el inventario final es muy pequeño, se descuenta menos en la fórmula y el CMV aumenta. "
                 "Memotecnia: «menos inventario final → más CMV».")
        return aprobado, comentario, retro


def n1_eval_open_ai(respuesta_estudiante: str) -> tuple[bool, str, str]:
    pregunta = "¿Qué ocurre con el CMV cuando el inventario final DISMINUYE y por qué?"
    criterios = (
        "1) Debe afirmar que el CMV AUMENTA.\n"
        "2) Debe justificar que al cierre se RESTA MENOS en la fórmula (CMV = InvI + Compras − InvF)."
    )
    return eval_ia_explicacion(pregunta, criterios, respuesta_estudiante)

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

    tabs = st.tabs(["🎧 Teoría", "🛠 Ejemplo guiado", "🎮 Práctica interactiva (IA)", "🏁 Evaluación para aprobar"])

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

        # IA opcional (evalúa específicamente el mini reto con criterios contables)
        ask_ai = st.checkbox("💬 Pedir feedback de IA (opcional)", key="n1_ex_ai", value=False)

        # Define la pregunta del mini reto para que quede claro en el validador (opcional, para diagnóstico/consistencia)
        mini_reto = (
            "¿Qué ocurriría con el **Costo de la Mercancía Vendida (CMV)** si no hubiera devoluciones en compras "
            "y el inventario final fuera muy pequeño?"
        )
        criterios_mini = (
            "1) Si no hay devoluciones en compras → las compras no se reducen (se mantienen altas).\n"
            "2) Inventario final muy pequeño → se RESTA MUY POCO al cierre, por tanto el **CMV AUMENTA**."
        )

        if st.button("💬 Comentar", key="n1_ex_fb"):
            if ask_ai:
                with st.spinner("Evaluando con IA…"):
                    ok, comentario, retro = eval_ia_explicacion(mini_reto, criterios_mini, razonamiento)

                with st.expander("💬 Resultado de la IA (mini reto)"):
                    st.markdown(f"**Pregunta:** {mini_reto}")
                    st.markdown(f"**Resultado:** {'✅ Aprobado' if ok else '❌ No aprobado'}")
                    st.markdown(f"**Comentario:** {comentario}")
                    st.markdown("---")
                    st.info(f"**Retroalimentación formativa:** {retro}")

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
                with st.spinner("Evaluando con IA..."):
                    # Definimos la pregunta y los criterios de evaluación pedagógicos
                    pregunta = (
                        "En este escenario, el estudiante debe calcular correctamente el "
                        "Costo de la Mercancía Vendida (CMV) aplicando la fórmula: "
                        "CMV = Inventario inicial + Compras − Devoluciones en compras − Inventario final. "
                        "Además, debe interpretar su resultado de forma contable y conceptual."
                    )

                    criterios = (
                        "1️⃣ El valor numérico del CMV debe coincidir (dentro de la tolerancia) con el cálculo correcto.\n"
                        "2️⃣ La explicación debe reflejar comprensión de la relación entre inventario final y CMV "
                        "(menor inventario final → se resta menos → CMV más alto).\n"
                        "3️⃣ Si no hay devoluciones en compras, el estudiante debe reconocer que las compras no se reducen.\n"
                        "4️⃣ La retroalimentación debe ser amable, clara y con una analogía práctica (mochila de costos, balanza, etc.)."
                    )

                    # Llamada a la IA (usa la función ya existente)
                    ok_ia, comentario_ia, retro_ia = eval_ia_explicacion(
                        pregunta=pregunta,
                        criterios=criterios,
                        respuesta_estudiante=(
                            f"CMV calculado por el estudiante: {user_cogs:.2f}. "
                            f"Datos: Inventario inicial={inv0:.2f}, Compras={compras:.2f}, "
                            f"Devoluciones en compras={devol:.2f}, Inventario final={invf:.2f}. "
                            f"CMV correcto={correct:.2f}. "
                            f"Explicación escrita: {user_comment.strip()}"
                        )
                    )

                # Mostrar el resultado de la IA en la interfaz
                with st.expander("💬 Feedback de la IA (práctica CMV)"):
                    st.markdown(f"**Resultado:** {'✅ Aprobado' if ok_ia else '❌ No aprobado'}")
                    st.markdown(f"**Comentario:** {comentario_ia}")
                    st.markdown("---")
                    st.info(f"**Retroalimentación pedagógica:** {retro_ia}")

    # Evaluación final — 5 preguntas (2 selección múltiple, 2 cálculo, 1 abierta IA)
    with tabs[3]:
        st.subheader("Evaluación final del Nivel 1")
        st.caption("Son 5 preguntas. Apruebas con **5 de 5**.")

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
        PASS_MIN = 5

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

            st.markdown(
                """
                <style>
                .val { font-family: monospace; font-size: 15px; color: #1a73e8; font-weight: 600; }
                .calc { line-height: 1.9; font-size: 1.02rem; }
                </style>
                """,
                unsafe_allow_html=True
            )

            def fmt_plain(v, dec=0):
                try:
                    return f"{v:,.{dec}f}".replace(",", ".")
                except Exception:
                    return str(v)

            def money(v, dec=0):
                return f"${fmt_plain(v, dec)}"

            # --------- Retroalimentación formativa por pregunta (1–4) ---------
            feedback_por_pregunta = []

            def fmt_valor_html(v):
                """Devuelve el valor monetario formateado y estilizado en HTML uniforme."""
                return f"<span style='font-family:monospace;font-size:15px;color:#1a73e8;font-weight:600;'> {peso(v)} </span>"

            if not ok1:
                feedback_por_pregunta.append(
                    ("1) Fórmula correcta",
                    "Recuerda la estructura en sistema periódico: "
                    "Costo de la mercancía vendida = Inventario inicial + Compras − Devoluciones en compras − Inventario final. "
                    "Las devoluciones en compras RESTAN del costo disponible y el inventario final se descuenta al cierre.")
                )
            else:
                feedback_por_pregunta.append(("1) Fórmula correcta", "¡Bien! Identificaste la fórmula sin confundir los signos."))

            if not ok2:
                feedback_por_pregunta.append(
                    ("2) Afirmación verdadera",
                    "Si el inventario final aumenta (manteniendo todo lo demás igual), se descuenta una cifra mayor al cierre, "
                    "por lo que el costo de la mercancía vendida DISMINUYE. Por eso la afirmación correcta es la que indica esa relación inversa.")
                )
            else:
                feedback_por_pregunta.append(("2) Afirmación verdadera", "Correcto: inventario final más alto ⇒ menor costo de la mercancía vendida."))

            # --- Textos HTML formateados para P3 y P4 ---
            texto_p3 = f"""
            <div class='calc'>
            Vuelve a aplicar la fórmula con los datos:<br>
            <span class='val'>{money(P3_invI)}</span> &plus;
            <span class='val'>{money(P3_comp)}</span> &minus;
            <span class='val'>{money(P3_dev)}</span> &minus;
            <span class='val'>{money(P3_invF)}</span>
            = <strong><span class='val'>{money(P3_CORRECTO)}</span></strong>.<br>
            <em>Error típico:</em> olvidar restar las devoluciones en compras.
            </div>
            """

            texto_p4 = f"""
            <div class='calc'>
            Despeja el Inventario final desde la fórmula:<br>
            Inventario final = Inventario inicial &plus; Compras &minus; Devoluciones en compras &minus; Costo de la mercancía vendida.<br>
            Con los datos:<br>
            <span class='val'>{money(P4_invI)}</span> &plus;
            <span class='val'>{money(P4_comp)}</span> &minus;
            <span class='val'>{money(P4_dev)}</span> &minus;
            <span class='val'>{money(P4_cmv)}</span>
            = <strong><span class='val'>{money(P4_CORRECTO)}</span></strong>.<br>
            <em>Error típico:</em> cambiar signos al despejar.
            </div>
            """

            if not ok3:
                feedback_por_pregunta.append(("3) Cálculo directo", texto_p3))
            else:
                feedback_por_pregunta.append(("3) Cálculo directo", "Cálculo correcto y bien aplicado el orden de operaciones."))

            if not ok4:
                feedback_por_pregunta.append(("4) Cálculo inverso", texto_p4))
            else:
                feedback_por_pregunta.append(("4) Cálculo inverso", "¡Bien! Despejaste correctamente el inventario final."))


            # --------- Mostrar panel de retroalimentación ---------
            with st.expander("🧠 Retroalimentación formativa por pregunta"):
                for titulo, nota in feedback_por_pregunta:
                    st.markdown(f"**{titulo}**", unsafe_allow_html=True)
                    st.markdown(nota, unsafe_allow_html=True)
                    st.markdown("---")


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
        st.subheader("Teoría · Métodos de valoración de inventarios (PEPS, UEPS y Promedio)")

        intro = """
    En contabilidad, los inventarios representan bienes disponibles para la venta y, al mismo tiempo, **costos acumulados** que impactan la utilidad del período. Valorar correctamente el inventario permite determinar con precisión el **Costo de la Mercancía Vendida (CMV)** y analizar la rentabilidad del negocio.
    """
        st.markdown(intro)

        st.markdown("### ⚖️ ¿Por qué existen distintos métodos?")
        st.markdown(
            "Los precios de compra cambian con el tiempo (inflación, descuentos, logística). "
            "Por eso, cada método responde a la pregunta: **¿qué costo asigno a lo vendido y qué costo queda en el inventario final?**  \n"
            "En Colombia, el **Estándar para Pymes (Sección 13)** permite tres enfoques:"
        )

        st.markdown("---")
        st.markdown("### 1) PEPS (Primero en Entrar, Primero en Salir)")
        st.markdown(
            "- **Idea básica:** salen primero las unidades que entraron primero; el inventario final queda con costos **más recientes**.  \n"
            "- **Efecto típico con precios al alza:** **CMV menor** y **mayor utilidad bruta** (porque se usan costos antiguos en las salidas).  \n"
            "- **Analogía:** una estantería donde se entregan primero los productos más viejos; los nuevos quedan en la repisa y valoran el inventario final."
        )

        st.markdown("---")
        st.markdown("### 2) UEPS (Último en Entrar, Primero en Salir)")
        st.markdown(
            "- **Idea básica:** salen primero las unidades que entraron de último; el inventario final queda con costos **más antiguos**.  \n"
            "- **Efecto típico con precios al alza:** **CMV mayor** y **menor utilidad bruta** (porque se usan costos recientes en las salidas).  \n"
            "- **Importante en Colombia:** el UEPS **no está permitido fiscalmente**; se usa para análisis internos o ejercicios académicos.  \n"
            "- **Analogía:** una pila de sacos donde tomas el de arriba (el más nuevo) y los de abajo (antiguos) permanecen en el inventario."
        )

        st.markdown("---")
        st.markdown("### 3) Promedio Ponderado")
        st.markdown(
            "- **Idea básica:** mezcla los costos de los lotes disponibles y calcula un **costo promedio por unidad**, que se usa para las salidas y el inventario final."
        )
        st.markdown("**Fórmula:**")
        st.latex(r"\text{Costo promedio ponderado}=\frac{\text{Costo total disponible}}{\text{Unidades totales disponibles}}")
        st.markdown(
            "- **Efecto contable:** **suaviza** la volatilidad de precios y es muy utilizado por su equilibrio entre **simplicidad** y **razonabilidad**.  \n"
            "- **Analogía:** una “olla de costos”: al vender, cada unidad se sirve con “una cucharada” de ese promedio."
        )

        st.markdown("---")
        st.markdown("### 💡 Para decidir")
        st.markdown(
            "Cada método ofrece una mirada distinta del costo y la utilidad. La decisión debe ser **coherente con la operación del negocio** y **consistente en el tiempo**. "
            "Recuerda: el método elegido afecta el **CMV, la utilidad** y la **carga tributaria**; por eso, comprender su efecto es clave para la toma de decisiones."
        )

        with st.expander("🔊 Escuchar explicación"):
            full_text = "\n\n".join([intro,
                "¿Por qué existen distintos métodos? Los precios de compra cambian con el tiempo...",
                "PEPS: idea básica, efecto con precios al alza y analogía de estantería.",
                "UEPS: idea básica, efecto con precios al alza, nota fiscal en Colombia y analogía de pila de sacos.",
                "Promedio Ponderado: idea básica, fórmula y analogía de olla de costos.",
                "Para decidir: coherencia del método, consistencia y efectos sobre CMV y utilidad."
            ])
            speak_block(full_text, key_prefix="teo-n2", lang_hint="es")

    with tabs[1]:
        st.subheader("KARDEX dinámico por método (PP · PEPS · UEPS)")

        # ===== Helpers internos KARDEX =====
        def _fmt_money(v):
            try:
                return peso(float(v))
            except Exception:
                return str(v)

        def _consume_layers(layers, qty_out):
            remaining = qty_out
            cost = 0.0
            det = []
            new_layers = []
            for qty, pu in layers:
                if remaining <= 0:
                    new_layers.append([qty, pu])
                    continue
                take = min(qty, remaining)
                if take > 0:
                    det.append({"qty": take, "pu": pu, "total": take * pu})
                    cost += take * pu
                    qty_rest = qty - take
                    remaining -= take
                    if qty_rest > 0:
                        new_layers.append([qty_rest, pu])
                else:
                    new_layers.append([qty, pu])
            return cost, det, new_layers, remaining

        def _kardex_two_ops(method_name, inv0_u, inv0_pu, comp_u, comp_pu, venta_u):
            cols = pd.MultiIndex.from_tuples([
                ("", "Fecha"), ("", "Descripción"),
                ("Entrada", "Cantidad"), ("Entrada", "Precio"), ("Entrada", "Total"),
                ("Salida", "Cantidad"), ("Salida", "Precio"), ("Salida", "Total"),
                ("Saldo", "Cantidad"), ("Saldo", "Precio"), ("Saldo", "Total"),
            ])

            rows = []

            # --- Fila 1: Saldo inicial ---
            saldo_layers = []
            if inv0_u > 0:
                saldo_layers = [[float(inv0_u), float(inv0_pu)]]
            saldo_qty = sum(q for q, _ in saldo_layers)
            saldo_val = sum(q * p for q, p in saldo_layers)
            saldo_pu = (saldo_val / saldo_qty) if saldo_qty > 0 else 0.0

            rows.append([
                "Día 1", "Saldo inicial",
                "", "", "",
                "", "", "",
                int(saldo_qty), round(saldo_pu, 2), round(saldo_val, 2)
            ])

            explain_lines = [f"- **Saldo inicial**: {int(inv0_u)} u @ {_fmt_money(inv0_pu)} → Saldo: {int(saldo_qty)} u, {_fmt_money(saldo_val)}."]

            # --- Fila 2: Compra ---
            entrada_total = comp_u * comp_pu
            if method_name == "Promedio Ponderado":
                new_qty = saldo_qty + comp_u
                new_val = saldo_val + entrada_total
                new_pu  = (new_val / new_qty) if new_qty > 0 else 0.0
                saldo_layers = [[new_qty, new_pu]]
                saldo_qty, saldo_val, saldo_pu = new_qty, new_val, new_pu
                rows.append([
                    "Día 2", "Compra",
                    int(comp_u), round(comp_pu, 2), round(entrada_total, 2),
                    "", "", "",
                    int(saldo_qty), round(saldo_pu, 2), round(saldo_val, 2)
                ])
                explain_lines.append(
                    f"- **Compra**: +{int(comp_u)} u @ {_fmt_money(comp_pu)}. Nuevo promedio: {_fmt_money(saldo_pu)} con {int(saldo_qty)} u en saldo."
                )
            else:
                saldo_layers.append([float(comp_u), float(comp_pu)])
                saldo_qty = sum(q for q, _ in saldo_layers)
                saldo_val = sum(q * p for q, p in saldo_layers)
                saldo_pu = (saldo_val / saldo_qty) if saldo_qty > 0 else 0.0
                rows.append([
                    "Día 2", "Compra",
                    int(comp_u), round(comp_pu, 2), round(entrada_total, 2),
                    "", "", "",
                    int(saldo_qty), round(saldo_pu, 2), round(saldo_val, 2)
                ])
                capas_txt = " · ".join([f"{int(q)}u@{_fmt_money(p)}" for q, p in saldo_layers])
                explain_lines.append(f"- **Compra**: +{int(comp_u)} u @ {_fmt_money(comp_pu)}. Capas ahora: {capas_txt}.")

            # --- Fila 3: Venta ---
            venta_total = 0.0
            salida_pu_mostrar = 0.0
            det_salida = []

            if venta_u > 0 and saldo_qty > 0:
                if method_name == "Promedio Ponderado":
                    salida_pu_mostrar = saldo_pu
                    venta_total = min(venta_u, saldo_qty) * salida_pu_mostrar
                    new_qty = max(saldo_qty - venta_u, 0)
                    new_val = max(saldo_val - venta_total, 0.0)
                    new_pu  = (new_val / new_qty) if new_qty > 0 else 0.0
                    saldo_layers = [[new_qty, new_pu]] if new_qty > 0 else []
                    saldo_qty, saldo_val, saldo_pu = new_qty, new_val, new_pu
                    det_salida = [{"qty": min(venta_u, new_qty + venta_u), "pu": salida_pu_mostrar, "total": venta_total}]
                    explain_lines.append(
                        f"- **Venta**: {int(venta_u)} u al costo promedio {_fmt_money(salida_pu_mostrar)} → **CMV**: {_fmt_money(venta_total)}. "
                        f"Saldo: {int(saldo_qty)} u, {_fmt_money(saldo_val)}."
                    )
                else:
                    layers = saldo_layers[:] if method_name == "PEPS (FIFO)" else saldo_layers[::-1]
                    venta_total, det_salida, layers_after, remaining = _consume_layers(layers, venta_u)
                    if method_name == "PEPS (FIFO)":
                        saldo_layers = layers_after
                    else:  # UEPS
                        saldo_layers = layers_after[::-1]
                    saldo_qty = sum(q for q, _ in saldo_layers)
                    saldo_val = sum(q * p for q, p in saldo_layers)
                    saldo_pu = (saldo_val / saldo_qty) if saldo_qty > 0 else 0.0
                    salida_pu_mostrar = (venta_total / venta_u) if venta_u > 0 else 0.0

                    if det_salida:
                        det_txt = " + ".join([f"{int(d['qty'])}u@{_fmt_money(d['pu'])}={_fmt_money(d['total'])}" for d in det_salida])
                        capas_txt = (" · ".join([f"{int(q)}u@{_fmt_money(p)}" for q, p in saldo_layers]) if saldo_layers else "0 u")
                        explain_lines.append(
                            f"- **Venta**: {int(venta_u)} u → {det_txt} ⇒ **CMV**: {_fmt_money(venta_total)}. Saldo: {capas_txt}."
                        )
                    else:
                        explain_lines.append("- **Venta**: no hay consumo (sin saldo).")

            rows.append([
                "Día 3", "Venta",
                "", "", "",
                int(venta_u) if venta_u > 0 else "", round(salida_pu_mostrar, 2) if venta_u > 0 else "", round(venta_total, 2) if venta_u > 0 else "",
                int(saldo_qty), round(saldo_pu, 2), round(saldo_val, 2)
            ])

            df = pd.DataFrame(rows, columns=cols)
            explain_md = "\n".join(explain_lines)
            return df, explain_md

        # ===== Controles del ejemplo (INSUMOS) — Layout narrativo por días =====
        st.markdown("#### Parámetros del escenario")

        # — Método en una sola fila —
        with st.container():
            st.markdown("**Método de valoración**")
            metodo = st.selectbox(
                "Método",
                ["Promedio Ponderado", "PEPS (FIFO)", "UEPS (LIFO)"],
                key="n2_kardex_met",
                label_visibility="collapsed"
            )

        # — Día 1: Saldo inicial —
        st.markdown("📦 **Día 1.** La empresa reporta un **saldo inicial del inventario** de:")
        c1a, c1b = st.columns([1,1], gap="small")
        with c1a:
            inv0_u = st.number_input(
                "Cantidades iniciales",
                min_value=0, value=100, step=10,
                key="n2_kx_inv_u"
            )
        with c1b:
            inv0_pu = st.number_input(
                "Costo unitario",
                min_value=0.0, value=10.0, step=0.5,
                key="n2_kx_inv_pu"
            )

        # — Día 2: Compra —
        st.markdown("🛒 **Día 2.** La empresa realizó una **compra** de:")
        c2a, c2b = st.columns([1,1], gap="small")
        with c2a:
            comp_u = st.number_input(
                "Compra (unidades)",
                min_value=0, value=60, step=10,
                key="n2_kx_comp_u"
            )
        with c2b:
            comp_pu = st.number_input(
                "Costo de la compra",
                min_value=0.0, value=12.0, step=0.5,
                key="n2_kx_comp_pu"
            )

        # — Día 3: Venta —
        st.markdown("💰 **Día 3.** La empresa realizó una **venta** de:")
        c3a, _ = st.columns([1,1], gap="small")
        with c3a:
            venta_u = st.number_input(
                "Venta (unidades)",
                min_value=0, value=120, step=10,
                key="n2_kx_venta_u"
            )


        # =========================
        # 🎬 DEMOSTRACIÓN NARRADA (PEPS/UEPS con filas por capas)
        # =========================
        st.markdown("---")
        st.markdown("### 🎬 Demostración narrada: llenado del KARDEX por método y capas")

        c_demo_a, c_demo_b = st.columns([1,1])
        with c_demo_a:
            narr_speed = st.slider("Velocidad de narración", 0.75, 1.50, 1.00, 0.05)
        with c_demo_b:
            narr_muted = st.toggle("Silenciar voz", value=False)

        def _sum_layers(layers):
            q = sum(q for q,_ in layers)
            v = sum(q*p for q,p in layers)
            pu = (v/q) if q>0 else 0.0
            return q, pu, v

        def _consume_layers_detail(layers, qty_out, fifo=True):
            """Devuelve (detalles_salida, layers_after) donde detalles_salida es lista de (qty, pu, total)."""
            order = layers[:] if fifo else layers[::-1]
            remaining = qty_out
            details = []
            after = []
            for (q,pu) in order:
                if remaining <= 0:
                    after.append([q,pu]); continue
                take = min(q, remaining)
                if take > 0:
                    details.append([take, pu, take*pu])
                    q_rest = q - take
                    remaining -= take
                    if q_rest > 0:
                        after.append([q_rest, pu])
                else:
                    after.append([q,pu])
            # Reconstruir en orden lógico original
            layers_after = after if fifo else after[::-1]
            return details, layers_after

        def compute_rows_and_script(method_name, inv0_u, inv0_pu, comp_u, comp_pu, venta_u):
            rows = []   # cada fila: dict con keys: fecha, desc, ent_q, ent_pu, ent_tot, sal_q, sal_pu, sal_tot, sdo_q, sdo_pu, sdo_tot
            script = [] # cada paso: {"title","text","actions":[{"row":i,"cell":"rX","money":bool,"val":v}, ...]}

            # Día 1: Saldo inicial (1 fila)
            layers = [[float(inv0_u), float(inv0_pu)]] if inv0_u > 0 else []
            s_q, s_pu, s_v = _sum_layers(layers)
            rows.append({
                "fecha":"Día 1", "desc":"Saldo inicial",
                "ent_q":"","ent_pu":"","ent_tot":"",
                "sal_q":"","sal_pu":"","sal_tot":"",
                "sdo_q":int(s_q), "sdo_pu":round(s_pu,2), "sdo_tot":round(s_v,2)
            })
            script.append({
                "title":"Paso 1 · Saldo inicial",
                "text":"Registramos cantidad y costo unitario del inventario existente. Calculamos el saldo: cantidad × precio.",
                "actions":[
                    {"row":0,"cell":"sdo_q","money":False,"val":int(s_q)},
                    {"row":0,"cell":"sdo_pu","money":True, "val":round(s_pu,2)},
                    {"row":0,"cell":"sdo_tot","money":True,"val":round(s_v,2)},
                ]
            })

            # Día 2: Compra
            if method_name == "Promedio Ponderado":
                # --------- SOLO UNA FILA (Compra) ---------
                ent_tot = comp_u * comp_pu
                # saldo nuevo (promedio)
                new_q = s_q + comp_u
                new_v = s_v + ent_tot
                new_p = (new_v/new_q) if new_q > 0 else 0.0
                layers = [[new_q, new_p]]
                s_q, s_pu, s_v = _sum_layers(layers)

                rows.append({
                    "fecha":"Día 2", "desc":"Compra",
                    "ent_q":int(comp_u), "ent_pu":round(comp_pu,2), "ent_tot":round(ent_tot,2),
                    "sal_q":"","sal_pu":"","sal_tot":"",
                    "sdo_q":int(s_q), "sdo_pu":round(s_pu,2), "sdo_tot":round(s_v,2)
                })
                script.append({
                    "title":"Paso 2 · Compra y nuevo promedio",
                    "text":"Registramos la compra y recalculamos el costo promedio: (valor saldo anterior + valor entrada) / (unidades anteriores + unidades de entrada).",
                    "actions":[
                        {"row":1,"cell":"ent_q","money":False,"val":int(comp_u)},
                        {"row":1,"cell":"ent_pu","money":True, "val":round(comp_pu,2)},
                        {"row":1,"cell":"ent_tot","money":True,"val":round(ent_tot,2)},
                        {"row":1,"cell":"sdo_q","money":False,"val":int(s_q)},
                        {"row":1,"cell":"sdo_pu","money":True, "val":round(s_pu,2)},
                        {"row":1,"cell":"sdo_tot","money":True,"val":round(s_v,2)},
                    ]
                })
                start_sale_row_index = 2

            else:
                # --------- PEPS / UEPS: dos filas en Día 2 ---------
                # Fila 1: “Saldo (día 1)” (copia del saldo previo; mantiene costo por capa)
                rows.append({
                    "fecha":"Día 2", "desc":"Saldo (día 1)",
                    "ent_q":"","ent_pu":"","ent_tot":"",
                    "sal_q":"","sal_pu":"","sal_tot":"",
                    "sdo_q":int(s_q), "sdo_pu":round(s_pu,2), "sdo_tot":round(s_v,2)
                })
                script.append({
                    "title":"Paso 2A · Saldo que viene del día 1",
                    "text":"Antes de registrar la compra, mostramos el saldo existente y su costo por capa.",
                    "actions":[
                        {"row":1,"cell":"sdo_q","money":False,"val":int(s_q)},
                        {"row":1,"cell":"sdo_pu","money":True, "val":round(s_pu,2)},
                        {"row":1,"cell":"sdo_tot","money":True,"val":round(s_v,2)},
                    ]
                })

                # Fila 2: “Compra” — entrada EXACTA 60 u @ 12; en Saldo SOLO la capa comprada
                ent_tot = comp_u * comp_pu                      # p.ej., 60 * 12 = 720
                layers.append([float(comp_u), float(comp_pu)])  # agrega la nueva capa a 12 (capas se mantienen separadas)
                rows.append({
                    "fecha":"Día 2", "desc":"Compra",
                    "ent_q":int(comp_u), "ent_pu":round(comp_pu,2), "ent_tot":round(ent_tot,2),
                    "sal_q":"","sal_pu":"","sal_tot":"",
                    "sdo_q":int(comp_u), "sdo_pu":round(comp_pu,2), "sdo_tot":round(ent_tot,2)  # SOLO la capa comprada
                })
                script.append({
                    "title":"Paso 2B · Registro de la compra",
                    "text":"Registramos la entrada con su costo unitario (sin promediar). En la columna Saldo de esta fila mostramos únicamente la nueva capa comprada.",
                    "actions":[
                        {"row":2,"cell":"ent_q","money":False,"val":int(comp_u)},
                        {"row":2,"cell":"ent_pu","money":True, "val":round(comp_pu,2)},
                        {"row":2,"cell":"ent_tot","money":True,"val":round(ent_tot,2)},
                        {"row":2,"cell":"sdo_q","money":False,"val":int(comp_u)},
                        {"row":2,"cell":"sdo_pu","money":True, "val":round(comp_pu,2)},
                        {"row":2,"cell":"sdo_tot","money":True,"val":round(ent_tot,2)},
                    ]
                })
                start_sale_row_index = 3  # porque ya tenemos 3 filas antes de la venta

            # Día 3: Venta
            if venta_u > 0 and s_q > 0:
                if method_name == "Promedio Ponderado":
                    # Una sola fila de venta con el promedio vigente
                    sal_q = min(venta_u, int(s_q))
                    sal_pu = layers[0][1] if layers else 0.0
                    sal_tot = sal_q * sal_pu
                    # saldo tras la venta
                    new_q = s_q - sal_q
                    new_v = s_v - sal_tot
                    new_p = (new_v/new_q) if new_q > 0 else 0.0
                    layers = [[new_q, new_p]] if new_q > 0 else []
                    s_q, s_pu, s_v = _sum_layers(layers)

                    rows.append({
                        "fecha":"Día 3", "desc":"Venta",
                        "ent_q":"","ent_pu":"","ent_tot":"",
                        "sal_q":int(sal_q), "sal_pu":round(sal_pu,2), "sal_tot":round(sal_tot,2),
                        "sdo_q":int(s_q), "sdo_pu":round(s_pu,2), "sdo_tot":round(s_v,2)
                    })
                    script.append({
                        "title":"Paso 3 · Venta (Promedio)",
                        "text":"Calculamos el CMV con el costo promedio vigente y actualizamos el saldo.",
                        "actions":[
                            {"row":start_sale_row_index,"cell":"sal_q","money":False,"val":int(sal_q)},
                            {"row":start_sale_row_index,"cell":"sal_pu","money":True, "val":round(sal_pu,2)},
                            {"row":start_sale_row_index,"cell":"sal_tot","money":True,"val":round(sal_tot,2)},
                            {"row":start_sale_row_index,"cell":"sdo_q","money":False,"val":int(s_q)},
                            {"row":start_sale_row_index,"cell":"sdo_pu","money":True, "val":round(s_pu,2)},
                            {"row":start_sale_row_index,"cell":"sdo_tot","money":True,"val":round(s_v,2)},
                        ]
                    })
                else:
                    # PEPS/UEPS: divisar venta por capas (tramos)
                    fifo = (method_name == "PEPS (FIFO)")
                    sale_details, layers_after = _consume_layers_detail(layers, venta_u, fifo=fifo)

                    acc_row = start_sale_row_index
                    running_layers = [l[:] for l in layers]  # copia para ir actualizando saldo por tramo
                    metodo_tag = "PEPS" if fifo else "UEPS"

                    for i, (q_take, pu_take, tot_take) in enumerate(sale_details, start=1):
                        # Actualiza running_layers consumiendo este tramo para mostrar saldo tras CADA tramo
                        _tmp_details, running_layers = _consume_layers_detail(running_layers, q_take, fifo=fifo)
                        rq, rpu, rv = _sum_layers(running_layers)

                        rows.append({
                            "fecha":"Día 3", "desc": f"Venta tramo {i} ({metodo_tag})",
                            "ent_q":"","ent_pu":"","ent_tot":"",
                            "sal_q":int(q_take), "sal_pu":round(pu_take,2), "sal_tot":round(tot_take,2),
                            "sdo_q":int(rq), "sdo_pu":round(rpu,2), "sdo_tot":round(rv,2)
                        })
                        script.append({
                            "title": f"Paso 3 · Venta (tramo {i})",
                            "text":"Consumimos unidades de la capa correspondiente: en PEPS salen primero las más antiguas; en UEPS, las últimas en entrar. Actualizamos el saldo tras el tramo.",
                            "actions":[
                                {"row":acc_row,"cell":"sal_q","money":False,"val":int(q_take)},
                                {"row":acc_row,"cell":"sal_pu","money":True, "val":round(pu_take,2)},
                                {"row":acc_row,"cell":"sal_tot","money":True,"val":round(tot_take,2)},
                                {"row":acc_row,"cell":"sdo_q","money":False,"val":int(rq)},
                                {"row":acc_row,"cell":"sdo_pu","money":True, "val":round(rpu,2)},
                                {"row":acc_row,"cell":"sdo_tot","money":True,"val":round(rv,2)},
                            ]
                        })
                        acc_row += 1

                    # Actualiza layers finales tras toda la venta
                    layers = layers_after
            else:
                # Sin venta o sin saldo
                rows.append({
                    "fecha":"Día 3", "desc":"Venta",
                    "ent_q":"","ent_pu":"","ent_tot":"",
                    "sal_q":"","sal_pu":"","sal_tot":"",
                    "sdo_q":int(s_q), "sdo_pu":round(s_pu,2), "sdo_tot":round(s_v,2)
                })
                script.append({
                    "title":"Paso 3 · Venta",
                    "text":"No hay venta o no hay saldo para consumir; el inventario permanece igual.",
                    "actions":[
                        {"row":start_sale_row_index,"cell":"sdo_q","money":False,"val":int(s_q)},
                        {"row":start_sale_row_index,"cell":"sdo_pu","money":True, "val":round(s_pu,2)},
                        {"row":start_sale_row_index,"cell":"sdo_tot","money":True,"val":round(s_v,2)},
                    ]
                })

            return rows, script


        demo_rows, demo_script = compute_rows_and_script(metodo, inv0_u, inv0_pu, comp_u, comp_pu, venta_u)

        import json as _json
        html_demo_template = """
        <style>
        .kx {border-collapse:collapse;width:100%;font-size:14px;margin-bottom:6px}
        .kx th,.kx td {border:1px solid #eaeaea;padding:6px 8px;text-align:center}
        .kx thead th {background:#f8fafc;font-weight:600}
        .kx .hi {background:#fff7e6;box-shadow:inset 0 0 0 9999px rgba(255,165,0,0.08)}
        .fill {transition: background 0.3s, color 0.3s}
        .controls {display:flex;gap:8px;align-items:center;margin:6px 0}
        .badge {display:inline-block;background:#eef;border:1px solid #dde;padding:2px 8px;border-radius:12px;font-size:12px}
        .btn {padding:6px 10px; border:1px solid #ddd; background:#fafafa; cursor:pointer; border-radius:6px;}
        .btn:hover {background:#f0f0f0}
        .muted {color:#999}
        #narr {margin-top:6px;font-size:15px}
        </style>

        <div class="controls">
        <button id="playDemo" class="btn">▶️ Reproducir demo</button>
        <button id="resetDemo" class="btn">↺ Reiniciar</button>
        <span class="badge">%%METODO%%</span>
        </div>

        <table class="kx" id="kxtable">
        <thead>
            <tr>
            <th></th><th></th>
            <th colspan="3">Entrada</th>
            <th colspan="3">Salida</th>
            <th colspan="3">Saldo</th>
            </tr>
            <tr>
            <th>Fecha</th><th>Descripción</th>
            <th>Cantidad</th><th>Precio</th><th>Total</th>
            <th>Cantidad</th><th>Precio</th><th>Total</th>
            <th>Cantidad</th><th>Precio</th><th>Total</th>
            </tr>
        </thead>
        <tbody id="kbody"></tbody>
        </table>
        <div id="narr"></div>

        <script>
        (function(){
        const rows = %%ROWS%%;
        const script = %%SCRIPT%%;
        const metodo = "%%METODO%%";
        const narrMuted = %%MUTED%%;
        const rate = %%RATE%%;

        const tbody = document.getElementById("kbody");
        const narrDiv = document.getElementById("narr");
        const btnPlay = document.getElementById("playDemo");
        const btnReset = document.getElementById("resetDemo");

        const pesos = (v)=> {
            try { return new Intl.NumberFormat('es-CO',{style:'currency', currency:'COP', maximumFractionDigits:2}).format(v); }
            catch(e){ return "$"+(Math.round(v*100)/100).toLocaleString('es-CO'); }
        };
        const fmt = (x)=> (x===null || x===undefined || x==="") ? "" : (typeof x==="number" ? (Number.isInteger(x)? x.toString(): (Math.round(x*100)/100).toString().replace(".",",")) : x);

        function speak(text){
            return new Promise((resolve)=>{
            if (narrMuted) return resolve();
            try{
                if (window.speechSynthesis.speaking) window.speechSynthesis.cancel();
                const u = new SpeechSynthesisUtterance(text);
                const voices = window.speechSynthesis.getVoices();
                const pick = voices.find(v=>/es|spanish|mex|col/i.test((v.name+" "+v.lang))) || voices[0];
                if (pick) u.voice = pick;
                u.rate = rate; u.pitch = 1.0;
                u.onend = ()=> resolve();
                window.speechSynthesis.speak(u);
            }catch(e){ resolve(); }
            });
        }
        const sleep = (ms)=> new Promise(r=>setTimeout(r, ms));

        function buildTable(){
            tbody.innerHTML = "";
            rows.forEach((r, i)=>{
            const tr = document.createElement("tr");
            tr.id = "row"+i;
            tr.innerHTML = `
                <td>${r.fecha}</td><td>${r.desc}</td>
                <td id="r${i}_ent_q"  class="fill muted"></td>
                <td id="r${i}_ent_pu" class="fill muted"></td>
                <td id="r${i}_ent_tot"class="fill muted"></td>
                <td id="r${i}_sal_q"  class="fill muted"></td>
                <td id="r${i}_sal_pu" class="fill muted"></td>
                <td id="r${i}_sal_tot"class="fill muted"></td>
                <td id="r${i}_sdo_q"  class="fill muted"></td>
                <td id="r${i}_sdo_pu" class="fill muted"></td>
                <td id="r${i}_sdo_tot"class="fill muted"></td>
            `;
            tbody.appendChild(tr);
            });
        }
        function clearTable(){
            [...tbody.querySelectorAll("td")].forEach(td=>{
            if (td.id) { td.textContent = ""; td.classList.add("muted"); }
            });
            [...tbody.querySelectorAll("tr")].forEach(tr=> tr.classList.remove("hi"));
            narrDiv.textContent = "";
        }
        function highlightRow(i){
            [...tbody.querySelectorAll("tr")].forEach((tr, idx)=>{
            tr.classList.toggle("hi", idx === i);
            });
        }
        function fillCell(rowIdx, key, val, money=false){
            const el = document.getElementById(`r${rowIdx}_${key}`);
            if (!el) return;
            el.classList.remove("muted");
            el.style.background = "#fffbe6";
            el.style.color = "#333";
            el.textContent = money ? pesos(val) : fmt(val);
            setTimeout(()=>{ el.style.background=""; }, 300);
        }

        async function runScript(){
            clearTable();
            // Pintar título de cada paso, resaltar fila y llenar celdas en orden
            for (const step of script){
            narrDiv.textContent = step.title;
            // resaltar filas involucradas: la primera acción indica la fila
            if (step.actions && step.actions.length>0){
                highlightRow(step.actions[0].row);
            }
            // duración base proporcional al texto
            const dur = Math.max(2200, Math.min(7000, step.text.length * 55 / rate));
            const chunks = Math.max(3, step.actions.length);
            const waits = Array.from({length:chunks-1}, (_,k)=> Math.floor(dur*(k+1)/chunks));

            const pVoice = speak(step.text);

            for (let i=0;i<step.actions.length;i++){
                const a = step.actions[i];
                if (i>0){ await sleep(waits[i-1]); }
                fillCell(a.row, a.cell, a.val, !!a.money);
            }

            await pVoice;
            await sleep(200);
            }
            highlightRow(-1);
        }

        buildTable();
        btnPlay.onclick  = runScript;
        btnReset.onclick = ()=>{ clearTable(); buildTable(); };
        if (window.speechSynthesis) { window.speechSynthesis.onvoiceschanged = ()=>{}; }
        })();
        </script>
        """

        html_demo = (
            html_demo_template
            .replace("%%ROWS%%", _json.dumps(demo_rows))
            .replace("%%SCRIPT%%", _json.dumps(demo_script))
            .replace("%%METODO%%", metodo)
            .replace("%%MUTED%%", "true" if narr_muted else "false")
            .replace("%%RATE%%", str(narr_speed))
        )

        components.html(html_demo, height=250, scrolling=True)

    with tabs[2]:
        st.subheader("Práctica IA: diligencia tu propio KARDEX")
        st.caption("Selecciona un método y, si quieres, genera un escenario aleatorio. También puedes editar los valores manualmente.")

        # =========================
        # Utilidades de estado
        # =========================
        def _ensure_default_state():
            ss = st.session_state
            ss.setdefault("n2_ex_metodo", "Promedio Ponderado")
            ss.setdefault("n2_ex_inv0_u", 80)
            ss.setdefault("n2_ex_inv0_pu", 10.0)
            ss.setdefault("n2_ex_comp1_u", 40)
            ss.setdefault("n2_ex_comp1_pu", 11.0)
            ss.setdefault("n2_ex_venta_u", 90)
            ss.setdefault("n2_ex_comp2_u", 50)
            ss.setdefault("n2_ex_comp2_pu", 13.0)

        def _randomize_scenario_values():
            # Genera un escenario razonable
            import random
            inv0_u  = random.choice([60, 80, 100, 120, 150])
            inv0_pu = random.choice([8.0, 9.0, 10.0, 11.0, 12.0])
            comp1_u = random.choice([30, 40, 50, 60, 70])
            comp1_pu= random.choice([inv0_pu - 1, inv0_pu, inv0_pu + 1, inv0_pu + 2])
            venta_u = random.choice([40, 60, 90, 110, 130])
            comp2_u = random.choice([30, 40, 50, 60, 80])
            comp2_pu= random.choice([comp1_pu - 1, comp1_pu, comp1_pu + 1, comp1_pu + 2])

            ss = st.session_state
            ss["n2_ex_inv0_u"]  = inv0_u
            ss["n2_ex_inv0_pu"] = float(max(1.0, round(inv0_pu, 2)))
            ss["n2_ex_comp1_u"] = comp1_u
            ss["n2_ex_comp1_pu"]= float(max(1.0, round(comp1_pu, 2)))
            ss["n2_ex_venta_u"] = venta_u
            ss["n2_ex_comp2_u"] = comp2_u
            ss["n2_ex_comp2_pu"]= float(max(1.0, round(comp2_pu, 2)))

        def _request_randomize():
            st.session_state["n2_ex_rand_request"] = True

        _ensure_default_state()

        # Atender aleatorización ANTES de instanciar widgets
        if st.session_state.get("n2_ex_rand_request", False):
            _randomize_scenario_values()
            st.session_state.pop("n2_ex_rand_request", None)
            st.rerun()

        # =========================
        # Método independiente del ejercicio
        # =========================
        c0a, c0b = st.columns([1, 3])
        with c0a:
            ex_metodo = st.selectbox(
                "Método (ejercicio)",
                ["Promedio Ponderado", "PEPS (FIFO)", "UEPS (LIFO)"],
                key="n2_ex_metodo"
            )

        # =========================
        # Escenario editable (Día 1-4)
        # =========================
        st.markdown("#### 🎯 Escenario del ejercicio")

        # Día 1
        st.markdown("**Día 1.** La empresa reporta un saldo inicial de inventario de:")
        c1a, c1b = st.columns([1, 1])
        with c1a:
            inv0_u_ex  = st.number_input("Cantidad (u) — Día 1", min_value=0, step=1, key="n2_ex_inv0_u")
        with c1b:
            inv0_pu_ex = st.number_input("Costo unitario — Día 1", min_value=0.0, step=0.1, key="n2_ex_inv0_pu")

        # Día 2
        st.markdown("**Día 2.** La empresa realizó una **compra** de:")
        c2a, c2b = st.columns([1, 1])
        with c2a:
            comp1_u  = st.number_input("Cantidad (u) — Día 2", min_value=0, step=1, key="n2_ex_comp1_u")
        with c2b:
            comp1_pu = st.number_input("Costo unitario — Día 2", min_value=0.0, step=0.1, key="n2_ex_comp1_pu")

        # Día 3
        st.markdown("**Día 3.** La empresa realizó una **venta** de:")
        venta_ex_u = st.number_input("Cantidad vendida (u) — Día 3", min_value=0, step=1, key="n2_ex_venta_u")

        # Día 4
        st.markdown("**Día 4.** La empresa realizó otra **compra** de:")
        c4a, c4b = st.columns([1, 1])
        with c4a:
            comp2_u  = st.number_input("Cantidad (u) — Día 4", min_value=0, step=1, key="n2_ex_comp2_u")
        with c4b:
            comp2_pu = st.number_input("Costo unitario — Día 4", min_value=0.0, step=0.1, key="n2_ex_comp2_pu")

        # Botón aleatorio al final (no modifica widgets directamente)
        st.button(
            "🎲 Generar escenario aleatorio",
            key="n2_ex_rand_btn",
            on_click=_request_randomize
        )

        # =========================
        # Helpers de cálculo
        # =========================
        def _sum_layers(layers):
            """layers: [[qty, pu], ...] → retorna (qty_total, pu_prom, val_total)"""
            q = sum(q for q, _ in layers)
            v = sum(q * p for q, p in layers)
            pu = (v / q) if q > 0 else 0.0
            return q, pu, v

        def _consume_layers_detail(layers, qty_out, fifo=True):
            """
            layers: [[qty, pu], ...]
            qty_out: cantidad a vender
            fifo=True (PEPS) o False (UEPS)
            Retorna:
            sale_details = [(q_take, pu_take, tot_take), ...]
            layers_after = capas remanentes en el orden natural (FIFO)
            """
            order = layers[:] if fifo else layers[::-1]
            remaining = qty_out
            sale_details = []
            updated = []
            for q, pu in order:
                if remaining <= 0:
                    updated.append([q, pu]); continue
                take = min(q, remaining)
                if take > 0:
                    sale_details.append((take, pu, take * pu))
                    rest = q - take
                    remaining -= take
                    if rest > 0:
                        updated.append([rest, pu])
                else:
                    updated.append([q, pu])
            final_layers = updated if fifo else updated[::-1]
            return sale_details, final_layers

        # =========================
        # Construcción PARAMÉTRICA de filas esperadas
        # =========================
        def build_expected_rows(method_name):
            """
            Devuelve una lista de dicts 'row' con columnas:
            Fecha, Descripción,
            Entrada_cant, Entrada_pu, Entrada_total,
            Salida_cant,  Salida_pu,  Salida_total,
            Saldo_cant,   Saldo_pu,   Saldo_total
            Estructura:
            - Promedio: 4 filas (Día 1, Día 2 compra, Día 3 venta, Día 4 compra 2)
            - PEPS/UEPS: Día 1; Día 2 (Saldo día 1) + (Compra); Día 3 (Venta tramo i ... n);
                        Día 4 (Compra 2)
                * En las filas de Compra (día 2 y día 4) el Saldo muestra SOLO la capa comprada.
                * En cada tramo de venta, el Saldo muestra el saldo tras ese tramo.
            """
            rows = []

            # Día 1: Saldo inicial
            layers = [[float(inv0_u_ex), float(inv0_pu_ex)]] if inv0_u_ex > 0 else []
            s_q, s_p, s_v = _sum_layers(layers)
            rows.append({
                "Fecha":"Día 1", "Descripción":"Saldo inicial",
                "Entrada_cant":"", "Entrada_pu":"", "Entrada_total":"",
                "Salida_cant":"",  "Salida_pu":"",  "Salida_total":"",
                "Saldo_cant": s_q, "Saldo_pu": round(s_p,2), "Saldo_total": round(s_v,2)
            })

            # Día 2: Compras
            if method_name == "Promedio Ponderado":
                # Compra 1 (promediada)
                ent_tot = comp1_u * comp1_pu
                q_new = s_q + comp1_u
                v_new = s_v + ent_tot
                p_new = (v_new / q_new) if q_new > 0 else 0.0
                layers = [[q_new, p_new]]
                s_q, s_p, s_v = _sum_layers(layers)
                rows.append({
                    "Fecha":"Día 2", "Descripción":"Compra 1",
                    "Entrada_cant": comp1_u, "Entrada_pu": round(comp1_pu,2), "Entrada_total": round(ent_tot,2),
                    "Salida_cant":"", "Salida_pu":"", "Salida_total":"",
                    "Saldo_cant": s_q, "Saldo_pu": round(s_p,2), "Saldo_total": round(s_v,2)
                })
            else:
                # PEPS/UEPS: dos filas en día 2
                # Fila 2A: Saldo (día 1)
                rows.append({
                    "Fecha":"Día 2", "Descripción":"Saldo (día 1)",
                    "Entrada_cant":"", "Entrada_pu":"", "Entrada_total":"",
                    "Salida_cant":"",  "Salida_pu":"",  "Salida_total":"",
                    "Saldo_cant": s_q, "Saldo_pu": round(s_p,2), "Saldo_total": round(s_v,2)
                })
                # Fila 2B: Compra 1 (saldo muestra SOLO la capa comprada)
                ent_tot = comp1_u * comp1_pu
                layers.append([float(comp1_u), float(comp1_pu)])
                rows.append({
                    "Fecha":"Día 2", "Descripción":"Compra 1",
                    "Entrada_cant": comp1_u, "Entrada_pu": round(comp1_pu,2), "Entrada_total": round(ent_tot,2),
                    "Salida_cant":"", "Salida_pu":"", "Salida_total":"",
                    "Saldo_cant": comp1_u, "Saldo_pu": round(comp1_pu,2), "Saldo_total": round(ent_tot,2)
                })
                # s_q/s_p/s_v se mantienen como el agregado total, pero para filas mostramos lo pedido.

            # Día 3: Venta
            if method_name == "Promedio Ponderado":
                if s_q > 0 and venta_ex_u > 0:
                    sale_q  = min(venta_ex_u, s_q)
                    sale_pu = layers[0][1] if layers else 0.0
                    sale_tot= sale_q * sale_pu
                    q2 = s_q - sale_q
                    v2 = s_v - sale_tot
                    p2 = (v2 / q2) if q2 > 0 else 0.0
                    layers = [[q2, p2]] if q2 > 0 else []
                    s_q, s_p, s_v = _sum_layers(layers)
                    rows.append({
                        "Fecha":"Día 3", "Descripción":"Venta",
                        "Entrada_cant":"", "Entrada_pu":"", "Entrada_total":"",
                        "Salida_cant": sale_q, "Salida_pu": round(sale_pu,2), "Salida_total": round(sale_tot,2),
                        "Saldo_cant": s_q, "Saldo_pu": round(s_p,2), "Saldo_total": round(s_v,2)
                    })
                else:
                    rows.append({
                        "Fecha":"Día 3", "Descripción":"Venta",
                        "Entrada_cant":"", "Entrada_pu":"", "Entrada_total":"",
                        "Salida_cant":"", "Salida_pu":"", "Salida_total":"",
                        "Saldo_cant": s_q, "Saldo_pu": round(s_p,2), "Saldo_total": round(s_v,2)
                    })
            else:
                fifo = (method_name == "PEPS (FIFO)")
                sale_details, layers_after = _consume_layers_detail(layers, venta_ex_u, fifo=fifo)
                # mostrar una fila por tramo
                running_layers = [l[:] for l in layers]
                metodo_tag = "PEPS" if fifo else "UEPS"
                for i, (q_take, pu_take, tot_take) in enumerate(sale_details, start=1):
                    # actualizar saldo tras este tramo
                    sub_details, running_layers = _consume_layers_detail(running_layers, q_take, fifo=fifo)
                    rq, rpu, rv = _sum_layers(running_layers)
                    rows.append({
                        "Fecha":"Día 3", "Descripción": f"Venta tramo {i} ({metodo_tag})",
                        "Entrada_cant":"", "Entrada_pu":"", "Entrada_total":"",
                        "Salida_cant": q_take, "Salida_pu": round(pu_take,2), "Salida_total": round(tot_take,2),
                        "Saldo_cant": rq, "Saldo_pu": round(rpu,2), "Saldo_total": round(rv,2)
                    })
                layers = layers_after
                s_q, s_p, s_v = _sum_layers(layers)

            # Día 4: Compra 2
            if method_name == "Promedio Ponderado":
                ent2_tot = comp2_u * comp2_pu
                q3 = s_q + comp2_u
                v3 = s_v + ent2_tot
                p3 = (v3 / q3) if q3 > 0 else 0.0
                layers = [[q3, p3]]
                s_q, s_p, s_v = _sum_layers(layers)
                rows.append({
                    "Fecha":"Día 4", "Descripción":"Compra 2",
                    "Entrada_cant": comp2_u, "Entrada_pu": round(comp2_pu,2), "Entrada_total": round(ent2_tot,2),
                    "Salida_cant":"", "Salida_pu":"", "Salida_total":"",
                    "Saldo_cant": s_q, "Saldo_pu": round(s_p,2), "Saldo_total": round(s_v,2)
                })
            else:
                ent2_tot = comp2_u * comp2_pu
                layers.append([float(comp2_u), float(comp2_pu)])
                # Mostrar SOLO la capa comprada en el saldo de esta fila:
                rows.append({
                    "Fecha":"Día 4", "Descripción":"Compra 2",
                    "Entrada_cant": comp2_u, "Entrada_pu": round(comp2_pu,2), "Entrada_total": round(ent2_tot,2),
                    "Salida_cant":"", "Salida_pu":"", "Salida_total":"",
                    "Saldo_cant": comp2_u, "Saldo_pu": round(comp2_pu,2), "Saldo_total": round(ent2_tot,2)
                })

            return rows

        expected_rows = build_expected_rows(ex_metodo)

        # =========================
        # Plantilla dinámica para edición
        # =========================
        def _blank_row(fecha, desc):
            return {
                "Fecha": fecha, "Descripción": desc,
                "Entrada_cant": "", "Entrada_pu": "", "Entrada_total": "",
                "Salida_cant": "",  "Salida_pu": "",  "Salida_total": "",
                "Saldo_cant": "",   "Saldo_pu": "",   "Saldo_total": ""
            }

        # Construye DataFrame con tantas filas como las esperadas (descripciones fijas)
        plant_rows = []
        for r in expected_rows:
            plant_rows.append(_blank_row(r["Fecha"], r["Descripción"]))
        plant = pd.DataFrame(plant_rows)

        st.markdown("#### ✍️ Completa la tabla (números)")
        st.caption("Escribe **valores numéricos**. Puedes dejar celdas no aplicables en blanco.")
        edited = st.data_editor(
            plant,
            use_container_width=True,
            num_rows="fixed",
            key="n2_kardex_student_table_var"
        )

        # =========================
        # Validación y feedback
        # =========================
        with st.form("n2_kardex_check_var"):
            ask_ai = st.checkbox("💬 Pedir retroalimentación de IA (procedimiento)", value=False, key="n2_kardex_ai_feedback_var")
            submitted_ex = st.form_submit_button("✅ Validar mi KARDEX")

        if submitted_ex:
            tol = 0.5

            def _to_float(x):
                try:
                    if x in (None, ""): return None
                    return float(x)
                except Exception:
                    return None

            def _near(a, b):
                if a is None or b is None: return False
                return abs(a - b) <= tol

            flags = []
            # Comparar fila a fila contra expected_rows
            for i in range(len(expected_rows)):
                user = edited.iloc[i].to_dict()
                exp  = expected_rows[i]
                ok_cells = []
                # Para cada celda numérica, si exp tiene número, validamos; si exp == "" no se exige
                for key in ["Entrada_cant","Entrada_pu","Entrada_total","Salida_cant","Salida_pu","Salida_total","Saldo_cant","Saldo_pu","Saldo_total"]:
                    exp_val = exp[key]
                    usr_val = _to_float(user.get(key, ""))
                    if exp_val == "":  # no obligatorio
                        ok = True
                    else:
                        ok = _near(usr_val, float(exp_val))
                    ok_cells.append(ok)
                ok_row = all(ok_cells)
                flags.append((f"{exp['Fecha']} · {exp['Descripción']}", ok_row))

            aciertos = sum(1 for _, ok in flags if ok)
            st.metric("Aciertos por fila", f"{aciertos}/{len(flags)}")
            for label, ok in flags:
                st.write(("✅ " if ok else "❌ ") + label)

            if aciertos == len(flags):
                st.success("¡Excelente! Tu procedimiento y saldos son coherentes con el método elegido.")
            else:
                st.warning("Hay diferencias en una o más filas. Revisa cantidades, costos unitarios y el método aplicado en cada día/tramo.")

            # Feedback IA opcional
            if ask_ai:
                # Arma una descripción compacta del intento del estudiante
                def _row_summary(idx):
                    r = edited.iloc[idx].to_dict()
                    def g(k):
                        v = _to_float(r.get(k, ""))
                        return "—" if v is None else f"{v:.2f}" if isinstance(v, float) else str(v)
                    return (f"{edited.iloc[idx]['Fecha']} {edited.iloc[idx]['Descripción']}: "
                            f"E({g('Entrada_cant')},{g('Entrada_pu')},{g('Entrada_total')}) | "
                            f"S({g('Salida_cant')},{g('Salida_pu')},{g('Salida_total')}) | "
                            f"Saldo({g('Saldo_cant')},{g('Saldo_pu')},{g('Saldo_total')})")

                intento = "\n".join(_row_summary(i) for i in range(len(expected_rows)))

                # Salidas clave para guía
                # saldo final esperado
                final_exp = expected_rows[-1]
                exp_qtyF = final_exp["Saldo_cant"] if final_exp["Saldo_cant"] != "" else None
                exp_valF = final_exp["Saldo_total"] if final_exp["Saldo_total"] != "" else None
                exp_puF  = final_exp["Saldo_pu"] if final_exp["Saldo_pu"] != "" else None

                sol_desc = (
                    f"Método: {ex_metodo}. "
                    f"Saldo final esperado: cant={exp_qtyF}, val={exp_valF}, pu={exp_puF}. "
                    f"Cantidad de filas esperadas: {len(expected_rows)}."
                )

                with st.spinner("Generando retroalimentación de IA…"):
                    fb_txt = ia_feedback(
                        "Evalúa el procedimiento de un KARDEX paso a paso. " + sol_desc +
                        "\nEl estudiante diligenció:\n" + intento +
                        "\nIndica: (1) si respeta el método (promedio/PEPS/UEPS) en cada día/tramo, "
                        "(2) posibles errores (promediar en PEPS/UEPS, usar costo equivocado en venta, no actualizar saldo), "
                        "(3) un tip memotécnico breve y aplicable."
                    )
                with st.expander("💬 Retroalimentación de la IA"):
                    st.write(fb_txt)


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

