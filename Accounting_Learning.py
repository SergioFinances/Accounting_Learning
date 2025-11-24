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
import json as _json
from datetime import datetime, timezone
import time

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
import json, re
import pandas as _pd

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
from openai import BadRequestError  # <-- agrega esto

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

PRIMARY_MODEL  = "deepseek/deepseek-chat-v3.1:free"
FALLBACK_MODEL = "openai/gpt-oss-20b:free"

def _chat_with_model(model_name: str, messages: list, temperature: float = 0.2):
    completion = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
        extra_body={}
    )
    return (completion.choices[0].message.content or "").strip()

def ia_call(messages: list, temperature: float = 0.2) -> str:
    """
    Llama primero al modelo primario (DeepSeek).
    Si está saturado o hay error de proveedor/red, intenta con el modelo de fallback.
    Si ambos fallan, lanza RuntimeError para que los niveles usen el fallback pedagógico local.
    """
    if not OPENROUTER_API_KEY:
        raise RuntimeError("IA_NO_API_KEY")

    # 1) Intento con modelo primario
    try:
        return _chat_with_model(PRIMARY_MODEL, messages, temperature)

    except BadRequestError as e:
        msg = str(e)
        # Errores típicos del proveedor donde vale la pena probar fallback
        if "Model is at capacity" in msg or "Provider returned error" in msg:
            # seguimos a fallback
            pass
        else:
            # Errores de prompt/malformed request → relanzamos
            raise

    except Exception:
        # Otros errores (timeout, red, etc.) → intentamos fallback
        pass

    # 2) Fallback con modelo alternativo
    try:
        return _chat_with_model(FALLBACK_MODEL, messages, temperature)
    except Exception as e2:
        # Ambos modelos fallaron
        raise RuntimeError(f"IA_BOTH_FAILED: {e2}")


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
    Usa OpenRouter con fallback de modelos para dar feedback educativo breve.
    Si no hay API key o fallan ambos modelos, devuelve mensaje local.
    """
    if not OPENROUTER_API_KEY:
        return "Feedback IA no disponible. Tus resultados se validaron localmente."

    messages = [
        {
            "role": "system",
            "content": (
                "Eres un tutor de contabilidad empático y claro. "
                "Explica en máximo 6 líneas el acierto/error del estudiante, "
                "resalta la fórmula clave o el concepto y ofrece 1 truco memotécnico."
            )
        },
        {"role": "user", "content": prompt_user}
    ]

    try:
        return ia_call(messages, temperature=0.3)
    except Exception:
        return "No pude generar feedback con IA ahora. Tus resultados se validaron localmente."


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
    """
    Evaluación flexible de la respuesta abierta del Nivel 1.
    Acepta respuestas cortas siempre que la idea central sea correcta:
    → Cuando el inventario final disminuye, el CMV AUMENTA.
    """

    pregunta = (
        "Explica qué ocurre con el costo de la mercancía vendida (CMV) cuando "
        "disminuye el inventario final, manteniendo constantes los demás elementos."
    )

    criterios = (
        "Evalúa con AMPLIA FLEXIBILIDAD:\n"
        "✔ La respuesta es CORRECTA si expresa la idea central, aunque sea muy corta:\n"
        "   → Cuando el inventario final disminuye, el CMV AUMENTA.\n"
        "✔ La justificación es DESEABLE pero NO obligatoria.\n"
        "❌ Solo debe marcarse como INCORRECTO si:\n"
        "   - Afirma que el CMV disminuye.\n"
        "   - Dice que no cambia.\n"
        "   - No tiene relación con el concepto contable.\n"
        "   - La explicación es totalmente errónea.\n"
        "\n"
        "Permite respuestas cortas como: 'Aumenta', 'El CMV sube', 'CMV ↑'."
    )

    # Llamamos a la misma función que usas en todo el proyecto
    return eval_ia_explicacion(
        pregunta=pregunta,
        criterios=criterios,
        respuesta_estudiante=respuesta_estudiante
    )

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
        <button id="{key_prefix}-pause">⏸️ Pausar</button>
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
        const btnPause = document.getElementById("{key_prefix}-pause");
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
            btnPause.textContent = "⏸️ Pausar";
          }} catch (e) {{}}
        }};

        btnPause.onclick = () => {{
          try {{
            if (!speechSynthesis.speaking && !speechSynthesis.paused) return;
            if (speechSynthesis.paused) {{
              speechSynthesis.resume();
              btnPause.textContent = "⏸️ Pausar";
            }} else {{
              speechSynthesis.pause();
              btnPause.textContent = "▶️ Reanudar";
            }}
          }} catch (e) {{}}
        }};

        btnStop.onclick = () => {{
          try {{
            speechSynthesis.cancel();
            btnPause.textContent = "⏸️ Pausar";
          }} catch (e) {{}}
        }};
      }})();
    </script>
    """
    components.html(html, height=160)

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

            # ===== LÓGICA DE AJUSTE DEL INVENTARIO FINAL =====
            # Cálculo del máximo inventario final coherente:
            # InvF_max = InvI + Compras − Devoluciones en compras
            invf_max = inv0 + compras - devol
            invf_calc = invf          # lo que usaremos en el cálculo interno
            ajustado = False

            # Si el InvF digitado hace que el CMV sea negativo, ajustamos InvF
            # para que el CMV sea exactamente 0 (equivalente a que no hubo ventas).
            cogs_bruto = inv0 + compras - devol - invf
            if cogs_bruto < 0:
                invf_calc = invf_max
                ajustado = True

            # Ahora calculamos el CMV usando el inventario final "efectivo"
            cogs = inv0 + compras - devol - invf_calc

            st.markdown(
                f"""
                <div style='line-height:1.8; font-size:1.05rem;'>
                    <b>1)</b> Inventario inicial + Compras → {peso(inv0)} + {peso(compras)} = <b>{peso(inv0+compras)}</b><br>
                    <b>2)</b> Menos devoluciones en compras → {peso(inv0+compras)} − {peso(devol)} = <b>{peso(inv0+compras-devol)}</b><br>
                    <b>3)</b> Menos inventario final → {peso(inv0+compras-devol)} − {peso(invf_calc)} = <b>{peso(cogs)}</b>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.success(f"**Costo de la mercancía vendida** = {peso(cogs)}")

            # Mensaje explicando el ajuste (solo si hubo que corregir)
            if ajustado:
                st.warning(
                    "⚠️ El inventario final que ingresaste era tan alto que el costo de la mercancía vendida "
                    "habría resultado **negativo**, lo cual no es coherente en un sistema periódico.\n\n"
                    f"- Inventario final digitado: {peso(invf)}\n"
                    f"- Máximo inventario final posible según la fórmula "
                    f"(InvI + Compras − Devoluciones): {peso(invf_max)}\n\n"
                    "Para efectos del cálculo, ajustamos internamente el inventario final a ese valor máximo, "
                    "lo que equivale a suponer que **no hubo ventas** y por eso el costo de la mercancía vendida es 0."
                )

            st.caption(
                "Interpretación: en la ‘mochila de costos’ entran el inventario inicial y las compras. "
                "Las devoluciones en compras restan costo disponible. Al final, el inventario que queda en la mochila "
                "(inventario final) **no** corresponde a ventas. La diferencia es el **costo de la mercancía vendida**. "
                "Si el inventario final fuera tan alto que el costo resultara negativo, en la práctica significa que "
                "no tuvo sentido ese dato y debe ajustarse al máximo posible (sin ventas)."
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

    # =========================================
    # TAB 1 · EJEMPLO GUIADO (KARDEX DINÁMICO)
    # =========================================
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
            entrada_inicial_total = inv0_u * inv0_pu

            # AHORA EL SALDO INICIAL SE MUESTRA COMO ENTRADA Y COMO SALDO
            rows.append([
                "Día 1", "Saldo inicial",
                int(inv0_u) if inv0_u > 0 else "", round(inv0_pu, 2) if inv0_u > 0 else "", round(entrada_inicial_total, 2) if inv0_u > 0 else "",
                "", "", "",
                int(saldo_qty), round(saldo_pu, 2), round(saldo_val, 2)
            ])

            explain_lines = [f"- **Saldo inicial**: registramos {int(inv0_u)} u @ {_fmt_money(inv0_pu)} como una entrada inicial al Kardex; "
                             f"el saldo muestra esas mismas {int(saldo_qty)} u, por {_fmt_money(saldo_val)}."]

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
                    f"- **Compra**: +{int(comp_u)} u @ {_fmt_money(comp_pu)}. A partir del saldo inicial ya registrado, "
                    f"recalculamos el costo promedio: ahora el saldo es de {int(saldo_qty)} u a {_fmt_money(saldo_pu)}."
                )
            else:
                # PEPS / UEPS: solo fila de Compra (sin “Saldo (día 1)”)
                saldo_layers.append([float(comp_u), float(comp_pu)])
                saldo_qty = sum(q for q, _ in saldo_layers)
                saldo_val = sum(q * p for q, p in saldo_layers)
                saldo_pu = (saldo_val / saldo_qty) if saldo_qty > 0 else 0.0
                rows.append([
                    "Día 2", "Compra",
                    int(comp_u), round(comp_pu, 2), round(entrada_total, 2),
                    "", "", "",
                    int(comp_u), round(comp_pu, 2), round(entrada_total, 2)
                ])
                capas_txt = " · ".join([f"{int(q)}u@{_fmt_money(p)}" for q, p in saldo_layers])
                explain_lines.append(
                    f"- **Compra**: +{int(comp_u)} u @ {_fmt_money(comp_pu)}. "
                    f"La fila del Día 2 muestra la nueva capa; el inventario total se observa combinando el saldo inicial del Día 1 y esta fila. "
                    f"Capas: {capas_txt}."
                )

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
            """
            Construye las filas del KARDEX y un 'script' con guiones pedagógicos
            adaptados a los valores que el estudiante definió.
            - Siempre registra el saldo inicial como una ENTRADA (Día 1).
            - Día 2: solo la compra (sin fila extra de saldo).
            - Día 3: venta según el método (PP una fila, PEPS/UEPS por tramos).
            En PEPS/UEPS NO se promedia el saldo: se trabaja por capas.
            """

            def _sum_layers(layers):
                q = sum(q for q, _ in layers)
                v = sum(q * p for q, p in layers)
                pu = (v / q) if q > 0 else 0.0
                return q, pu, v

            def _consume_layers_detail(layers, qty_out, fifo=True):
                """Devuelve (detalles_salida, layers_after) donde detalles_salida es lista de (qty, pu, total)."""
                order = layers[:] if fifo else layers[::-1]
                remaining = qty_out
                details = []
                after = []
                for (q, pu) in order:
                    if remaining <= 0:
                        after.append([q, pu])
                        continue
                    take = min(q, remaining)
                    if take > 0:
                        details.append([take, pu, take * pu])
                        q_rest = q - take
                        remaining -= take
                        if q_rest > 0:
                            after.append([q_rest, pu])
                    else:
                        after.append([q, pu])
                layers_after = after if fifo else after[::-1]
                return details, layers_after

            rows = []   # cada fila: dict con keys: fecha, desc, ent_q, ent_pu, ent_tot, sal_q, sal_pu, sal_tot, sdo_q, sdo_pu, sdo_tot
            script = [] # guiones para la narración

            # ----------------------------------------------------
            # DÍA 1 · SALDO INICIAL (SIEMPRE COMO ENTRADA + SALDO)
            # ----------------------------------------------------
            if inv0_u > 0:
                ent_q_1 = int(inv0_u)
                ent_pu_1 = float(inv0_pu)
                ent_tot_1 = ent_q_1 * ent_pu_1
                s_q, s_pu, s_v = float(ent_q_1), ent_pu_1, ent_tot_1
                layers = [[s_q, s_pu]]
            else:
                ent_q_1 = None
                ent_pu_1 = None
                ent_tot_1 = None
                layers = []
                s_q, s_pu, s_v = 0.0, 0.0, 0.0

            rows.append({
                "fecha": "Día 1", "desc": "Saldo inicial",
                "ent_q": ent_q_1, "ent_pu": ent_pu_1, "ent_tot": ent_tot_1,
                "sal_q": None, "sal_pu": None, "sal_tot": None,
                "sdo_q": int(s_q), "sdo_pu": round(s_pu, 2), "sdo_tot": round(s_v, 2),
            })

            if inv0_u > 0:
                script.append({
                    "title": "Paso 1 · Saldo inicial como entrada",
                    "text": (
                        f"Empezamos registrando el saldo inicial como una ENTRADA: "
                        f"{ent_q_1} unidades a un costo unitario de {_fmt_money(ent_pu_1)} pesos. "
                        f"En la columna Total calculamos {ent_q_1} por {_fmt_money(ent_pu_1)} pesos, "
                        f"lo que da {_fmt_money(ent_tot_1)} pesos. Ese mismo valor pasa a la columna Saldo, "
                        f"porque al inicio solo existe esta capa de inventario."
                    ),
                    "actions": [
                        {"row": 0, "cell": "ent_q", "money": False, "val": ent_q_1},
                        {"row": 0, "cell": "ent_pu", "money": True,  "val": round(ent_pu_1, 2)},
                        {"row": 0, "cell": "ent_tot", "money": True, "val": round(ent_tot_1, 2)},
                        {"row": 0, "cell": "sdo_q", "money": False, "val": int(s_q)},
                        {"row": 0, "cell": "sdo_pu", "money": True, "val": round(s_pu, 2)},
                        {"row": 0, "cell": "sdo_tot", "money": True, "val": round(s_v, 2)},
                    ]
                })

            else:
                script.append({
                    "title": "Paso 1 · Sin saldo inicial",
                    "text": (
                        "En este escenario no hay saldo inicial de inventarios. Dejamos vacías las columnas "
                        "de Entrada en el Día 1 y el Saldo comienza en cero."
                    ),
                    "actions": [
                        {"row": 0, "cell": "sdo_q", "money": False, "val": 0},
                        {"row": 0, "cell": "sdo_pu", "money": True, "val": 0.0},
                        {"row": 0, "cell": "sdo_tot", "money": True, "val": 0.0},
                    ]
                })

            # ---------------------------------
            # DÍA 2 · COMPRA (SOLO UNA FILA)
            # ---------------------------------
            ent_q_2 = int(comp_u) if comp_u > 0 else None
            ent_pu_2 = float(comp_pu) if comp_u > 0 else None
            ent_tot_2 = ent_q_2 * ent_pu_2 if comp_u > 0 else None

            if method_name == "Promedio Ponderado":
                # Guardar saldo previo para explicación
                prev_q, prev_pu, prev_v = s_q, s_pu, s_v

                if comp_u > 0:
                    new_q = prev_q + comp_u
                    new_v = prev_v + ent_tot_2
                    new_p = (new_v / new_q) if new_q > 0 else 0.0
                    layers = [[new_q, new_p]]
                    s_q, s_pu, s_v = new_q, new_p, new_v

                rows.append({
                    "fecha": "Día 2", "desc": "Compra",
                    "ent_q": ent_q_2, "ent_pu": ent_pu_2, "ent_tot": ent_tot_2,
                    "sal_q": None, "sal_pu": None, "sal_tot": None,
                    "sdo_q": int(s_q), "sdo_pu": round(s_pu, 2), "sdo_tot": round(s_v, 2),
                })

                script.append({
                    "title": "Paso 2 · Compra y nuevo costo promedio",
                    "text": (
                        f"Registramos la compra del Día 2 como ENTRADA: {ent_q_2} unidades a "
                        f"{_fmt_money(ent_pu_2)} pesos, con un total de {_fmt_money(ent_tot_2)} pesos.\n\n"
                        f"Para actualizar el saldo aplicamos el Promedio Ponderado:\n"
                        f"Valor anterior del saldo: {_fmt_money(prev_v)} pesos con {int(prev_q)} unidades.\n"
                        f"Valor de la compra: {_fmt_money(ent_tot_2)} pesos con {ent_q_2} unidades.\n"
                        f"Nuevo costo promedio = (valor anterior + valor de la compra) / "
                        f"(unidades anteriores + unidades compradas).\n"
                        f"Es decir: {_fmt_money(prev_v)} pesos más {_fmt_money(ent_tot_2)} pesos, dividido entre "
                        f"{int(prev_q)} más {ent_q_2} unidades, nos da un costo promedio de {_fmt_money(s_pu)} pesos por unidad.\n"
                        f"Con ese costo promedio mostramos el nuevo SALDO de {int(s_q)} unidades "
                        f"por {_fmt_money(s_pu)} pesos, para un total de {_fmt_money(s_v)} pesos."
                    ),
                    "actions": [
                        {"row": 1, "cell": "ent_q", "money": False, "val": ent_q_2},
                        {"row": 1, "cell": "ent_pu", "money": True,  "val": round(ent_pu_2, 2)},
                        {"row": 1, "cell": "ent_tot", "money": True, "val": round(ent_tot_2, 2)},
                        {"row": 1, "cell": "sdo_q", "money": False, "val": int(s_q)},
                        {"row": 1, "cell": "sdo_pu", "money": True, "val": round(s_pu, 2)},
                        {"row": 1, "cell": "sdo_tot", "money": True, "val": round(s_v, 2)},
                    ]
                })
                start_sale_row_index = 2

            else:
                # PEPS / UEPS → se manejan capas, pero en esta fila mostramos solo la capa de la compra
                if comp_u > 0:
                    layers.append([float(comp_u), float(comp_pu)])
                s_q, _, _ = _sum_layers(layers)

                rows.append({
                    "fecha": "Día 2", "desc": "Compra",
                    "ent_q": ent_q_2, "ent_pu": ent_pu_2, "ent_tot": ent_tot_2,
                    "sal_q": None, "sal_pu": None, "sal_tot": None,
                    "sdo_q": ent_q_2 or 0,
                    "sdo_pu": round(ent_pu_2 or 0.0, 2),
                    "sdo_tot": round(ent_tot_2 or 0.0, 2),
                })

                metodo_tag = "PEPS" if method_name == "PEPS (FIFO)" else "UEPS (LIFO)"
                script.append({
                    "title": f"Paso 2 · Compra como nueva capa ({metodo_tag})",
                    "text": (
                        f"En {metodo_tag} NO promediamos el costo. La compra del Día 2 se registra como una nueva capa: "
                        f"{ent_q_2} unidades a {_fmt_money(ent_pu_2)} pesos, total {_fmt_money(ent_tot_2)} pesos.\n\n"
                        f"En la columna SALDO de esta fila mostramos solo esa capa comprada... "
                        f"más adelante la venta consumirá primero una u otra capa según el método, "
                        f"en lugar de combinar todo en un único costo promedio."
                    ),
                    "actions": [
                        {"row": 1, "cell": "ent_q", "money": False, "val": ent_q_2},
                        {"row": 1, "cell": "ent_pu", "money": True,  "val": round(ent_pu_2, 2)},
                        {"row": 1, "cell": "ent_tot", "money": True, "val": round(ent_tot_2, 2)},
                        {"row": 1, "cell": "sdo_q", "money": False, "val": ent_q_2 or 0},
                        {"row": 1, "cell": "sdo_pu", "money": True, "val": round(ent_pu_2 or 0.0, 2)},
                        {"row": 1, "cell": "sdo_tot", "money": True, "val": round(ent_tot_2 or 0.0, 2)},
                    ]
                })
                start_sale_row_index = 2

            # ---------------------------------
            # DÍA 3 · VENTA (PP vs PEPS/UEPS)
            # ---------------------------------
            if venta_u > 0 and s_q > 0:
                # ======== PROMEDIO PONDERADO ========
                if method_name == "Promedio Ponderado":
                    sal_q = min(venta_u, int(s_q))
                    sal_pu = layers[0][1] if layers else 0.0
                    sal_tot = sal_q * sal_pu

                    prev_q, prev_pu, prev_v = s_q, s_pu, s_v
                    new_q = prev_q - sal_q
                    new_v = prev_v - sal_tot
                    new_p = (new_v / new_q) if new_q > 0 else 0.0
                    layers = [[new_q, new_p]] if new_q > 0 else []
                    s_q, s_pu, s_v = new_q, new_p, new_v

                    rows.append({
                        "fecha": "Día 3", "desc": "Venta",
                        "ent_q": None, "ent_pu": None, "ent_tot": None,
                        "sal_q": int(sal_q), "sal_pu": round(sal_pu, 2), "sal_tot": round(sal_tot, 2),
                        "sdo_q": int(s_q), "sdo_pu": round(s_pu, 2), "sdo_tot": round(s_v, 2),
                    })

                    script.append({
                        "title": "Paso 3 · Venta con Promedio Ponderado",
                        "text": (
                            f"En el Día 3 registramos la VENTA como una SALIDA de {sal_q} unidades. "
                            f"Como estamos en Promedio Ponderado, usamos el costo promedio vigente: "
                            f"{_fmt_money(sal_pu)} pesos por unidad.\n\n"
                            f"El CMV de esta fila es {sal_q} por {_fmt_money(sal_pu)} pesos, "
                            f"lo que da {_fmt_money(sal_tot)} pesos.\n"
                            f"Para actualizar el SALDO restamos esas unidades y ese valor al saldo anterior "
                            f"({int(prev_q)} unidades por {_fmt_money(prev_pu)} pesos, total {_fmt_money(prev_v)} pesos), "
                            f"y volvemos a calcular el nuevo costo promedio del inventario que queda."
                        ),
                        "actions": [
                            {"row": start_sale_row_index, "cell": "sal_q", "money": False, "val": int(sal_q)},
                            {"row": start_sale_row_index, "cell": "sal_pu", "money": True,  "val": round(sal_pu, 2)},
                            {"row": start_sale_row_index, "cell": "sal_tot", "money": True, "val": round(sal_tot, 2)},
                            {"row": start_sale_row_index, "cell": "sdo_q", "money": False, "val": int(s_q)},
                            {"row": start_sale_row_index, "cell": "sdo_pu", "money": True, "val": round(s_pu, 2)},
                            {"row": start_sale_row_index, "cell": "sdo_tot", "money": True, "val": round(s_v, 2)},
                        ]
                    })

                # ======== PEPS / UEPS (TRAMOS, SIN PROMEDIO) ========
                else:
                    fifo = (method_name == "PEPS (FIFO)")
                    metodo_tag = "PEPS" if fifo else "UEPS"

                    # Copia de capas para irlas consumiendo tramo a tramo
                    layers_for_calc = [[float(q), float(p)] for (q, p) in layers]
                    sale_remaining = float(venta_u)
                    tramo_index = 1
                    acc_row = start_sale_row_index

                    while sale_remaining > 0 and any(q > 0 for q, _ in layers_for_calc):
                        # Seleccionar la capa según el método
                        if fifo:
                            idx_layer = next(i for i, (q, _) in enumerate(layers_for_calc) if q > 0)
                        else:
                            idx_layer = max(i for i, (q, _) in enumerate(layers_for_calc) if q > 0)

                        layer_q, layer_pu = layers_for_calc[idx_layer]
                        q_take = min(layer_q, sale_remaining)
                        tot_take = q_take * layer_pu
                        sale_remaining -= q_take

                        # Actualizar la capa consumida
                        q_rem = layer_q - q_take
                        layers_for_calc[idx_layer][0] = q_rem

                        # 🔴 CORRECCIÓN: SALDO POR TRAMO EN PEPS/UEPS
                        # Si se agota la capa en este tramo → saldo 0 a ese mismo costo.
                        if q_rem > 0:
                            # Quedan unidades en la MISMA capa que acabamos de consumir
                            sdo_q = q_rem
                            sdo_pu = layer_pu
                        else:
                            # Capa agotada: en esta fila mostramos saldo 0 a ese costo
                            sdo_q = 0.0
                            sdo_pu = layer_pu

                        sdo_tot = sdo_q * sdo_pu

                        rows.append({
                            "fecha": "Día 3",
                            "desc": f"Venta tramo {tramo_index} ({metodo_tag})",
                            "ent_q": None, "ent_pu": None, "ent_tot": None,
                            "sal_q": int(q_take), "sal_pu": round(layer_pu, 2), "sal_tot": round(tot_take, 2),
                            "sdo_q": int(sdo_q), "sdo_pu": round(sdo_pu, 2), "sdo_tot": round(sdo_tot, 2),
                        })

                        if fifo:
                            frase_capa = (
                                "En PEPS, primero salen las unidades más antiguas. "
                                "Por eso este tramo consume unidades de la capa que entró primero."
                            )
                        else:
                            frase_capa = (
                                "En UEPS, primero salen las unidades más recientes. "
                                "Por eso este tramo consume unidades de la capa que entró de último."
                            )

                        script.append({
                            "title": f"Paso 3 · Venta (tramo {tramo_index}) — {metodo_tag}",
                            "text": (
                                f"En este tramo sacamos {int(q_take)} unidades de una capa valorada a "
                                f"{_fmt_money(layer_pu)} pesos. El costo del tramo es {int(q_take)} por "
                                f"{_fmt_money(layer_pu)} pesos, es decir {_fmt_money(tot_take)} pesos.\n\n"
                                f"{frase_capa}\n"
                                f"En la columna SALDO de esta fila mostramos **la misma capa** después del tramo. "
                                f"Si se agotó, verás 0 unidades a ese mismo costo; si quedaron unidades, "
                                f"verás cuántas siguen en esa capa, siempre sin promediar."
                            ),
                            "actions": [
                                {"row": acc_row, "cell": "sal_q", "money": False, "val": int(q_take)},
                                {"row": acc_row, "cell": "sal_pu", "money": True,  "val": round(layer_pu, 2)},
                                {"row": acc_row, "cell": "sal_tot", "money": True, "val": round(tot_take, 2)},
                                {"row": acc_row, "cell": "sdo_q", "money": False, "val": int(sdo_q)},
                                {"row": acc_row, "cell": "sdo_pu", "money": True, "val": round(sdo_pu, 2)},
                                {"row": acc_row, "cell": "sdo_tot", "money": True, "val": round(sdo_tot, 2)},
                            ]
                        })

                        tramo_index += 1
                        acc_row += 1

            else:
                # No hay venta o no hay saldo
                rows.append({
                    "fecha": "Día 3", "desc": "Venta",
                    "ent_q": None, "ent_pu": None, "ent_tot": None,
                    "sal_q": None, "sal_pu": None, "sal_tot": None,
                    "sdo_q": int(s_q), "sdo_pu": round(s_pu, 2), "sdo_tot": round(s_v, 2),
                })
                script.append({
                    "title": "Paso 3 · Sin venta o sin saldo",
                    "text": (
                        "En este escenario no registramos una venta efectiva (o no hay inventario para vender). "
                        "Por eso las columnas de SALIDA quedan vacías y el SALDO permanece igual que en el Día 2."
                    ),
                    "actions": [
                        {"row": start_sale_row_index, "cell": "sdo_q", "money": False, "val": int(s_q)},
                        {"row": start_sale_row_index, "cell": "sdo_pu", "money": True, "val": round(s_pu, 2)},
                        {"row": start_sale_row_index, "cell": "sdo_tot", "money": True, "val": round(s_v, 2)},
                    ]
                })

            return rows, script

        demo_rows, demo_script = compute_rows_and_script(metodo, inv0_u, inv0_pu, comp_u, comp_pu, venta_u)

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
        <button id="pauseDemo" class="btn">⏸️ Pausa</button>
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
        const btnPause = document.getElementById("pauseDemo");
        const btnReset = document.getElementById("resetDemo");

        const pesos = (v)=> {
            try {
                return new Intl.NumberFormat('es-CO', {
                    style: 'currency',
                    currency: 'COP',
                    maximumFractionDigits: 2
                }).format(v);
            } catch(e){
                return "$" + (Math.round(v*100)/100).toLocaleString('es-CO');
            }
        };

        const fmt = (x)=> (
            x===null || x===undefined || x===""
                ? ""
                : (typeof x==="number"
                    ? (Number.isInteger(x)
                        ? x.toString()
                        : (Math.round(x*100)/100).toString().replace(".",",")
                    )
                    : x
                )
        );

        // 🔹 NUEVO: limpiar el texto para la voz
        function cleanForSpeak(text) {
            if (!text) return "";

            let t = text;

            // 1) Casos tipo: US$100, US$ 100, $100, $ 100, COP 100 → "100 pesos"
            t = t.replace(/\bUS?\$\s*(\d+(?:[\\.,]\d+)*)\s*(pesos)?/gi, "$1 pesos");
            t = t.replace(/\$\s*(\d+(?:[\\.,]\d+)*)\s*(pesos)?/g, "$1 pesos");
            t = t.replace(/\bCOP\s*(\d+(?:[\\.,]\d+)*)\s*(pesos)?/gi, "$1 pesos");
            // 100 $, 100 US$, 100 COP → "100 pesos"
            t = t.replace(/(\d+(?:[\\.,]\d+)*)\s*(US?\$|COP|\$)\b/gi, "$1 pesos");

            // 2) Si quedó "pesos 100" → "100 pesos"
            t = t.replace(/pesos\s+(\d+(?:[\\.,]\d+)*)/gi, "$1 pesos");

            // 3) Si quedó "100 pesos pesos" → "100 pesos"
            t = t.replace(/(\d+(?:[\\.,]\d+)*)\s+pesos\s+pesos/gi, "$1 pesos");

            // 4) Si quedó "pesos 100 pesos" → "100 pesos"
            t = t.replace(/pesos\s+(\d+(?:[\\.,]\d+)*)\s+pesos/gi, "$1 pesos");

            // 5) Limpiar espacios dobles
            t = t.replace(/\s{2,}/g, " ");

            return t;
        }

        let isRunning = false;
        let isPaused = false;
        let shouldStop = false;

        function speak(text){
            return new Promise((resolve)=>{
                if (narrMuted) return resolve();
                try{
                    // NO cancelamos aquí para permitir pausa/reanudar
                    const u = new SpeechSynthesisUtterance(cleanForSpeak(text));
                    const voices = window.speechSynthesis.getVoices();
                    const pick = voices.find(v=>/es|spanish|mex|col/i.test((v.name+" "+v.lang))) || voices[0];
                    if (pick) u.voice = pick;
                    u.rate = rate;
                    u.pitch = 1.0;
                    u.onend = ()=> resolve();
                    window.speechSynthesis.speak(u);
                } catch(e){
                    resolve();
                }
            });
        }

        const sleep = (ms)=> new Promise(r=>setTimeout(r, ms));

        async function waitWhilePaused(){
            while(isPaused){
                await sleep(150);
            }
        }

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
            // Si ya está corriendo y está en pausa, al hacer clic en Reproducir reanudamos
            if (isRunning && isPaused){
                isPaused = false;
                if (window.speechSynthesis && window.speechSynthesis.resume){
                    window.speechSynthesis.resume();
                }
                btnPause.textContent = "⏸️ Pausa";
                return;
            }
            // Si ya está corriendo y no está en pausa, ignoramos
            if (isRunning) return;

            isRunning = true;
            shouldStop = false;
            isPaused = false;
            btnPause.textContent = "⏸️ Pausa";

            clearTable();
            buildTable();

            // Pintar título de cada paso, resaltar fila y llenar celdas en orden
            for (let sIdx = 0; sIdx < script.length; sIdx++){
                if (shouldStop) break;
                const step = script[sIdx];

                await waitWhilePaused();
                if (shouldStop) break;

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
                    if (shouldStop) break;
                    await waitWhilePaused();
                    if (shouldStop) break;

                    const a = step.actions[i];
                    if (i>0){
                        await sleep(waits[i-1]);
                        await waitWhilePaused();
                        if (shouldStop) break;
                    }
                    fillCell(a.row, a.cell, a.val, !!a.money);
                }

                await pVoice;
                if (shouldStop) break;
                await waitWhilePaused();
                if (shouldStop) break;
                await sleep(200);
            }
            highlightRow(-1);
            isRunning = false;
        }

        // --- Botones ---
        // Reproducir / continuar
        btnPlay.onclick  = runScript;

        // Pausar / reanudar
        btnPause.onclick = ()=>{
            if (!isRunning) return;
            if (!isPaused){
                isPaused = true;
                btnPause.textContent = "▶️ Reanudar";
                if (window.speechSynthesis && window.speechSynthesis.pause){
                    window.speechSynthesis.pause();
                }
            } else {
                isPaused = false;
                btnPause.textContent = "⏸️ Pausa";
                if (window.speechSynthesis && window.speechSynthesis.resume){
                    window.speechSynthesis.resume();
                }
            }
        };

        // Reiniciar todo
        btnReset.onclick = ()=>{
            shouldStop = true;
            isPaused = false;
            isRunning = false;
            btnPause.textContent = "⏸️ Pausa";
            if (window.speechSynthesis && window.speechSynthesis.cancel){
                window.speechSynthesis.cancel();
            }
            clearTable();
            buildTable();
            highlightRow(-1);
            narrDiv.textContent = "";
        };

        buildTable();
        if (window.speechSynthesis) {
            window.speechSynthesis.onvoiceschanged = ()=>{};
        }
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

            Lógica alineada con el EJEMPLO GUIADO:
            - Día 1: saldo inicial como ENTRADA y como SALDO.
            - Día 2: SOLO una fila de Compra 1 (sin “Saldo (día 1)”).
            * Promedio Ponderado: saldo con promedio.
            * PEPS/UEPS: saldo solo de la capa comprada.
            - Día 3: ventas por método.
            * PP: una fila.
            * PEPS/UEPS: tramos sin promedios; el saldo del tramo muestra la capa del tramo (0 si se agota).
            - Día 4: Compra 2 (saldo solo de la capa comprada en PEPS/UEPS).
            """
            rows = []

            # ------------------------------
            # Día 1: Saldo inicial
            # ------------------------------
            if inv0_u_ex > 0:
                ent_q1 = int(inv0_u_ex)
                ent_pu1 = float(inv0_pu_ex)
                ent_tot1 = ent_q1 * ent_pu1
                layers = [[float(ent_q1), ent_pu1]]
            else:
                ent_q1 = ent_pu1 = ent_tot1 = None
                layers = []

            s_q, s_p, s_v = _sum_layers(layers)

            rows.append({
                "Fecha": "Día 1", "Descripción": "Saldo inicial",
                "Entrada_cant": ent_q1,
                "Entrada_pu": round(ent_pu1, 2) if ent_pu1 else None,
                "Entrada_total": round(ent_tot1, 2) if ent_tot1 else None,
                "Salida_cant": None, "Salida_pu": None, "Salida_total": None,
                "Saldo_cant": s_q,
                "Saldo_pu": round(s_p, 2),
                "Saldo_total": round(s_v, 2)
            })

            # ------------------------------
            # Día 2: Compra 1
            # ------------------------------
            ent_tot = comp1_u * comp1_pu

            if method_name == "Promedio Ponderado":
                q_new = s_q + comp1_u
                v_new = s_v + ent_tot
                p_new = (v_new / q_new) if q_new > 0 else 0.0
                layers = [[q_new, p_new]]
                s_q, s_p, s_v = _sum_layers(layers)

                rows.append({
                    "Fecha": "Día 2", "Descripción": "Compra 1",
                    "Entrada_cant": comp1_u,
                    "Entrada_pu": round(comp1_pu, 2),
                    "Entrada_total": round(ent_tot, 2),
                    "Salida_cant": None, "Salida_pu": None, "Salida_total": None,
                    "Saldo_cant": s_q,
                    "Saldo_pu": round(s_p, 2),
                    "Saldo_total": round(s_v, 2)
                })

            else:
                # PEPS/UEPS → una sola fila "Compra 1"
                if comp1_u > 0:
                    layers.append([float(comp1_u), float(comp1_pu)])

                rows.append({
                    "Fecha": "Día 2", "Descripción": "Compra 1",
                    "Entrada_cant": comp1_u,
                    "Entrada_pu": round(comp1_pu, 2),
                    "Entrada_total": round(ent_tot, 2),
                    "Salida_cant": None, "Salida_pu": None, "Salida_total": None,
                    "Saldo_cant": comp1_u,
                    "Saldo_pu": round(comp1_pu, 2),
                    "Saldo_total": round(ent_tot, 2)
                })

            # ------------------------------
            # Día 3: Venta
            # ------------------------------
            if method_name == "Promedio Ponderado":
                if s_q > 0 and venta_ex_u > 0:
                    sale_q = min(venta_ex_u, s_q)
                    sale_pu = layers[0][1] if layers else 0.0
                    sale_tot = sale_q * sale_pu

                    q2 = s_q - sale_q
                    v2 = s_v - sale_tot
                    p2 = (v2 / q2) if q2 > 0 else 0.0
                    layers = [[q2, p2]] if q2 > 0 else []
                    s_q, s_p, s_v = _sum_layers(layers)

                    rows.append({
                        "Fecha": "Día 3", "Descripción": "Venta",
                        "Entrada_cant": None, "Entrada_pu": None, "Entrada_total": None,
                        "Salida_cant": sale_q,
                        "Salida_pu": round(sale_pu, 2),
                        "Salida_total": round(sale_tot, 2),
                        "Saldo_cant": s_q,
                        "Saldo_pu": round(s_p, 2),
                        "Saldo_total": round(s_v, 2)
                    })
                else:
                    rows.append({
                        "Fecha": "Día 3", "Descripción": "Venta",
                        "Entrada_cant": None, "Entrada_pu": None, "Entrada_total": None,
                        "Salida_cant": None, "Salida_pu": None, "Salida_total": None,
                        "Saldo_cant": s_q,
                        "Saldo_pu": round(s_p, 2),
                        "Saldo_total": round(s_v, 2)
                    })

            else:
                # PEPS / UEPS (venta por tramos, SALDO solo de la capa del tramo)
                fifo = (method_name == "PEPS (FIFO)")
                metodo_tag = "PEPS" if fifo else "UEPS"

                layers_for_calc = [[float(q), float(p)] for (q, p) in layers]
                sale_remaining = float(venta_ex_u)
                tramo_index = 1

                while sale_remaining > 0 and any(q > 0 for q, _ in layers_for_calc):

                    # Selección de capa
                    if fifo:
                        idx = next(i for i,(q,_) in enumerate(layers_for_calc) if q>0)
                    else:
                        idx = max(i for i,(q,_) in enumerate(layers_for_calc) if q>0)

                    layer_q, layer_pu = layers_for_calc[idx]
                    q_take = min(layer_q, sale_remaining)
                    tot_take = q_take * layer_pu
                    sale_remaining -= q_take

                    # Actualizar capa consumida
                    q_rem = layer_q - q_take
                    layers_for_calc[idx][0] = q_rem

                    # 🔴 NUEVA LÓGICA: SALDO muestra SOLO la capa de ese tramo
                    # Si se agota, queda 0 unidades al mismo costo unitario.
                    if q_rem > 0:
                        sdo_q = q_rem
                        sdo_pu = layer_pu
                    else:
                        sdo_q = 0.0
                        sdo_pu = layer_pu

                    sdo_tot = sdo_q * sdo_pu

                    rows.append({
                        "Fecha": "Día 3",
                        "Descripción": f"Venta tramo {tramo_index} ({metodo_tag})",
                        "Entrada_cant": None, "Entrada_pu": None, "Entrada_total": None,
                        "Salida_cant": int(q_take),
                        "Salida_pu": round(layer_pu, 2),
                        "Salida_total": round(tot_take, 2),
                        "Saldo_cant": int(sdo_q),
                        "Saldo_pu": round(sdo_pu, 2),
                        "Saldo_total": round(sdo_tot, 2)
                    })

                    tramo_index += 1

                layers = [(q,p) for (q,p) in layers_for_calc if q>0]
                s_q, s_p, s_v = _sum_layers(layers)

            # ------------------------------
            # Día 4: Compra 2
            # ------------------------------
            ent2_tot = comp2_u * comp2_pu

            if method_name == "Promedio Ponderado":
                q3 = s_q + comp2_u
                v3 = s_v + ent2_tot
                p3 = (v3 / q3) if q3 > 0 else 0.0
                layers = [[q3, p3]]
                s_q, s_p, s_v = _sum_layers(layers)

                rows.append({
                    "Fecha": "Día 4", "Descripción": "Compra 2",
                    "Entrada_cant": comp2_u,
                    "Entrada_pu": round(comp2_pu, 2),
                    "Entrada_total": round(ent2_tot, 2),
                    "Salida_cant": None, "Salida_pu": None, "Salida_total": None,
                    "Saldo_cant": s_q,
                    "Saldo_pu": round(s_p, 2),
                    "Saldo_total": round(s_v, 2)
                })

            else:
                # PEPS / UEPS: saldo solo de la nueva capa
                if comp2_u > 0:
                    layers.append([float(comp2_u), float(comp2_pu)])

                rows.append({
                    "Fecha": "Día 4", "Descripción": "Compra 2",
                    "Entrada_cant": comp2_u,
                    "Entrada_pu": round(comp2_pu, 2),
                    "Entrada_total": round(ent2_tot, 2),
                    "Salida_cant": None, "Salida_pu": None, "Salida_total": None,
                    "Saldo_cant": comp2_u,
                    "Saldo_pu": round(comp2_pu, 2),
                    "Saldo_total": round(ent2_tot, 2)
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
                    if x in (None, ""):
                        return None
                    return float(x)
                except Exception:
                    return None

            def _near(a, b):
                if a is None or b is None:
                    return False
                return abs(a - b) <= tol

            def _is_empty_exp(v):
                # Valores que NO queremos exigir en la comparación
                if v is None or v == "":
                    return True
                try:
                    import math
                    return isinstance(v, float) and math.isnan(v)
                except Exception:
                    return False

            flags = []
            # Comparar fila a fila contra expected_rows
            for i in range(len(expected_rows)):
                user = edited.iloc[i].to_dict()
                exp  = expected_rows[i]
                ok_cells = []
                # Para cada celda numérica, si exp tiene número, validamos; si está vacío, no se exige
                for key in [
                    "Entrada_cant","Entrada_pu","Entrada_total",
                    "Salida_cant","Salida_pu","Salida_total",
                    "Saldo_cant","Saldo_pu","Saldo_total"
                ]:
                    exp_val = exp[key]
                    usr_val = _to_float(user.get(key, ""))

                    if _is_empty_exp(exp_val):  # no obligatorio
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

    # ====== Helpers de performance y parseo ======
    def _extract_json(txt: str) -> dict:
        """
        Intenta extraer el primer bloque JSON válido en un texto.
        Devuelve {} si no puede parsear.
        """
        try:
            # Busca primer bloque delimitado por { ... }
            start = txt.find("{")
            end = txt.rfind("}")
            if start != -1 and end != -1 and end > start:
                return json.loads(txt[start:end+1])
        except Exception:
            pass
        return {}

    @st.cache_data(show_spinner=False)
    def cached_solve_pp(inv0_u, inv0_pu, comp1_u, comp1_pu, venta_u, comp2_u, comp2_pu):
        # Mismo cuerpo que tu `solve_pp()` actual
        q0 = inv0_u
        v0 = inv0_u * inv0_pu
        # Día 2
        v1 = v0 + comp1_u * comp1_pu
        q1 = inv0_u + comp1_u
        pu1 = (v1 / q1) if q1 > 0 else 0.0
        saldo_after_c1 = (q1, pu1, v1)
        # Día 3
        sale_q = min(venta_u, q1)
        cmv = sale_q * pu1
        q2 = q1 - sale_q
        v2 = v1 - cmv
        pu2 = (v2 / q2) if q2 > 0 else 0.0
        saldo_after_sale = (q2, pu2, v2)
        # Día 4
        q3 = q2 + comp2_u
        v3 = v2 + comp2_u * comp2_pu
        pu3 = (v3 / q3) if q3 > 0 else 0.0
        saldo_final = (q3, pu3, v3)
        return saldo_after_c1, saldo_after_sale, saldo_final

    @st.cache_data(show_spinner=False)
    def cached_solve_peps_rows(peps_inv0_u, peps_inv0_pu, peps_comp1_u, peps_comp1_pu, peps_venta_u, peps_comp2_u, peps_comp2_pu):
        def _sum_layers(layers):
            q = sum(q for q, _ in layers)
            v = sum(q*p for q, p in layers)
            pu = (v / q) if q > 0 else 0.0
            return q, pu, v
        def _consume_layers_detail(layers, qty_out, fifo=True):
            order = layers[:] if fifo else layers[::-1]
            remaining = qty_out
            sale_details = []
            new_layers = []
            for q, pu in order:
                if remaining <= 0:
                    new_layers.append([q, pu]); continue
                take = min(q, remaining)
                if take > 0:
                    sale_details.append((take, pu, take*pu))
                    rest = q - take
                    remaining -= take
                    if rest > 0:
                        new_layers.append([rest, pu])
            layers_after = new_layers if fifo else new_layers[::-1]
            return sale_details, layers_after

        layers = [[float(peps_inv0_u), float(peps_inv0_pu)]]
        s_q, s_pu, s_v = _sum_layers(layers)
        rows = [{
            "fecha":"Día 1","desc":"Saldo inicial",
            "ent_q":None,"ent_pu":None,"ent_tot":None,
            "sal_q":None,"sal_pu":None,"sal_tot":None,
            "sdo_q":int(s_q),"sdo_pu":round(s_pu,2),"sdo_tot":round(s_v,2)
        }]
        # Día 2 — saldo día 1 + compra a su costo
        rows.append({
            "fecha":"Día 2","desc":"Saldo (día 1)",
            "ent_q":None,"ent_pu":None,"ent_tot":None,
            "sal_q":None,"sal_pu":None,"sal_tot":None,
            "sdo_q":int(s_q),"sdo_pu":round(s_pu,2),"sdo_tot":round(s_v,2)
        })
        ent_tot = peps_comp1_u * peps_comp1_pu
        layers.append([float(peps_comp1_u), float(peps_comp1_pu)])
        rows.append({
            "fecha":"Día 2","desc":"Compra 1",
            "ent_q":int(peps_comp1_u),"ent_pu":round(peps_comp1_pu,2),"ent_tot":round(ent_tot,2),
            "sal_q":None,"sal_pu":None,"sal_tot":None,
            "sdo_q":int(peps_comp1_u),"sdo_pu":round(peps_comp1_pu,2),"sdo_tot":round(ent_tot,2)
        })
        # Día 3 — venta en tramos
        def _sum_layers_copy(lrs):
            q = sum(q for q,_ in lrs); v = sum(q*p for q,p in lrs); pu = (v/q) if q>0 else 0.0
            return q, pu, v

        sale_details, layers_after = _consume_layers_detail(layers, peps_venta_u, fifo=True)
        running_layers = [l[:] for l in layers]
        for i, (q_take, pu_take, tot_take) in enumerate(sale_details, start=1):
            sd, running_layers = _consume_layers_detail(running_layers, q_take, fifo=True)
            rq, rpu, rv = _sum_layers_copy(running_layers)
            rows.append({
                "fecha":"Día 3","desc": f"Venta tramo {i} (PEPS)",
                "ent_q":None,"ent_pu":None,"ent_tot":None,
                "sal_q":int(q_take),"sal_pu":round(pu_take,2),"sal_tot":round(tot_take,2),
                "sdo_q":int(rq),"sdo_pu":round(rpu,2),"sdo_tot":round(rv,2)
            })
        layers = layers_after
        # Día 4 — compra 2
        ent2_tot = peps_comp2_u * peps_comp2_pu
        layers.append([float(peps_comp2_u), float(peps_comp2_pu)])
        rows.append({
            "fecha":"Día 4","desc":"Compra 2",
            "ent_q":int(peps_comp2_u),"ent_pu":round(peps_comp2_pu,2),"ent_tot":round(ent2_tot,2),
            "sal_q":None,"sal_pu":None,"sal_tot":None,
            "sdo_q":int(peps_comp2_u),"sdo_pu":round(peps_comp2_pu,2),"sdo_tot":round(ent2_tot,2)
        })
        qF, puF, vF = _sum_layers(layers)
        return rows, (qF, puF, vF)

    def grade_open_with_ai_batched(ans2: str, ans3: str):
        """
        1 request para las dos preguntas abiertas.
        Devuelve (score2, fb2, score3, fb3)
        """
        prompt = f"""
    Eres un evaluador. Devuelve SOLO un JSON con este formato EXACTO:
    {{
    "q2": {{"score": 0|1, "feedback": "texto breve"}},
    "q3": {{"score": 0|1, "feedback": "texto breve"}}
    }}

    Criterios:
    - q2 (importancia de elegir método): puntúa 1 si menciona al menos dos de:
    CMV/utilidad, estados financieros/comparabilidad, impuestos/decisiones.
    - q3 (cuando conviene Promedio Ponderado): puntúa 1 si propone un caso plausible
    (compras frecuentes/costos variables/menor volatilidad/simplificación operativa) y lo justifica.

    Respuestas del estudiante:
    [Q2]
    {ans2}

    [Q3]
    {ans3}
    """
        raw = ia_feedback(prompt)
        data = _extract_json(raw)
        try:
            s2 = int(data.get("q2", {}).get("score", 0))
            f2 = str(data.get("q2", {}).get("feedback", "")).strip()
            s3 = int(data.get("q3", {}).get("score", 0))
            f3 = str(data.get("q3", {}).get("feedback", "")).strip()
            return s2, f2, s3, f3
        except Exception:
            def quick(a, kws, minhits):
                hit = sum(1 for k in kws if k.lower() in (a or "").lower())
                return (1 if hit >= minhits else 0, "Respuesta breve; refuerza con ejemplos y efectos en CMV/estados.")
            s2, f2 = quick(ans2, ["CMV","utilidad","estados","financieros","impuestos","decisiones","comparabilidad"], 2)
            s3, f3 = quick(ans3, ["promedio","compras frecuentes","costos variables","volatilidad","simplificar"], 2)
            return s2, f2, s3, f3
        

    def _fail(msg):
        return False, "❌ No alcanza los criterios.", msg

    def _pass(msg):
        return True, "✅ Bien logrado.", msg

    def _heuristic_fail(answer: str, min_words=12, banned=None):
        if banned is None:
            banned = []
        a = (answer or "").strip().lower()
        if len(a.split()) < min_words:
            return True, "La respuesta es demasiado corta. Explica en 2–4 líneas con ideas completas."
        for bad in banned:
            if bad in a:
                return True, f"La idea central no es pertinente: mencionas “{bad}” sin relacionarlo con valoración/CMV/utilidad."
        return False, ""

    def n2_eval_open_ai_q2(answer: str):
        """
        Q2: Importancia de elegir correctamente el método de valoración.
        Criterios (necesita 2 de 3): (a) impacto en CMV/utilidad/bruta,
        (b) estados financieros/comparabilidad, (c) impuestos/decisiones de gestión.
        """
        # 1) Heurística previa
        bad, why = _heuristic_fail(
            answer,
            banned=["gastos financieros", "pasivos", "endeudamiento", "apalancamiento"]
        )
        if bad:
            return _fail(f"{why} Debes relacionar la elección del método con **CMV**, **utilidad**, **estados financieros** e **impuestos/decisiones**.")

        # 2) Llamada IA (request separado, formato rígido)
        prompt = (
            "Evalúa esta respuesta del estudiante.\n"
            "Primera línea DEBE SER EXACTAMENTE 'SCORE: 1' o 'SCORE: 0'.\n"
            "Criterios para SCORE: 1 (cumplir al menos 2):\n"
            "  • Menciona impacto en CMV y/o utilidad (bruta o neta).\n"
            "  • Cita efectos en los estados financieros o comparabilidad.\n"
            "  • Alude a impuestos o decisiones (precios, compras, márgenes).\n"
            "Tras la primera línea, escribe 2–3 líneas de feedback pedagógico (conciso).\n"
            f"RESPUESTA DEL ESTUDIANTE:\n{answer}"
        )
        fb = ia_feedback(prompt) or ""

        # 3) Parser estricto: primera línea
        first = fb.strip().splitlines()[0] if fb.strip() else ""
        m = re.fullmatch(r"SCORE:\s*([01])\s*", first.strip())
        ok = (m and m.group(1) == "1")
        # Feedback corto + ampliado
        fb_short = "✅ Cumple criterios" if ok else "❌ No cumple criterios"
        fb_formativo = "\n".join(fb.strip().splitlines()[1:]) if fb.strip() else ""
        if ok:
            return _pass(fb_formativo)
        else:
            # Feedback mínimo si el modelo no devolvió nada útil
            if not fb_formativo:
                fb_formativo = ("Recuerda conectar el método con el **CMV y la utilidad**, "
                                "su efecto en **estados financieros** y posibles impactos en **impuestos/decisiones**.")
            return _fail(fb_formativo)

    def n2_eval_open_ai_q3(answer: str):
        """
        Q3: Situación en que Promedio Ponderado es más conveniente que PEPS/UEPS.
        Criterios (necesita 2 de 3): (a) compras frecuentes/costos variables,
        (b) suaviza volatilidad del CMV, (c) simplifica operación/registro.
        Debe justificar.
        """
        bad, why = _heuristic_fail(
            answer,
            banned=["pasivos", "gastos financieros", "apalancamiento"]
        )
        if bad:
            return _fail(f"{why} Describe un caso **operativo** (compras frecuentes/costos cambiantes), "
                        "cómo el promedio **suaviza el CMV** y por qué **simplifica** la gestión.")

        prompt = (
            "Evalúa esta respuesta del estudiante.\n"
            "Primera línea DEBE SER EXACTAMENTE 'SCORE: 1' o 'SCORE: 0'.\n"
            "Criterios para SCORE: 1 (cumplir al menos 2):\n"
            "  • Compras frecuentes o costos variables → PP es natural.\n"
            "  • PP reduce la volatilidad del CMV frente a PEPS/UEPS.\n"
            "  • PP simplifica el registro (un costo unitario corriente).\n"
            "Debe haber **justificación** explícita.\n"
            "Tras la primera línea, escribe 2–3 líneas de feedback con un tip.\n"
            f"RESPUESTA DEL ESTUDIANTE:\n{answer}"
        )
        fb = ia_feedback(prompt) or ""

        first = fb.strip().splitlines()[0] if fb.strip() else ""
        m = re.fullmatch(r"SCORE:\s*([01])\s*", first.strip())
        ok = (m and m.group(1) == "1")

        fb_short = "✅ Caso y justificación adecuados" if ok else "❌ Caso/justificación insuficiente"
        fb_formativo = "\n".join(fb.strip().splitlines()[1:]) if fb.strip() else ""
        if ok:
            return _pass(fb_formativo)
        else:
            if not fb_formativo:
                fb_formativo = ("Propón un contexto con **compras frecuentes** y **precios cambiantes**, "
                                "explica que el PP **suaviza el CMV** y resume cómo **simplifica** la operación.")
            return _fail(fb_formativo)

    with tabs[3]:
        st.subheader("Evaluación final del Nivel 2")
        st.caption("Debes acertar **5 de 5** para aprobar y avanzar al siguiente nivel.")

        # ========= Helpers comunes (PEPS dinámico y PP) =========
        def _fmt_money(x):
            try: return peso(float(x))
            except: return str(x)

        def _sum_layers(layers):
            """layers: [[qty, pu], ...]  -> (qty_total, pu_promedio, val_total)"""
            q = sum(q for q, _ in layers)
            v = sum(q*p for q, p in layers)
            pu = (v / q) if q > 0 else 0.0
            return q, pu, v

        # ========= Escenarios diferenciados =========
        # --- Promedio Ponderado (Bolsos)
        inv0_u, inv0_pu = 80, 10.0
        comp1_u, comp1_pu = 40, 11.0
        venta_u = 90
        comp2_u, comp2_pu = 50, 13.0

        # --- PEPS (Camisas)
        peps_inv0_u, peps_inv0_pu = 100, 9.0
        peps_comp1_u, peps_comp1_pu = 60, 10.5
        peps_venta_u = 120
        peps_comp2_u, peps_comp2_pu = 70, 12.0

        # ---------- Solución PP ----------    
        pp_after_c1, pp_after_sale, pp_final = cached_solve_pp(
            inv0_u, inv0_pu, comp1_u, comp1_pu, venta_u, comp2_u, comp2_pu
        )

        # ---------- Solución PEPS (filas esperadas, SIN “Saldo (día 1)”) ----------
        def build_peps_exam_rows():
            """
            Construye las filas esperadas del Kardex PEPS para el examen, con esta lógica:
            - Día 1: saldo inicial como ENTRADA + SALDO.
            - Día 2: solo la fila de Compra 1 (sin 'Saldo (día 1)').
            - Día 3: una fila por tramo de venta, mostrando el saldo como la capa activa
                    (no se promedia el costo). Si una capa se agota en el tramo,
                    el SALDO de esa fila muestra 0 unidades al costo de esa capa.
            - Día 4: Compra 2, el saldo muestra solo la nueva capa comprada.
            """
            rows = []

            # ----------------- Día 1: Saldo inicial como entrada + saldo -----------------
            layers = []
            if peps_inv0_u > 0:
                ent_q = peps_inv0_u
                ent_pu = peps_inv0_pu
                ent_tot = ent_q * ent_pu
                sdo_q = peps_inv0_u
                sdo_pu = peps_inv0_pu
                sdo_tot = ent_tot
                layers.append([float(peps_inv0_u), float(peps_inv0_pu)])
            else:
                ent_q = ent_pu = ent_tot = ""
                sdo_q = sdo_pu = sdo_tot = 0.0

            rows.append({
                "fecha": "Día 1", "desc": "Saldo inicial",
                "ent_q": ent_q if ent_q != "" else "",
                "ent_pu": ent_pu if ent_q != "" else "",
                "ent_tot": ent_tot if ent_q != "" else "",
                "sal_q": "", "sal_pu": "", "sal_tot": "",
                "sdo_q": sdo_q, "sdo_pu": sdo_pu, "sdo_tot": sdo_tot,
            })

            # ----------------- Día 2: Compra 1 (nueva capa) -----------------
            if peps_comp1_u > 0:
                ent_q2 = peps_comp1_u
                ent_pu2 = peps_comp1_pu
                ent_tot2 = ent_q2 * ent_pu2
                rows.append({
                    "fecha": "Día 2", "desc": "Compra 1",
                    "ent_q": ent_q2, "ent_pu": ent_pu2, "ent_tot": ent_tot2,
                    "sal_q": "", "sal_pu": "", "sal_tot": "",
                    # En SALDO se muestra solo la capa comprada
                    "sdo_q": ent_q2, "sdo_pu": ent_pu2, "sdo_tot": ent_tot2,
                })
                layers.append([float(peps_comp1_u), float(peps_comp1_pu)])
            else:
                rows.append({
                    "fecha": "Día 2", "desc": "Compra 1",
                    "ent_q": "", "ent_pu": "", "ent_tot": "",
                    "sal_q": "", "sal_pu": "", "sal_tot": "",
                    "sdo_q": 0, "sdo_pu": 0.0, "sdo_tot": 0.0,
                })

            # ----------------- Día 3: Venta en tramos (PEPS, sin promediar) -----------------
            total_inventario = sum(q for q, _ in layers)
            if peps_venta_u > 0 and total_inventario > 0:
                layers_for_calc = [[float(q), float(p)] for q, p in layers]
                sale_remaining = float(peps_venta_u)
                tramo_index = 1

                while sale_remaining > 0 and any(q > 0 for q, _ in layers_for_calc):
                    # En PEPS se toma siempre la capa más antigua con unidades disponibles
                    idx_layer = next(i for i, (q, _) in enumerate(layers_for_calc) if q > 0)
                    layer_q, layer_pu = layers_for_calc[idx_layer]
                    q_take = min(layer_q, sale_remaining)
                    tot_take = q_take * layer_pu
                    sale_remaining -= q_take

                    # Actualizar capa consumida
                    q_rem = layer_q - q_take
                    layers_for_calc[idx_layer][0] = q_rem

                    # 💡 NUEVA LÓGICA: SALDO de la fila muestra SIEMPRE la capa del tramo
                    if q_rem > 0:
                        # Quedan unidades en la misma capa que se está consumiendo
                        sdo_q = q_rem
                        sdo_pu = layer_pu
                    else:
                        # La capa se agotó en este tramo: saldo 0 unidades al mismo costo
                        sdo_q = 0.0
                        sdo_pu = layer_pu
                    sdo_tot = sdo_q * sdo_pu

                    rows.append({
                        "fecha": "Día 3",
                        "desc": f"Venta tramo {tramo_index} (PEPS)",
                        "ent_q": "", "ent_pu": "", "ent_tot": "",
                        "sal_q": q_take, "sal_pu": layer_pu, "sal_tot": tot_take,
                        "sdo_q": sdo_q, "sdo_pu": sdo_pu, "sdo_tot": sdo_tot,
                    })

                    tramo_index += 1

                # Actualizamos las capas finales (si quisieras usarlas luego)
                layers = layers_for_calc
            else:
                # Escenario sin venta efectiva
                q_tot, pu_tot, v_tot = _sum_layers(layers) if layers else (0, 0.0, 0.0)
                rows.append({
                    "fecha": "Día 3",
                    "desc": "Venta",
                    "ent_q": "", "ent_pu": "", "ent_tot": "",
                    "sal_q": "", "sal_pu": "", "sal_tot": "",
                    "sdo_q": q_tot, "sdo_pu": pu_tot, "sdo_tot": v_tot,
                })

            # ----------------- Día 4: Compra 2 (nueva capa) -----------------
            if peps_comp2_u > 0:
                ent_q4 = peps_comp2_u
                ent_pu4 = peps_comp2_pu
                ent_tot4 = ent_q4 * ent_pu4
                rows.append({
                    "fecha": "Día 4", "desc": "Compra 2",
                    "ent_q": ent_q4, "ent_pu": ent_pu4, "ent_tot": ent_tot4,
                    "sal_q": "", "sal_pu": "", "sal_tot": "",
                    # Igual que en el ejemplo guiado: el saldo de esta fila muestra solo la nueva capa
                    "sdo_q": ent_q4, "sdo_pu": ent_pu4, "sdo_tot": ent_tot4,
                })
                layers.append([float(peps_comp2_u), float(peps_comp2_pu)])
            else:
                rows.append({
                    "fecha": "Día 4", "desc": "Compra 2",
                    "ent_q": "", "ent_pu": "", "ent_tot": "",
                    "sal_q": "", "sal_pu": "", "sal_tot": "",
                    "sdo_q": 0, "sdo_pu": 0.0, "sdo_tot": 0.0,
                })

            return rows

        peps_rows_expected = build_peps_exam_rows()

        # ========= UI: una sola FORM para todo =========
        with st.form("n2_eval_all"):
            # 1) Selección múltiple
            st.markdown("### 1) Selección múltiple (1 punto)")
            st.markdown(
                "¿Cuál de los siguientes métodos de valoración de inventarios genera normalmente una **mayor utilidad bruta** cuando los precios de compra están en aumento?"
            )
            q1 = st.radio(
                "Elige una opción:",
                [
                    "A) UEPS (Último en Entrar, Primero en Salir)",
                    "B) PEPS (Primero en Entrar, Primero en Salir)",
                    "C) Promedio Ponderado",
                    "D) Ninguno, todos generan la misma utilidad",
                ],
                index=None,
                key="n2_eval_q1_new",
            )

            st.markdown("---")

            # 2) Preguntas abiertas
            st.markdown("### 2) Preguntas abiertas (2 puntos)")
            a1 = st.text_area(
                "2) Explica, con tus propias palabras, por qué es importante elegir correctamente el método de valoración de inventarios en una empresa.",
                key="n2_eval_a1", height=130
            )
            a2 = st.text_area(
                "3) Describe una situación práctica en la que el método del **Promedio Ponderado** podría ser más conveniente que PEPS o UEPS, y justifica tu respuesta.",
                key="n2_eval_a2", height=130
            )
            ask_ai_open = st.checkbox("💬 Pedir calificación y feedback de IA para las preguntas abiertas", value=True)

            st.markdown("---")

            # 3) Ejercicio PP (tabla fija 4 filas)
            st.markdown("### 3) Ejercicio (1 punto): Promedio Ponderado")
            st.markdown(
                f"""
                **Contexto:**  
                La empresa **ABC S.A.S.**, dedicada a la venta de bolsos, desea calcular el costo de su inventario bajo el método **Promedio Ponderado**.  
                A continuación se describe el movimiento de mercancías durante la semana:

                - **Día 1:** Inventario inicial de **{inv0_u} bolsos** a un costo unitario de **${inv0_pu:,.2f}**.  
                - **Día 2:** Compra de **{comp1_u} bolsos adicionales** a **${comp1_pu:,.2f}** cada uno.  
                - **Día 3:** Venta de **{venta_u} bolsos**.  
                - **Día 4:** Nueva compra de **{comp2_u} bolsos** a **${comp2_pu:,.2f}** por unidad.  

                Completa el siguiente **Kardex**, aplicando correctamente el método de **Promedio Ponderado**.
                """
            )

            pp_template = pd.DataFrame([
                {"Fecha":"Día 1","Descripción":"Saldo inicial",
                "Entrada_cant":None,"Entrada_pu":None,"Entrada_total":None,
                "Salida_cant":None,"Salida_pu":None,"Salida_total":None,
                "Saldo_cant":None,"Saldo_pu":None,"Saldo_total":None},
                {"Fecha":"Día 2","Descripción":"Compra 1",
                "Entrada_cant":None,"Entrada_pu":None,"Entrada_total":None,
                "Salida_cant":None,"Salida_pu":None,"Salida_total":None,
                "Saldo_cant":None,"Saldo_pu":None,"Saldo_total":None},
                {"Fecha":"Día 3","Descripción":"Venta",
                "Entrada_cant":None,"Entrada_pu":None,"Entrada_total":None,
                "Salida_cant":None,"Salida_pu":None,"Salida_total":None,
                "Saldo_cant":None,"Saldo_pu":None,"Saldo_total":None},
                {"Fecha":"Día 4","Descripción":"Compra 2",
                "Entrada_cant":None,"Entrada_pu":None,"Entrada_total":None,
                "Salida_cant":None,"Salida_pu":None,"Salida_total":None,
                "Saldo_cant":None,"Saldo_pu":None,"Saldo_total":None},
            ])
            pp_edit = st.data_editor(
                pp_template.astype("string"),
                use_container_width=True,
                num_rows="fixed",
                key="n2_eval_pp_tbl"
            )

            st.markdown("---")

            # 4) Ejercicio PEPS (tabla dinámica por filas esperadas)
            st.markdown("### 4) Ejercicio (1 punto): PEPS (FIFO)")
            st.markdown(
                f"""
                **Contexto:**  
                La empresa **MODA JOVEN S.A.S.**, dedicada a la comercialización de camisas, aplica el método **PEPS (Primero en Entrar, Primero en Salir)** para valorar su inventario.  
                Durante la semana ocurrieron los siguientes movimientos:

                - **Día 1:** Inventario inicial de **{peps_inv0_u} camisas** a un costo unitario de **${peps_inv0_pu:,.2f}**.  
                - **Día 2:** Compra de **{peps_comp1_u} camisas** a **${peps_comp1_pu:,.2f}** cada una.  
                - **Día 3:** Venta de **{peps_venta_u} camisas**.  
                - **Día 4:** Nueva compra de **{peps_comp2_u} camisas** a **${peps_comp2_pu:,.2f}** por unidad.  

                Diligencia el siguiente **Kardex PEPS**, asegurando que las salidas respeten el orden cronológico de entrada.
                """
            )

            # Construimos plantilla con la MISMA cantidad de filas esperadas
            peps_cols = ["Fecha","Descripción",
                        "Entrada_cant","Entrada_pu","Entrada_total",
                        "Salida_cant","Salida_pu","Salida_total",
                        "Saldo_cant","Saldo_pu","Saldo_total"]
            peps_template = pd.DataFrame([{k: "" for k in peps_cols} for _ in range(len(peps_rows_expected))])
            for i, r in enumerate(peps_rows_expected):
                peps_template.loc[i, "Fecha"] = r["fecha"]
                peps_template.loc[i, "Descripción"] = r["desc"]

            peps_edit = st.data_editor(
                peps_template.astype("string"),
                use_container_width=True,
                num_rows="fixed",
                key="n2_eval_peps_tbl"
            )

            # ===== Submit único
            submitted_all = st.form_submit_button("🧪 Validar evaluación N2")

        # ========= Evaluación / Puntuación =========
        if submitted_all:
            total_score = 0
            details_msgs = []

            # --- MCQ
            correct_label = "B) PEPS (Primero en Entrar, Primero en Salir)"
            mcq_ok = (q1 == correct_label)
            total_score += 1 if mcq_ok else 0
            details_msgs.append(f"Selección múltiple: {'✅' if mcq_ok else '❌'}")

            # --- Abiertas (cada una en su request, con heurística previa y parser estricto)
            ok_a1, fb1_short, fb1_formativo = n2_eval_open_ai_q2(a1 or "")
            ok_a2, fb2_short, fb2_formativo = n2_eval_open_ai_q3(a2 or "")
            total_score += (1 if ok_a1 else 0) + (1 if ok_a2 else 0)
            details_msgs.append(f"Pregunta abierta 2: {'✅' if ok_a1 else '❌'}")
            details_msgs.append(f"Pregunta abierta 3: {'✅' if ok_a2 else '❌'}")

            # --- Ejercicio PP (validación clave de saldos por día)
            def _get_num_safe(row, key):
                v = row.get(key, "")
                try:
                    if v in (None, ""): return None
                    return float(v)
                except:
                    return None

            tol = 0.5
            def near(a,b):
                return (a is not None) and (abs(a-b) <= tol)

            # Esperados PP
            pp_r1_q = inv0_u
            pp_r1_tot = inv0_u * inv0_pu
            pp_r2_q, pp_r2_pu, pp_r2_tot = pp_after_c1[0], pp_after_c1[1], pp_after_c1[2]
            pp_r3_q, pp_r3_pu, pp_r3_tot = pp_after_sale[0], pp_after_sale[1], pp_after_sale[2]
            pp_r4_q, pp_r4_pu, pp_r4_tot = pp_final[0], pp_final[1], pp_final[2]

            try:
                r1 = pp_edit.iloc[0].to_dict()
                r2 = pp_edit.iloc[1].to_dict()
                r3 = pp_edit.iloc[2].to_dict()
                r4 = pp_edit.iloc[3].to_dict()

                ok_pp = (
                    near(_get_num_safe(r1,"Saldo_cant"), pp_r1_q) and
                    near(_get_num_safe(r1,"Saldo_total"), pp_r1_tot) and
                    near(_get_num_safe(r2,"Saldo_cant"), pp_r2_q) and
                    ( (_get_num_safe(r2,"Saldo_pu") is None) or near(_get_num_safe(r2,"Saldo_pu"), pp_r2_pu) ) and
                    near(_get_num_safe(r2,"Saldo_total"), pp_r2_tot) and
                    near(_get_num_safe(r3,"Saldo_cant"), pp_r3_q) and
                    ( (_get_num_safe(r3,"Saldo_pu") is None) or near(_get_num_safe(r3,"Saldo_pu"), pp_r3_pu) ) and
                    near(_get_num_safe(r3,"Saldo_total"), pp_r3_tot) and
                    near(_get_num_safe(r4,"Saldo_cant"), pp_r4_q) and
                    ( (_get_num_safe(r4,"Saldo_pu") is None) or near(_get_num_safe(r4,"Saldo_pu"), pp_r4_pu) ) and
                    near(_get_num_safe(r4,"Saldo_total"), pp_r4_tot)
                )
            except Exception:
                ok_pp = False

            total_score += 1 if ok_pp else 0
            details_msgs.append(f"Ejercicio PP: {'✅' if ok_pp else '❌'}")

            # --- Ejercicio PEPS (validación por fila esperada, con tolerancia)
            ok_peps = True
            try:
                for i, exp in enumerate(peps_rows_expected):
                    row = peps_edit.iloc[i].to_dict()
                    # Entrada (si aplica)
                    if exp["ent_q"] != "":
                        ok_peps &= near(_get_num_safe(row,"Entrada_cant"), exp["ent_q"])
                        ok_peps &= near(_get_num_safe(row,"Entrada_pu"), exp["ent_pu"])
                        ok_peps &= near(_get_num_safe(row,"Entrada_total"), exp["ent_tot"])
                    # Salida (si aplica)
                    if exp["sal_q"] != "":
                        ok_peps &= near(_get_num_safe(row,"Salida_cant"), exp["sal_q"])
                        ok_peps &= near(_get_num_safe(row,"Salida_pu"), exp["sal_pu"])
                        ok_peps &= near(_get_num_safe(row,"Salida_total"), exp["sal_tot"])
                    # Saldo
                    ok_peps &= ( (_get_num_safe(row,"Saldo_cant") is None) or near(_get_num_safe(row,"Saldo_cant"), exp["sdo_q"]) )
                    ok_peps &= ( (_get_num_safe(row,"Saldo_pu")   is None) or near(_get_num_safe(row,"Saldo_pu"),   exp["sdo_pu"]) )
                    ok_peps &= ( (_get_num_safe(row,"Saldo_total")is None) or near(_get_num_safe(row,"Saldo_total"),exp["sdo_tot"]) )
            except Exception:
                ok_peps = False

            total_score += 1 if ok_peps else 0
            details_msgs.append(f"Ejercicio PEPS: {'✅' if ok_peps else '❌'}")

            # ===== Resultado final =====
            st.markdown("---")
            st.metric("Puntaje total", f"{total_score}/5")
            st.write(" | ".join(details_msgs))

            passed = (total_score >= 5)
            record_attempt(username, level=2, score=total_score, passed=passed)

            if passed:
                set_level_passed(st.session_state["progress_col"], username, "level2", total_score)
                st.session_state["sidebar_next_select"] = "Nivel 3: Devoluciones"
                start_celebration(
                    message_md=(
                        "<b>¡Nivel 2 aprobado!</b> 🎉<br><br>"
                        "Dominaste <b>PP / PEPS / UEPS</b> en teoría y práctica. "
                        "Sigamos con el <b>Nivel 3</b>: <i>Devoluciones</i>."
                    ),
                    next_label="Ir al Nivel 3",
                    next_key_value="Nivel 3: Devoluciones"
                )
            else:
                st.error("No aprobado. Necesitas 5/5. Revisa tus respuestas y vuelve a intentarlo.")
                if ask_ai_open:
                    with st.expander("💬 Feedback de la IA (preguntas abiertas)"):
                        st.markdown("**Pregunta 2**")
                        st.write(fb1_short)
                        st.info(fb1_formativo)
                        st.markdown("---")
                        st.markdown("**Pregunta 3**")
                        st.write(fb2_short)
                        st.info(fb2_formativo)

# ===========================
# NIVEL 3 (Devoluciones)
# ===========================
def page_level3(username):
    st.title("Nivel 3 · Casos con Devoluciones (compras y ventas)")

    tabs = st.tabs(["🎧 Teoría", "🛠 Ejemplos guiados", "🎮 Práctica (IA)", "🏁 Evaluación para aprobar"])

    # =========================================
    # TAB 0 · TEORÍA
    # =========================================
    with tabs[0]:
        st.subheader("Teoría · Devoluciones en compras y ventas (PP, PEPS y UEPS)")

        intro = """
    En inventarios, las **devoluciones** ajustan movimientos ya registrados y afectan directamente el **Costo de la Mercancía Vendida (CMV)** y la utilidad del periodo. 
    Hay dos tipos: **devoluciones en compras** (a proveedor) y **devoluciones en ventas** (del cliente a la empresa). 
    Comprender su tratamiento —siempre consistente con el método de valoración (Promedio, PEPS, UEPS)— evita distorsiones en los estados financieros.
    """
        st.markdown(intro)

        st.markdown("### 🛒 Devoluciones en compras")
        st.markdown(
            "- **Qué son:** mercancía que la empresa **devuelve al proveedor** (defecto, exceso, inconformidad).  \n"
            "- **Efecto contable:** **disminuyen** el total de compras; por ende, reducen el **pool** de costo disponible para vender.  \n"
            "- **Compras netas:** se calculan restando las devoluciones de las compras brutas."
        )
        st.markdown("**Fórmula:**")
        st.latex(r"\text{Compras netas} \;=\; \text{Compras brutas} \;-\; \text{Devoluciones en compras}")

        st.markdown("---")
        st.markdown("### 🧾 Devoluciones en ventas")
        st.markdown(
            "- **Qué son:** el **cliente devuelve** unidades previamente vendidas; estas **reingresan al inventario**.  \n"
            "- **Efecto en el CMV:** el reingreso **reduce el CMV neto** del periodo (parte del costo reconocido como vendido regresa al inventario).  \n"
            "- **¿Cómo se valora el reingreso?** Siempre debe hacerse al **mismo costo con el que salieron** esas unidades, de forma coherente con el método aplicado."
        )
        st.markdown(
            "  - **Promedio Ponderado (PP):** las devoluciones revierten el **costo promedio** que se usó en la operación original "
            "(no el promedio vigente en una fecha distinta). Ese costo se suma de nuevo al inventario y se recalcula un **nuevo promedio**.\n"
            "  - **PEPS (FIFO):** se respeta la cronología: las primeras capas son las que salen primero; las devoluciones se reconocen con "
            "los **costos de esas mismas capas** desde las que se originó la venta.\n"
            "  - **UEPS (LIFO):** salen primero las capas más recientes; las devoluciones se reconocen con los **costos de las capas recientes** "
            "que dieron origen a la salida."
        )

        st.markdown("**Relación simple con el CMV neto:**")
        st.latex(r"\text{COGS neto} \;\approx\; \text{COGS bruto} \;-\; \text{Costo de unidades devueltas (reingresadas)}")

        st.markdown("---")
        st.markdown("### 🔎 Coherencia con el método (PP, PEPS, UEPS)")
        st.markdown(
            "- **PP:** todo se valora a un **costo promedio**; las devoluciones en ventas **revierten el mismo promedio** que se usó para el CMV "
            "de la venta original y, al reingresar, se recalcula un **nuevo promedio** del saldo.\n"
            "- **PEPS:** se respeta la **cronología**: las primeras capas (más antiguas) son las que salen primero; las devoluciones en ventas "
            "se reconocen con los **costos de esas mismas capas** de donde salieron.\n"
            "- **UEPS:** salen primero las capas más recientes; las devoluciones en ventas se reconocen con los **costos de las capas recientes** "
            "que originaron la salida.\n"
            "- **Nota en Colombia:** el **UEPS no es aceptado fiscalmente**, pero se usa en análisis internos o ejercicios académicos."
        )

        st.markdown("---")
        st.markdown("### 💡 Idea clave")
        st.markdown(
            "Registra siempre las devoluciones de forma **consistente con el método de inventario** y al **costo con el que salieron**. "
            "Así mantienes el **CMV**, el **saldo de inventario** y la **utilidad** correctamente medidos, "
            "facilitando decisiones y la comparabilidad entre periodos."
        )

        with st.expander("🔊 Escuchar explicación"):
            full_text = "\n\n".join([
                intro.strip(),
                "Devoluciones en compras: disminuyen las compras y el pool de costo disponible.",
                "Devoluciones en ventas: reingresan unidades al costo con el que salieron y reducen el COGS neto.",
                "Coherencia con PP/PEPS/UEPS y nota sobre UEPS en Colombia.",
                "Idea clave: consistencia en el costo aplicado a la salida y a la devolución."
            ])
            speak_block(full_text, key_prefix="teo-n3", lang_hint="es")

    # =========================================
    # TAB 1 · EJEMPLO GUIADO (KARDEX DINÁMICO)
    # =========================================
    with tabs[1]:
        st.subheader("KARDEX dinámico con devoluciones (PP · PEPS · UEPS)")
        st.caption(
            "Días 1–3 prellenados según el método. "
            "La demostración narrada se centra en el Día 4 (devolución de compra) y Día 5 (devolución de venta)."
        )

        # ========= Helpers =========
        def _fmt_money(v):
            try:
                return peso(float(v))
            except Exception:
                return str(v)

        def _sum_layers(layers):
            q = sum(q for q, _ in layers)
            v = sum(q * p for q, p in layers)
            pu = (v / q) if q > 0 else 0.0
            return q, pu, v

        def _consume_layers_detail(layers, qty_out, fifo=True):
            """
            Devuelve (sale_details, layers_after)
            sale_details: lista de tramos [(take_qty, pu, total), ...]
            layers_after: capas restantes tras consumir qty_out
            """
            order = layers[:] if fifo else layers[::-1]
            remaining = qty_out
            sale_details = []
            new_layers = []
            for q, pu in order:
                if remaining <= 0:
                    new_layers.append([q, pu])
                    continue
                take = min(q, remaining)
                if take > 0:
                    sale_details.append((take, pu, take * pu))
                    rest = q - take
                    remaining -= take
                    if rest > 0:
                        new_layers.append([rest, pu])
            layers_after = new_layers if fifo else new_layers[::-1]
            return sale_details, layers_after

        # ========= Parámetros del escenario =========
        st.markdown("#### Parámetros del escenario")

        st.markdown("**Método de valoración**")
        metodo = st.selectbox(
            "Método",
            ["Promedio Ponderado", "PEPS (FIFO)", "UEPS (LIFO)"],
            key="n3_kx_met",
            label_visibility="collapsed",
        )

        # Días 1–3
        st.markdown("📦 **Día 1.** Saldo inicial:")
        c1a, c1b = st.columns([1, 1], gap="small")
        with c1a:
            inv0_u = st.number_input(
                "Cantidad (u)", min_value=0, value=100, step=10, key="n3_kx_inv_u"
            )
        with c1b:
            inv0_pu = st.number_input(
                "Costo unitario ($/u)",
                min_value=0.0,
                value=10.0,
                step=0.5,
                key="n3_kx_inv_pu",
            )

        st.markdown("🛒 **Día 2.** Compra:")
        c2a, c2b = st.columns([1, 1], gap="small")
        with c2a:
            comp_u = st.number_input(
                "Compra (u)", min_value=0, value=60, step=10, key="n3_kx_comp_u"
            )
        with c2b:
            comp_pu = st.number_input(
                "Costo compra ($/u)",
                min_value=0.0,
                value=12.0,
                step=0.5,
                key="n3_kx_comp_pu",
            )

        st.markdown("💰 **Día 3.** Venta:")
        venta_u = st.number_input(
            "Venta (u)", min_value=0, value=120, step=10, key="n3_kx_venta_u"
        )

        st.markdown("---")
        st.markdown("↩️ **Día 4.** Devolución de compra (a proveedor):")
        dev_comp_u = st.number_input(
            "Unidades devueltas al proveedor",
            min_value=0,
            value=10,
            step=5,
            key="n3_kx_dev_comp_u",
        )
        st.caption(
            "La devolución se valora al **mismo costo unitario de la compra original**. "
            "En PP ajusta el promedio; en PEPS/UEPS se retiran unidades de la capa de esa compra."
        )

        st.markdown("↪️ **Día 5.** Devolución de venta (del cliente):")
        dev_venta_u = st.number_input(
            "Unidades devueltas por el cliente",
            min_value=0,
            value=8,
            step=2,
            key="n3_kx_dev_venta_u",
        )
        st.caption(
            "La devolución se valora al **mismo costo unitario con el que salieron en la venta original**. "
            "En PEPS/UEPS se reingresan a esa misma capa, sin promediar."
        )

        st.markdown("---")
        st.markdown("### 🎬 Demostración narrada (arranca en Día 4 y 5)")

        c_demo_a, c_demo_b = st.columns([1, 1])
        with c_demo_a:
            narr_speed = st.slider(
                "Velocidad de narración", 0.75, 1.50, 1.00, 0.05, key="n3_kx_rate"
            )
        with c_demo_b:
            narr_muted = st.toggle(
                "Silenciar voz", value=False, key="n3_kx_mute"
            )

        # ========= Generación de filas y guion =========
        def compute_rows_and_script_with_returns(
            method_name,
            inv0_u,
            inv0_pu,
            comp_u,
            comp_pu,
            venta_u,
            dev_comp_u,
            dev_venta_u,
        ):
            """
            Construye todas las filas (Día 1–5).
            Días 1–3: prellenados según el método.
            Días 4–5: se narran (acciones en script).
            En PEPS/UEPS NUNCA se promedia: se trabaja siempre por capas.
            """
            rows = []
            script = []

            # Costo unitario para usar en la devolución de venta
            sale_unit_cost = None

            # =====================
            # DÍA 1 · SALDO INICIAL
            # =====================
            if inv0_u > 0:
                ent_q_1 = int(inv0_u)
                ent_pu_1 = float(inv0_pu)
                ent_tot_1 = ent_q_1 * ent_pu_1
                layers = [[float(ent_q_1), float(ent_pu_1)]]
                s_q = float(ent_q_1)
                s_pu = ent_pu_1
                s_v = ent_tot_1
            else:
                ent_q_1 = None
                ent_pu_1 = None
                ent_tot_1 = None
                layers = []
                s_q, s_pu, s_v = 0.0, 0.0, 0.0

            rows.append(
                {
                    "fecha": "Día 1",
                    "desc": "Saldo inicial",
                    "ent_q": ent_q_1,
                    "ent_pu": ent_pu_1,
                    "ent_tot": ent_tot_1,
                    "sal_q": None,
                    "sal_pu": None,
                    "sal_tot": None,
                    "sdo_q": int(s_q) if s_q > 0 else 0,
                    "sdo_pu": round(s_pu, 2),
                    "sdo_tot": round(s_v, 2),
                }
            )

            # =====================
            # DÍA 2 · COMPRA
            # =====================
            if method_name == "Promedio Ponderado":
                ent_tot = comp_u * comp_pu

                if comp_u > 0:
                    new_q = s_q + comp_u
                    new_v = s_v + ent_tot
                    new_p = (new_v / new_q) if new_q > 0 else 0.0
                    layers = [[new_q, new_p]]
                    s_q, s_pu, s_v = new_q, new_p, new_v

                rows.append(
                    {
                        "fecha": "Día 2",
                        "desc": "Compra",
                        "ent_q": int(comp_u) if comp_u > 0 else None,
                        "ent_pu": round(comp_pu, 2) if comp_u > 0 else None,
                        "ent_tot": round(ent_tot, 2) if comp_u > 0 else None,
                        "sal_q": None,
                        "sal_pu": None,
                        "sal_tot": None,
                        "sdo_q": int(s_q),
                        "sdo_pu": round(s_pu, 2),
                        "sdo_tot": round(s_v, 2),
                    }
                )
            else:
                # PEPS / UEPS: solo fila Compra; el saldo de esa fila muestra solo la capa comprada
                ent_tot = comp_u * comp_pu
                if comp_u > 0:
                    layers.append([float(comp_u), float(comp_pu)])
                s_q_total, s_pu_total, s_v_total = _sum_layers(layers)

                rows.append(
                    {
                        "fecha": "Día 2",
                        "desc": "Compra",
                        "ent_q": int(comp_u) if comp_u > 0 else None,
                        "ent_pu": round(comp_pu, 2) if comp_u > 0 else None,
                        "ent_tot": round(ent_tot, 2) if comp_u > 0 else None,
                        "sal_q": None,
                        "sal_pu": None,
                        "sal_tot": None,
                        # Saldo mostrado: solamente la capa de la compra
                        "sdo_q": int(comp_u) if comp_u > 0 else 0,
                        "sdo_pu": round(comp_pu, 2) if comp_u > 0 else 0.0,
                        "sdo_tot": round(ent_tot, 2) if comp_u > 0 else 0.0,
                    }
                )
                s_q, s_pu, s_v = s_q_total, s_pu_total, s_v_total

            # =====================
            # DÍA 3 · VENTA (PRELLENADA)
            # =====================
            if venta_u > 0 and s_q > 0:
                # ------- PROMEDIO PONDERADO -------
                if method_name == "Promedio Ponderado":
                    sal_q = min(venta_u, s_q)
                    sal_pu = layers[0][1] if layers else 0.0
                    sale_unit_cost = sal_pu  # costo de esa venta (para devolución)
                    sal_tot = sal_q * sal_pu

                    new_q = s_q - sal_q
                    new_v = s_v - sal_tot
                    new_p = (new_v / new_q) if new_q > 0 else 0.0
                    layers = [[new_q, new_p]] if new_q > 0 else []
                    s_q, s_pu, s_v = new_q, new_p, new_v

                    rows.append(
                        {
                            "fecha": "Día 3",
                            "desc": "Venta",
                            "ent_q": None,
                            "ent_pu": None,
                            "ent_tot": None,
                            "sal_q": int(sal_q),
                            "sal_pu": round(sal_pu, 2),
                            "sal_tot": round(sal_tot, 2),
                            "sdo_q": int(s_q),
                            "sdo_pu": round(s_pu, 2),
                            "sdo_tot": round(s_v, 2),
                        }
                    )

                # ------- PEPS / UEPS (SIN PROMEDIAR) -------
                else:
                    fifo = method_name == "PEPS (FIFO)"
                    metodo_tag = "PEPS" if fifo else "UEPS"

                    # Simulamos por capas SIN promediar, tramo a tramo
                    sim_layers = [l[:] for l in layers]
                    remaining_sale = venta_u
                    sale_details = []

                    while remaining_sale > 0 and sim_layers:
                        idx = 0 if fifo else len(sim_layers) - 1
                        q_layer, pu_layer = sim_layers[idx]
                        take = min(q_layer, remaining_sale)
                        if take <= 0:
                            break

                        tot_take = take * pu_layer
                        sale_details.append((take, pu_layer, tot_take))

                        new_q_layer = q_layer - take
                        if new_q_layer > 0:
                            sim_layers[idx][0] = new_q_layer
                        else:
                            sim_layers.pop(idx)

                        # Saldo mostrado SOLO de la capa afectada en este tramo
                        saldo_q = new_q_layer if new_q_layer > 0 else 0
                        saldo_pu = pu_layer
                        saldo_tot = saldo_q * saldo_pu

                        rows.append(
                            {
                                "fecha": "Día 3",
                                "desc": f"Venta tramo {len(sale_details)} ({metodo_tag})",
                                "ent_q": None,
                                "ent_pu": None,
                                "ent_tot": None,
                                "sal_q": int(take),
                                "sal_pu": round(pu_layer, 2),
                                "sal_tot": round(tot_take, 2),
                                "sdo_q": int(saldo_q),
                                "sdo_pu": round(saldo_pu, 2),
                                "sdo_tot": round(saldo_tot, 2),
                            }
                        )

                        remaining_sale -= take

                    layers = sim_layers
                    s_q, s_pu, s_v = _sum_layers(layers)

                    # Costo de referencia para devolución en ventas
                    if sale_details:
                        sale_unit_cost = (
                            sale_details[0][1] if fifo else sale_details[-1][1]
                        )
                    else:
                        sale_unit_cost = None
            else:
                rows.append(
                    {
                        "fecha": "Día 3",
                        "desc": "Venta",
                        "ent_q": None,
                        "ent_pu": None,
                        "ent_tot": None,
                        "sal_q": None,
                        "sal_pu": None,
                        "sal_tot": None,
                        "sdo_q": int(s_q),
                        "sdo_pu": round(s_pu, 2),
                        "sdo_tot": round(s_v, 2),
                    }
                )

            # A partir de aquí se narran solo las devoluciones
            narr_start_idx = len(rows)

            # =====================
            # DÍA 4 · DEVOLUCIÓN DE COMPRA
            # =====================
            if dev_comp_u > 0 and s_q > 0:
                if method_name == "Promedio Ponderado":
                    take_q = min(dev_comp_u, s_q)
                    # Devolvemos al precio original de compra
                    take_pu = comp_pu
                    take_val = take_q * take_pu

                    new_q = max(s_q - take_q, 0)
                    new_v = max(s_v - take_val, 0.0)
                    new_p = (new_v / new_q) if new_q > 0 else 0.0
                    layers = [[new_q, new_p]] if new_q > 0 else []
                    s_q, s_pu, s_v = new_q, new_p, new_v

                    rows.append(
                        {
                            "fecha": "Día 4",
                            "desc": "Devolución de compra",
                            "ent_q": None,
                            "ent_pu": None,
                            "ent_tot": None,
                            "sal_q": int(take_q),
                            "sal_pu": round(take_pu, 2),
                            "sal_tot": round(take_val, 2),
                            "sdo_q": int(s_q),
                            "sdo_pu": round(s_pu, 2),
                            "sdo_tot": round(s_v, 2),
                        }
                    )

                    script.append(
                        {
                            "title": "Día 4 · Devolución de compra (PP)",
                            "text": (
                                f"En el Día 4 registramos una **devolución de compra**: devolvemos "
                                f"{int(take_q)} unidades al proveedor al mismo costo de compra "
                                f"({_fmt_money(take_pu)} por unidad). Esto reduce el inventario y se "
                                f"recalcula el costo promedio del saldo."
                            ),
                            "actions": [
                                {
                                    "row": len(rows) - 1,
                                    "cell": "sal_q",
                                    "money": False,
                                    "val": int(take_q),
                                },
                                {
                                    "row": len(rows) - 1,
                                    "cell": "sal_pu",
                                    "money": True,
                                    "val": round(take_pu, 2),
                                },
                                {
                                    "row": len(rows) - 1,
                                    "cell": "sal_tot",
                                    "money": True,
                                    "val": round(take_val, 2),
                                },
                                {
                                    "row": len(rows) - 1,
                                    "cell": "sdo_q",
                                    "money": False,
                                    "val": int(s_q),
                                },
                                {
                                    "row": len(rows) - 1,
                                    "cell": "sdo_pu",
                                    "money": True,
                                    "val": round(s_pu, 2),
                                },
                                {
                                    "row": len(rows) - 1,
                                    "cell": "sdo_tot",
                                    "money": True,
                                    "val": round(s_v, 2),
                                },
                            ],
                        }
                    )
                else:
                    # PEPS / UEPS: la devolución en compras sale SIEMPRE de la última compra (LIFO),
                    # pero el método solo cambia la narrativa.
                    fifo = method_name == "PEPS (FIFO)"
                    metodo_tag = "PEPS" if fifo else "UEPS"

                    # Consumimos desde las últimas capas (LIFO) para simular la devolución de la compra
                    sale_details, layers_after = _consume_layers_detail(
                        layers, dev_comp_u, fifo=False  # siempre desde la compra más reciente
                    )
                    layers = layers_after
                    s_q, s_pu, s_v = _sum_layers(layers)

                    take_q = sum(q for q, _, _ in sale_details)
                    take_val = sum(t for _, _, t in sale_details)
                    take_pu = (take_val / take_q) if take_q > 0 else 0.0

                    # Para mostrar el SALDO en la fila:
                    # buscamos la(s) capa(s) que quedaron con ese mismo costo de compra
                    eps = 1e-6
                    saldo_q = 0.0
                    for q, p in layers:
                        if abs(p - take_pu) < eps:
                            saldo_q += q
                    saldo_tot = saldo_q * take_pu

                    rows.append(
                        {
                            "fecha": "Día 4",
                            "desc": "Devolución de compra",
                            "ent_q": None,
                            "ent_pu": None,
                            "ent_tot": None,
                            "sal_q": int(take_q),
                            "sal_pu": round(take_pu, 2),
                            "sal_tot": round(take_val, 2),
                            "sdo_q": int(saldo_q),
                            "sdo_pu": round(take_pu, 2),
                            "sdo_tot": round(saldo_tot, 2),
                        }
                    )

                    script.append(
                        {
                            "title": f"Día 4 · Devolución de compra ({metodo_tag})",
                            "text": (
                                f"Registramos una **devolución de compra**: retiramos "
                                f"{int(take_q)} unidades de la capa de la compra que estamos devolviendo, "
                                f"al costo en que fueron compradas ({_fmt_money(take_pu)} por unidad).\n\n"
                                f"En el SALDO del Día 4 ves cuántas unidades quedan en esa misma capa y con el mismo costo; "
                                f"no promediamos, respetamos siempre el valor de la capa."
                            ),
                            "actions": [
                                {
                                    "row": len(rows) - 1,
                                    "cell": "sal_q",
                                    "money": False,
                                    "val": int(take_q),
                                },
                                {
                                    "row": len(rows) - 1,
                                    "cell": "sal_pu",
                                    "money": True,
                                    "val": round(take_pu, 2),
                                },
                                {
                                    "row": len(rows) - 1,
                                    "cell": "sal_tot",
                                    "money": True,
                                    "val": round(take_val, 2),
                                },
                                {
                                    "row": len(rows) - 1,
                                    "cell": "sdo_q",
                                    "money": False,
                                    "val": int(saldo_q),
                                },
                                {
                                    "row": len(rows) - 1,
                                    "cell": "sdo_pu",
                                    "money": True,
                                    "val": round(take_pu, 2),
                                },
                                {
                                    "row": len(rows) - 1,
                                    "cell": "sdo_tot",
                                    "money": True,
                                    "val": round(saldo_tot, 2),
                                },
                            ],
                        }
                    )

            # =====================
            # DÍA 5 · DEVOLUCIÓN DE VENTA (REINGRESO)
            # =====================
            if dev_venta_u > 0:
                base_pu_for_return = (
                    sale_unit_cost if sale_unit_cost is not None else (s_pu if s_q > 0 else 0.0)
                )

                # ------- PROMEDIO PONDERADO -------
                if method_name == "Promedio Ponderado":
                    in_q = dev_venta_u
                    in_pu = base_pu_for_return
                    in_val = in_q * in_pu

                    new_q = s_q + in_q
                    new_v = s_v + in_val
                    new_p = (new_v / new_q) if new_q > 0 else 0.0
                    layers = [[new_q, new_p]] if new_q > 0 else []
                    s_q, s_pu, s_v = new_q, new_p, new_v

                    rows.append(
                        {
                            "fecha": "Día 5",
                            "desc": "Devolución de venta (reingreso)",
                            "ent_q": int(in_q),
                            "ent_pu": round(in_pu, 2),
                            "ent_tot": round(in_val, 2),
                            "sal_q": None,
                            "sal_pu": None,
                            "sal_tot": None,
                            "sdo_q": int(s_q),
                            "sdo_pu": round(s_pu, 2),
                            "sdo_tot": round(s_v, 2),
                        }
                    )

                    script.append(
                        {
                            "title": "Día 5 · Devolución de venta (PP)",
                            "text": (
                                f"El cliente devuelve {int(in_q)} unidades. En Promedio Ponderado, "
                                f"reingresan al mismo costo con el que se reconoció la venta "
                                f"({_fmt_money(in_pu)} por unidad), revirtiendo parte del CMV neto "
                                f"y recalculando el costo promedio del inventario."
                            ),
                            "actions": [
                                {
                                    "row": len(rows) - 1,
                                    "cell": "ent_q",
                                    "money": False,
                                    "val": int(in_q),
                                },
                                {
                                    "row": len(rows) - 1,
                                    "cell": "ent_pu",
                                    "money": True,
                                    "val": round(in_pu, 2),
                                },
                                {
                                    "row": len(rows) - 1,
                                    "cell": "ent_tot",
                                    "money": True,
                                    "val": round(in_val, 2),
                                },
                                {
                                    "row": len(rows) - 1,
                                    "cell": "sdo_q",
                                    "money": False,
                                    "val": int(s_q),
                                },
                                {
                                    "row": len(rows) - 1,
                                    "cell": "sdo_pu",
                                    "money": True,
                                    "val": round(s_pu, 2),
                                },
                                {
                                    "row": len(rows) - 1,
                                    "cell": "sdo_tot",
                                    "money": True,
                                    "val": round(s_v, 2),
                                },
                            ],
                        }
                    )

                # ------- PEPS / UEPS (SIN PROMEDIAR) -------
                else:
                    fifo = method_name == "PEPS (FIFO)"
                    metodo_tag = "PEPS" if fifo else "UEPS"

                    in_q = dev_venta_u
                    in_pu = base_pu_for_return
                    in_val = in_q * in_pu

                    # Fusionar con la capa correspondiente o crear nueva
                    eps = 1e-6
                    layer_idx = None

                    if fifo:
                        # PEPS: buscamos desde las capas más antiguas
                        for i, (q, p) in enumerate(layers):
                            if abs(p - in_pu) < eps:
                                layer_idx = i
                                break
                        if layer_idx is not None:
                            layers[layer_idx][0] += in_q
                        else:
                            layers = [[float(in_q), float(in_pu)]] + layers
                            layer_idx = 0
                    else:
                        # UEPS: buscamos desde las capas más recientes
                        for i in range(len(layers) - 1, -1, -1):
                            q, p = layers[i]
                            if abs(p - in_pu) < eps:
                                layer_idx = i
                                break
                        if layer_idx is not None:
                            layers[layer_idx][0] += in_q
                        else:
                            layers = layers + [[float(in_q), float(in_pu)]]
                            layer_idx = len(layers) - 1

                    capa_q, capa_pu = layers[layer_idx]
                    capa_tot = capa_q * capa_pu

                    s_q_total, s_pu_total, s_v_total = _sum_layers(layers)
                    s_q, s_pu, s_v = s_q_total, s_pu_total, s_v_total

                    rows.append(
                        {
                            "fecha": "Día 5",
                            "desc": "Devolución de venta (reingreso)",
                            "ent_q": int(in_q),
                            "ent_pu": round(in_pu, 2),
                            "ent_tot": round(in_val, 2),
                            "sal_q": None,
                            "sal_pu": None,
                            "sal_tot": None,
                            # Saldo mostrado: SOLO la capa asociada a la devolución
                            "sdo_q": int(capa_q),
                            "sdo_pu": round(capa_pu, 2),
                            "sdo_tot": round(capa_tot, 2),
                        }
                    )

                    script.append(
                        {
                            "title": f"Día 5 · Devolución de venta ({metodo_tag})",
                            "text": (
                                f"El cliente devuelve {int(in_q)} unidades. En {metodo_tag}, las reingresamos "
                                f"al mismo costo con el que salieron en la venta original, "
                                f"{_fmt_money(in_pu)} por unidad.\n\n"
                                f"Si todavía existía inventario en esa misma capa, simplemente le sumamos las unidades devueltas; "
                                f"el costo por unidad NO cambia. En la fila del Día 5 el SALDO muestra la capa asociada a esa devolución."
                            ),
                            "actions": [
                                {
                                    "row": len(rows) - 1,
                                    "cell": "ent_q",
                                    "money": False,
                                    "val": int(in_q),
                                },
                                {
                                    "row": len(rows) - 1,
                                    "cell": "ent_pu",
                                    "money": True,
                                    "val": round(in_pu, 2),
                                },
                                {
                                    "row": len(rows) - 1,
                                    "cell": "ent_tot",
                                    "money": True,
                                    "val": round(in_val, 2),
                                },
                                {
                                    "row": len(rows) - 1,
                                    "cell": "sdo_q",
                                    "money": False,
                                    "val": int(capa_q),
                                },
                                {
                                    "row": len(rows) - 1,
                                    "cell": "sdo_pu",
                                    "money": True,
                                    "val": round(capa_pu, 2),
                                },
                                {
                                    "row": len(rows) - 1,
                                    "cell": "sdo_tot",
                                    "money": True,
                                    "val": round(capa_tot, 2),
                                },
                            ],
                        }
                    )

            return rows, script, narr_start_idx

        # Llamada
        demo_rows, demo_script, narr_start_idx = compute_rows_and_script_with_returns(
            metodo,
            inv0_u,
            inv0_pu,
            comp_u,
            comp_pu,
            venta_u,
            dev_comp_u,
            dev_venta_u,
        )

        # ========= HTML + JS de la demo narrada =========
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
        <button id="pauseDemo" class="btn">⏸️ Pausar</button>
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
        const rows     = %%ROWS%%;
        const script   = %%SCRIPT%%;
        const metodo   = "%%METODO%%";
        const narrStart = %%NARRSTART%%;
        const narrMuted = %%MUTED%%;
        const rate     = %%RATE%%;

        const tbody    = document.getElementById("kbody");
        const narrDiv  = document.getElementById("narr");
        const btnPlay  = document.getElementById("playDemo");
        const btnPause = document.getElementById("pauseDemo");
        const btnReset = document.getElementById("resetDemo");

        let isPaused   = false;
        let isRunning  = false;

        const pesos = (v)=> {
            try { return new Intl.NumberFormat('es-CO',{style:'currency', currency:'COP', maximumFractionDigits:2}).format(v); }
            catch(e){ return "$"+(Math.round(v*100)/100).toLocaleString('es-CO'); }
        };

        const fmt = (x)=> (x===null || x===undefined || x==="")
            ? ""
            : (typeof x==="number"
                ? (Number.isInteger(x)? x.toString() : (Math.round(x*100)/100).toString().replace(".",","))
                : x);

        function buildTable(){
            tbody.innerHTML = "";
            rows.forEach((r, i)=>{
                const tr = document.createElement("tr");
                tr.id = "row"+i;

                const isNarr = (i >= narrStart);

                const ent_q   = isNarr ? "" : fmt(r.ent_q);
                const ent_pu  = isNarr ? "" : (r.ent_pu!==null && r.ent_pu!==undefined ? pesos(r.ent_pu): "");
                const ent_tot = isNarr ? "" : (r.ent_tot!==null && r.ent_tot!==undefined ? pesos(r.ent_tot): "");

                const sal_q   = isNarr ? "" : fmt(r.sal_q);
                const sal_pu  = isNarr ? "" : (r.sal_pu!==null && r.sal_pu!==undefined ? pesos(r.sal_pu): "");
                const sal_tot = isNarr ? "" : (r.sal_tot!==null && r.sal_tot!==undefined ? pesos(r.sal_tot): "");

                const sdo_q   = isNarr ? "" : fmt(r.sdo_q);
                const sdo_pu  = isNarr ? "" : (r.sdo_pu!==null && r.sdo_pu!==undefined ? pesos(r.sdo_pu): "");
                const sdo_tot = isNarr ? "" : (r.sdo_tot!==null && r.sdo_tot!==undefined ? pesos(r.sdo_tot): "");

                tr.innerHTML = `
                    <td>${r.fecha}</td><td>${r.desc}</td>
                    <td id="r${i}_ent_q"  class="fill">${ent_q}</td>
                    <td id="r${i}_ent_pu" class="fill">${ent_pu}</td>
                    <td id="r${i}_ent_tot"class="fill">${ent_tot}</td>
                    <td id="r${i}_sal_q"  class="fill">${sal_q}</td>
                    <td id="r${i}_sal_pu" class="fill">${sal_pu}</td>
                    <td id="r${i}_sal_tot"class="fill">${sal_tot}</td>
                    <td id="r${i}_sdo_q"  class="fill">${sdo_q}</td>
                    <td id="r${i}_sdo_pu" class="fill">${sdo_pu}</td>
                    <td id="r${i}_sdo_tot"class="fill">${sdo_tot}</td>
                `;
                tbody.appendChild(tr);
            });
        }

        function clearHi(){
            [...tbody.querySelectorAll("tr")].forEach(tr=> tr.classList.remove("hi"));
        }
        function highlightRow(i){
            clearHi();
            const tr = document.getElementById("row"+i);
            if (tr) tr.classList.add("hi");
        }
        function fillCell(rowIdx, key, val, money=false){
            const el = document.getElementById(`r${rowIdx}_${key}`);
            if (!el) return;
            el.textContent = money ? pesos(val) : fmt(val);
            el.style.background = "#fffbe6";
            setTimeout(()=>{ el.style.background=""; }, 300);
        }

        const sleep = (ms)=> new Promise(r => setTimeout(r, ms));

        async function waitIfPaused(){
            while(isPaused){
                await sleep(150);
            }
        }

        function speak(text){
            return new Promise((resolve)=>{
                if (narrMuted) return resolve();
                try{
                    if (window.speechSynthesis.speaking) window.speechSynthesis.cancel();
                    const u = new SpeechSynthesisUtterance(text);
                    const voices = window.speechSynthesis.getVoices();
                    const pick = voices.find(v => /es|spanish|mex|col/i.test((v.name+" "+v.lang))) || voices[0];
                    if (pick) u.voice = pick;
                    u.rate = rate;
                    u.pitch = 1.0;
                    u.onend = ()=> resolve();
                    window.speechSynthesis.speak(u);
                } catch(e){
                    resolve();
                }
            });
        }

        async function runScript(){
            if (isRunning) return;

            isRunning = true;
            isPaused = false;
            btnPause.textContent = "⏸️ Pausar";

            buildTable();
            clearHi();
            narrDiv.textContent = "";

            for (let sIdx = 0; sIdx < script.length; sIdx++){
                const step = script[sIdx];
                await waitIfPaused();

                narrDiv.textContent = step.title;

                if (step.actions && step.actions.length>0){
                    highlightRow(step.actions[0].row);
                }

                const dur = Math.max(2200, Math.min(7000, step.text.length * 55 / rate));
                const chunks = Math.max(3, step.actions.length);
                const waits = Array.from({length:chunks-1}, (_,k)=> Math.floor(dur*(k+1)/chunks));

                const pVoice = speak(step.text);

                for (let i=0;i<step.actions.length;i++){
                    await waitIfPaused();
                    const a = step.actions[i];
                    if (i>0){ await sleep(waits[i-1]); }
                    fillCell(a.row, a.cell, a.val, !!a.money);
                }

                await pVoice;
                await waitIfPaused();
                await sleep(200);
            }

            clearHi();
            isRunning = false;
        }

        btnPlay.onclick = runScript;

        btnPause.onclick = ()=>{
            if (!isRunning) return;
            if (!isPaused){
                isPaused = true;
                if (window.speechSynthesis && window.speechSynthesis.speaking){
                    window.speechSynthesis.pause();
                }
                btnPause.textContent = "▶️ Reanudar";
            } else {
                isPaused = false;
                if (window.speechSynthesis && window.speechSynthesis.paused){
                    window.speechSynthesis.resume();
                }
                btnPause.textContent = "⏸️ Pausar";
            }
        };

        btnReset.onclick = ()=>{
            try{ if (window.speechSynthesis) window.speechSynthesis.cancel(); }catch(e){}
            isPaused = false;
            isRunning = false;
            btnPause.textContent = "⏸️ Pausar";
            buildTable();
            clearHi();
            narrDiv.textContent = "";
        };

        buildTable();
        })();
        </script>
        """

        html_demo = (
            html_demo_template.replace("%%ROWS%%", _json.dumps(demo_rows))
            .replace("%%SCRIPT%%", _json.dumps(demo_script))
            .replace("%%METODO%%", metodo)
            .replace("%%NARRSTART%%", str(narr_start_idx))
            .replace("%%MUTED%%", "true" if narr_muted else "false")
            .replace("%%RATE%%", str(narr_speed))
        )

        components.html(html_demo, height=360, scrolling=True)

    # =========================================
    # TAB 2 · PRÁCTICA IA (KARDEX DÍAS 1–5)
    # =========================================
    with tabs[2]:
        st.subheader("Práctica IA: diligencia tu propio KARDEX (Nivel 3)")
        st.caption("Completa TODAS las filas. Deja en BLANCO las celdas que no aplican en cada fila. Secuencia: Día 1 a Día 5.")

        # =========================
        # Estado y escenario
        # =========================
        def _ensure_default_state():
            ss = st.session_state
            ss.setdefault("n2_ex_metodo", "Promedio Ponderado")
            # Día 1
            ss.setdefault("n2_ex_inv0_u", 80)
            ss.setdefault("n2_ex_inv0_pu", 10.0)
            # Día 2
            ss.setdefault("n2_ex_comp1_u", 40)
            ss.setdefault("n2_ex_comp1_pu", 11.0)
            # Día 3
            ss.setdefault("n2_ex_venta_u", 90)
            # Día 4 (devolución de compra a proveedor)
            ss.setdefault("n2_ex_dev_comp_u", 8)
            # Día 5 (devolución de venta del cliente)
            ss.setdefault("n2_ex_dev_venta_u", 6)

        def _randomize_scenario_values():
            import random
            inv0_u = random.choice([60, 80, 100, 120, 150])
            inv0_pu = random.choice([8.0, 9.0, 10.0, 11.0, 12.0])
            comp1_u = random.choice([30, 40, 50, 60, 70])
            comp1_pu = random.choice([inv0_pu - 1, inv0_pu, inv0_pu + 1, inv0_pu + 2])
            venta_u = random.choice([40, 60, 90, 110, 130])
            dev_comp_u = max(0, min(comp1_u, random.choice([5, 8, 10, 12, 15])))
            dev_venta_u = max(0, min(venta_u, random.choice([4, 6, 8, 10, 12])))

            ss = st.session_state
            ss["n2_ex_inv0_u"] = inv0_u
            ss["n2_ex_inv0_pu"] = float(max(1.0, round(inv0_pu, 2)))
            ss["n2_ex_comp1_u"] = comp1_u
            ss["n2_ex_comp1_pu"] = float(max(1.0, round(comp1_pu, 2)))
            ss["n2_ex_venta_u"] = venta_u
            ss["n2_ex_dev_comp_u"] = dev_comp_u
            ss["n2_ex_dev_venta_u"] = dev_venta_u

        def _request_randomize():
            st.session_state["n2_ex_rand_request"] = True

        _ensure_default_state()

        if st.session_state.get("n2_ex_rand_request", False):
            _randomize_scenario_values()
            st.session_state.pop("n2_ex_rand_request", None)
            st.rerun()

        # =========================
        # Método
        # =========================
        c0a, c0b = st.columns([1, 3])
        with c0a:
            ex_metodo = st.selectbox(
                "Método (ejercicio)",
                ["Promedio Ponderado", "PEPS (FIFO)", "UEPS (LIFO)"],
                key="n2_ex_metodo"
            )

        # =========================
        # Escenario (textos solicitados)
        # =========================
        st.markdown("#### 🎯 Escenario del ejercicio")

        st.markdown("**Día 1. La empresa reporta un saldo inicial de inventario de:**")
        c1a, c1b = st.columns([1, 1])
        with c1a:
            inv0_u_ex = st.number_input("Cantidad (u) — Día 1", min_value=0, step=1, key="n2_ex_inv0_u")
        with c1b:
            inv0_pu_ex = st.number_input("Costo unitario — Día 1", min_value=0.0, step=0.1, key="n2_ex_inv0_pu")

        st.markdown("**Día 2. La empresa realizó una compra de:**")
        c2a, c2b = st.columns([1, 1])
        with c2a:
            comp1_u = st.number_input("Cantidad (u) — Día 2", min_value=0, step=1, key="n2_ex_comp1_u")
        with c2b:
            comp1_pu = st.number_input("Costo unitario — Día 2", min_value=0.0, step=0.1, key="n2_ex_comp1_pu")

        st.markdown("**Día 3. La empresa realizó una venta de:**")
        venta_ex_u = st.number_input("Cantidad vendida (u) — Día 3", min_value=0, step=1, key="n2_ex_venta_u")

        st.markdown("**Día 4. La empresa realizó una devolución de compra (al proveedor) por:**")
        dev_comp_u = st.number_input("Unidades devueltas al proveedor (u) — Día 4", min_value=0, step=1, key="n2_ex_dev_comp_u")

        st.markdown("**Día 5. La empresa recibió una devolución de venta (del cliente) por:**")
        dev_venta_u = st.number_input("Unidades devueltas por el cliente (u) — Día 5", min_value=0, step=1, key="n2_ex_dev_venta_u")

        st.button("🎲 Generar escenario aleatorio", key="n2_ex_rand_btn", on_click=_request_randomize)

        # =========================
        # Helpers para práctica
        # =========================
        def _sum_layers(layers):
            q = sum(q for q, _ in layers)
            v = sum(q * p for q, p in layers)
            pu = (v / q) if q > 0 else 0.0
            return q, pu, v

        def _consume_layers_detail(layers, qty_out, fifo=True):
            """
            Devuelve:
            - sale_details: [(q_take, pu, total), ...]
            - final_layers: capas remanentes en orden FIFO natural
            """
            order = layers[:] if fifo else layers[::-1]
            remaining = qty_out
            sale_details = []
            updated = []
            for q, pu in order:
                if remaining <= 0:
                    updated.append([q, pu])
                    continue
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
        # Filas ESPERADAS (D1–D5)
        # =========================
        def build_expected_rows(method_name):
            """
            Devuelve filas con columnas:
            Fecha, Descripción,
            Entrada_cant, Entrada_pu, Entrada_total,
            Salida_cant,  Salida_pu,  Salida_total,
            Saldo_cant,   Saldo_pu,   Saldo_total

            Secuencia:
            D1: Saldo inicial (como ENTRADA + SALDO).
            D2: Compra 1
                - PP: saldo con promedio.
                - PEPS/UEPS: saldo muestra SOLO la capa comprada (sin promediar).
            D3: Venta
                - PP: una fila.
                - PEPS/UEPS: tramos por capa; si se agota una capa, el saldo de esa fila muestra 0 a ese costo.
            D4: Devolución de compra
                - PP: al costo de la compra original (comp1_pu).
                - PEPS/UEPS: descuenta capas según el método.
            D5: Devolución de venta
                - PP: reingreso al costo de la venta original (sale_unit_cost).
                - PEPS/UEPS: reingreso al mismo costo de salida, fusionando capa y mostrando SOLO esa capa en el saldo.
            """
            rows = []
            sale_unit_cost = None  # para devolver las ventas al mismo costo

            # --------------------------
            # Día 1 · Saldo inicial (ENTRADA + SALDO)
            # --------------------------
            if inv0_u_ex > 0:
                ent_q1 = int(inv0_u_ex)
                ent_pu1 = float(inv0_pu_ex)
                ent_tot1 = ent_q1 * ent_pu1
                layers = [[float(ent_q1), ent_pu1]]
            else:
                ent_q1 = ent_pu1 = ent_tot1 = None
                layers = []

            s_q, s_p, s_v = _sum_layers(layers)

            rows.append({
                "Fecha": "Día 1", "Descripción": "Saldo inicial",
                "Entrada_cant": ent_q1,
                "Entrada_pu": round(ent_pu1, 2) if ent_pu1 is not None else None,
                "Entrada_total": round(ent_tot1, 2) if ent_tot1 is not None else None,
                "Salida_cant": None, "Salida_pu": None, "Salida_total": None,
                "Saldo_cant": s_q,
                "Saldo_pu": round(s_p, 2),
                "Saldo_total": round(s_v, 2)
            })

            # --------------------------
            # Día 2 · Compra 1 (SIN "Saldo día 1")
            # --------------------------
            ent_tot2 = comp1_u * comp1_pu
            if method_name == "Promedio Ponderado":
                # PP: se promedia con el saldo anterior
                q_new = s_q + comp1_u
                v_new = s_v + ent_tot2
                p_new = (v_new / q_new) if q_new > 0 else 0.0
                layers = [[q_new, p_new]]
                s_q, s_p, s_v = _sum_layers(layers)

                rows.append({
                    "Fecha": "Día 2", "Descripción": "Compra 1",
                    "Entrada_cant": comp1_u,
                    "Entrada_pu": round(comp1_pu, 2),
                    "Entrada_total": round(ent_tot2, 2),
                    "Salida_cant": None, "Salida_pu": None, "Salida_total": None,
                    "Saldo_cant": s_q,
                    "Saldo_pu": round(s_p, 2),
                    "Saldo_total": round(s_v, 2)
                })
            else:
                # PEPS / UEPS: la fila muestra SOLO la capa de la compra
                if comp1_u > 0:
                    layers.append([float(comp1_u), float(comp1_pu)])

                rows.append({
                    "Fecha": "Día 2", "Descripción": "Compra 1",
                    "Entrada_cant": comp1_u,
                    "Entrada_pu": round(comp1_pu, 2),
                    "Entrada_total": round(ent_tot2, 2),
                    "Salida_cant": None, "Salida_pu": None, "Salida_total": None,
                    "Saldo_cant": comp1_u,
                    "Saldo_pu": round(comp1_pu, 2),
                    "Saldo_total": round(ent_tot2, 2)
                })
                # Estado real para siguientes días (todas las capas)
                s_q, s_p, s_v = _sum_layers(layers)

            # --------------------------
            # Día 3 · Venta
            # --------------------------
            if method_name == "Promedio Ponderado":
                if s_q > 0 and venta_ex_u > 0:
                    sale_q = min(venta_ex_u, s_q)
                    sale_pu = layers[0][1] if layers else 0.0
                    sale_unit_cost = sale_pu
                    sale_tot = sale_q * sale_pu

                    q2 = s_q - sale_q
                    v2 = s_v - sale_tot
                    p2 = (v2 / q2) if q2 > 0 else 0.0
                    layers = [[q2, p2]] if q2 > 0 else []
                    s_q, s_p, s_v = _sum_layers(layers)

                    rows.append({
                        "Fecha": "Día 3", "Descripción": "Venta",
                        "Entrada_cant": None, "Entrada_pu": None, "Entrada_total": None,
                        "Salida_cant": sale_q,
                        "Salida_pu": round(sale_pu, 2),
                        "Salida_total": round(sale_tot, 2),
                        "Saldo_cant": s_q,
                        "Saldo_pu": round(s_p, 2),
                        "Saldo_total": round(s_v, 2)
                    })
                else:
                    rows.append({
                        "Fecha": "Día 3", "Descripción": "Venta",
                        "Entrada_cant": None, "Entrada_pu": None, "Entrada_total": None,
                        "Salida_cant": None, "Salida_pu": None, "Salida_total": None,
                        "Saldo_cant": s_q,
                        "Saldo_pu": round(s_p, 2),
                        "Saldo_total": round(s_v, 2)
                    })
            else:
                # PEPS / UEPS: venta por tramos, sin promediar
                fifo = (method_name == "PEPS (FIFO)")
                metodo_tag = "PEPS" if fifo else "UEPS"

                total_inv = sum(q for q, _ in layers)
                if venta_ex_u > 0 and total_inv > 0:
                    layers_for_state = [[float(q), float(p)] for (q, p) in layers]
                    sale_remaining = float(venta_ex_u)
                    tramo_index = 1
                    sale_details = []

                    while sale_remaining > 0 and any(q > 0 for q, _ in layers_for_state):
                        # Capa según método
                        if fifo:
                            idx_layer = next(i for i, (q, _) in enumerate(layers_for_state) if q > 0)
                        else:
                            idx_layer = max(i for i, (q, _) in enumerate(layers_for_state) if q > 0)

                        layer_q, layer_pu = layers_for_state[idx_layer]
                        q_take = min(layer_q, sale_remaining)
                        tot_take = q_take * layer_pu
                        sale_remaining -= q_take

                        # Actualizar capa
                        q_rem = layer_q - q_take
                        layers_for_state[idx_layer][0] = q_rem

                        # Guardar detalle para costo de devolución de venta
                        sale_details.append((q_take, layer_pu, tot_take))

                        # SALDO que se muestra: SOLO la capa de este tramo
                        if q_rem > 0:
                            sdo_q = q_rem
                            sdo_pu = layer_pu
                        else:
                            sdo_q = 0.0
                            sdo_pu = layer_pu
                        sdo_tot = sdo_q * sdo_pu

                        rows.append({
                            "Fecha": "Día 3",
                            "Descripción": f"Venta tramo {tramo_index} ({metodo_tag})",
                            "Entrada_cant": None, "Entrada_pu": None, "Entrada_total": None,
                            "Salida_cant": int(q_take),
                            "Salida_pu": round(layer_pu, 2),
                            "Salida_total": round(tot_take, 2),
                            "Saldo_cant": int(sdo_q),
                            "Saldo_pu": round(sdo_pu, 2),
                            "Saldo_total": round(sdo_tot, 2)
                        })

                        tramo_index += 1

                    # Costo unitario para la devolución de venta:
                    if sale_details:
                        if fifo:
                            # PEPS: costo de la primera capa vendida
                            sale_unit_cost = sale_details[0][1]
                        else:
                            # UEPS: costo de la última capa vendida
                            sale_unit_cost = sale_details[-1][1]

                    # Capas remanentes reales para días 4–5
                    layers = [(q, p) for (q, p) in layers_for_state if q > 0]
                    s_q, s_p, s_v = _sum_layers(layers)
                else:
                    rows.append({
                        "Fecha": "Día 3",
                        "Descripción": "Venta",
                        "Entrada_cant": None, "Entrada_pu": None, "Entrada_total": None,
                        "Salida_cant": None, "Salida_pu": None, "Salida_total": None,
                        "Saldo_cant": s_q,
                        "Saldo_pu": round(s_p, 2),
                        "Saldo_total": round(s_v, 2)
                    })

            # --------------------------
            # Día 4 · Devolución de compra
            # --------------------------
            if method_name == "Promedio Ponderado":
                if dev_comp_u > 0 and s_q > 0:
                    # Devolución al COSTO ORIGINAL de la compra 1 (no al promedio vigente)
                    take_q = min(dev_comp_u, s_q)
                    take_pu = float(comp1_pu)
                    take_val = take_q * take_pu

                    q4 = max(s_q - take_q, 0.0)
                    v4 = max(s_v - take_val, 0.0)
                    p4 = (v4 / q4) if q4 > 0 else 0.0
                    layers = [[q4, p4]] if q4 > 0 else []
                    s_q, s_p, s_v = _sum_layers(layers)

                    rows.append({
                        "Fecha": "Día 4", "Descripción": "Devolución de compra",
                        "Entrada_cant": None, "Entrada_pu": None, "Entrada_total": None,
                        "Salida_cant": take_q,
                        "Salida_pu": round(take_pu, 2),
                        "Salida_total": round(take_val, 2),
                        "Saldo_cant": s_q,
                        "Saldo_pu": round(s_p, 2),
                        "Saldo_total": round(s_v, 2)
                    })
                else:
                    rows.append({
                        "Fecha": "Día 4", "Descripción": "Devolución de compra",
                        "Entrada_cant": None, "Entrada_pu": None, "Entrada_total": None,
                        "Salida_cant": None, "Salida_pu": None, "Salida_total": None,
                        "Saldo_cant": s_q,
                        "Saldo_pu": round(s_p, 2),
                        "Saldo_total": round(s_v, 2)
                    })
            else:
                # PEPS / UEPS: devolvemos siguiendo capas del método
                if dev_comp_u > 0 and s_q > 0:
                    fifo = (method_name == "PEPS (FIFO)")
                    sale_details, layers_after = _consume_layers_detail(layers, dev_comp_u, fifo=fifo)
                    take_q = sum(q for q, _, _ in sale_details)
                    take_val = sum(t for _, _, t in sale_details)
                    take_pu = (take_val / take_q) if take_q > 0 else 0.0
                    layers = layers_after
                    s_q, s_p, s_v = _sum_layers(layers)

                    rows.append({
                        "Fecha": "Día 4", "Descripción": "Devolución de compra",
                        "Entrada_cant": None, "Entrada_pu": None, "Entrada_total": None,
                        "Salida_cant": take_q,
                        "Salida_pu": round(take_pu, 2),
                        "Salida_total": round(take_val, 2),
                        "Saldo_cant": s_q,
                        "Saldo_pu": round(s_p, 2),
                        "Saldo_total": round(s_v, 2)
                    })
                else:
                    rows.append({
                        "Fecha": "Día 4", "Descripción": "Devolución de compra",
                        "Entrada_cant": None, "Entrada_pu": None, "Entrada_total": None,
                        "Salida_cant": None, "Salida_pu": None, "Salida_total": None,
                        "Saldo_cant": s_q,
                        "Saldo_pu": round(s_p, 2),
                        "Saldo_total": round(s_v, 2)
                    })

            # --------------------------
            # Día 5 · Devolución de venta (reingreso)
            # --------------------------
            if method_name == "Promedio Ponderado":
                if dev_venta_u > 0:
                    in_q = dev_venta_u
                    in_pu = sale_unit_cost if sale_unit_cost is not None else s_p
                    in_val = in_q * in_pu

                    q5 = s_q + in_q
                    v5 = s_v + in_val
                    p5 = (v5 / q5) if q5 > 0 else 0.0
                    layers = [[q5, p5]] if q5 > 0 else []
                    s_q, s_p, s_v = _sum_layers(layers)

                    rows.append({
                        "Fecha": "Día 5", "Descripción": "Devolución de venta (reingreso)",
                        "Entrada_cant": in_q,
                        "Entrada_pu": round(in_pu, 2),
                        "Entrada_total": round(in_val, 2),
                        "Salida_cant": None, "Salida_pu": None, "Salida_total": None,
                        "Saldo_cant": s_q,
                        "Saldo_pu": round(s_p, 2),
                        "Saldo_total": round(s_v, 2)
                    })
                else:
                    rows.append({
                        "Fecha": "Día 5", "Descripción": "Devolución de venta (reingreso)",
                        "Entrada_cant": None, "Entrada_pu": None, "Entrada_total": None,
                        "Salida_cant": None, "Salida_pu": None, "Salida_total": None,
                        "Saldo_cant": s_q,
                        "Saldo_pu": round(s_p, 2),
                        "Saldo_total": round(s_v, 2)
                    })
            else:
                # PEPS / UEPS: reingresamos al costo de la venta original, a la capa correspondiente
                if dev_venta_u > 0:
                    fifo = (method_name == "PEPS (FIFO)")
                    base_pu = sale_unit_cost if sale_unit_cost is not None else s_p
                    in_q = dev_venta_u
                    in_pu = base_pu
                    in_val = in_q * in_pu

                    # Fusionar con capa del mismo costo (sin promediar)
                    eps = 1e-6
                    layer_idx = None

                    if fifo:
                        # PEPS: buscamos desde el inicio
                        for i, (q, p) in enumerate(layers):
                            if abs(p - in_pu) < eps:
                                layer_idx = i
                                break
                        if layer_idx is not None:
                            layers[layer_idx][0] += in_q
                        else:
                            layers = [[float(in_q), float(in_pu)]] + layers
                            layer_idx = 0
                    else:
                        # UEPS: buscamos desde el final
                        for i in range(len(layers) - 1, -1, -1):
                            q, p = layers[i]
                            if abs(p - in_pu) < eps:
                                layer_idx = i
                                break
                        if layer_idx is not None:
                            layers[layer_idx][0] += in_q
                        else:
                            layers = layers + [[float(in_q), float(in_pu)]]
                            layer_idx = len(layers) - 1

                    # Capa asociada a la devolución (saldo que se muestra SOLO con esa capa)
                    capa_q, capa_pu = layers[layer_idx]
                    capa_tot = capa_q * capa_pu

                    # Estado global (por si se usara después)
                    s_q, s_p, s_v = _sum_layers(layers)

                    rows.append({
                        "Fecha": "Día 5", "Descripción": "Devolución de venta (reingreso)",
                        "Entrada_cant": in_q,
                        "Entrada_pu": round(in_pu, 2),
                        "Entrada_total": round(in_val, 2),
                        "Salida_cant": None, "Salida_pu": None, "Salida_total": None,
                        "Saldo_cant": int(capa_q),
                        "Saldo_pu": round(capa_pu, 2),
                        "Saldo_total": round(capa_tot, 2)
                    })
                else:
                    rows.append({
                        "Fecha": "Día 5", "Descripción": "Devolución de venta (reingreso)",
                        "Entrada_cant": None, "Entrada_pu": None, "Entrada_total": None,
                        "Salida_cant": None, "Salida_pu": None, "Salida_total": None,
                        "Saldo_cant": s_q,
                        "Saldo_pu": round(s_p, 2),
                        "Saldo_total": round(s_v, 2)
                    })

            return rows

        expected_rows = build_expected_rows(ex_metodo)

        # =========================
        # Editor: tabla COMPLETAMENTE EN BLANCO
        # =========================
        def _blank_like(r):
            return {
                "Fecha": r["Fecha"], "Descripción": r["Descripción"],
                "Entrada_cant": None, "Entrada_pu": None, "Entrada_total": None,
                "Salida_cant": None, "Salida_pu": None, "Salida_total": None,
                "Saldo_cant": None, "Saldo_pu": None, "Saldo_total": None
            }

        plant_rows = [_blank_like(r) for r in expected_rows]
        df_for_editor = pd.DataFrame(plant_rows)

        all_cols = ["Fecha", "Descripción",
                    "Entrada_cant", "Entrada_pu", "Entrada_total",
                    "Salida_cant", "Salida_pu", "Salida_total",
                    "Saldo_cant", "Saldo_pu", "Saldo_total"]
        for c in all_cols:
            if c not in df_for_editor.columns:
                df_for_editor[c] = ""
        df_for_editor = df_for_editor[all_cols]

        col_config = {
            "Fecha": st.column_config.TextColumn(disabled=True),
            "Descripción": st.column_config.TextColumn(disabled=True),
            "Entrada_cant": st.column_config.NumberColumn(step=1, help="Unidades que ingresan"),
            "Entrada_pu": st.column_config.NumberColumn(step=0.01, help="Costo unitario de la entrada"),
            "Entrada_total": st.column_config.NumberColumn(step=0.01, help="Valor total de la entrada"),
            "Salida_cant": st.column_config.NumberColumn(step=1, help="Unidades que salen"),
            "Salida_pu": st.column_config.NumberColumn(step=0.01, help="Costo unitario de la salida"),
            "Salida_total": st.column_config.NumberColumn(step=0.01, help="Valor total de la salida"),
            "Saldo_cant": st.column_config.NumberColumn(step=1, help="Unidades en saldo"),
            "Saldo_pu": st.column_config.NumberColumn(step=0.01, help="Costo unitario del saldo"),
            "Saldo_total": st.column_config.NumberColumn(step=0.01, help="Valor total del saldo"),
        }

        st.markdown("#### ✍️ Completa TODAS las filas (Días 1–5)")
        st.caption("Deja en BLANCO las celdas que no aplican (por ejemplo, Entradas en una fila de Venta).")
        edited = st.data_editor(
            df_for_editor,
            use_container_width=True,
            num_rows="fixed",
            disabled=False,
            column_config=col_config,
            hide_index=True,
            key="n2_kardex_student_table_lvl2"
        )

        # =========================
        # Validación
        # =========================
        with st.form("n2_kardex_check_lvl2"):
            ask_ai = st.checkbox("💬 Retroalimentación de IA (opcional)", value=False, key="n2_kardex_ai_lvl2")
            submitted_ex = st.form_submit_button("✅ Validar mi KARDEX")

        if submitted_ex:
            tol = 0.5

            def _to_float(x):
                try:
                    if x in (None, ""):
                        return None
                    return float(x)
                except Exception:
                    return None

            def _near(a, b):
                if a is None or b is None:
                    return False
                return abs(a - b) <= tol

            def _is_empty_exp(v):
                if v is None or v == "":
                    return True
                try:
                    import math
                    return isinstance(v, float) and math.isnan(v)
                except:
                    return False

            flags = []
            for i in range(len(expected_rows)):
                user = edited.iloc[i].to_dict()
                exp = expected_rows[i]
                ok_cells = []

                for key in [
                    "Entrada_cant", "Entrada_pu", "Entrada_total",
                    "Salida_cant", "Salida_pu", "Salida_total",
                    "Saldo_cant", "Saldo_pu", "Saldo_total"
                ]:
                    exp_val = exp[key]
                    usr_val = _to_float(user.get(key, ""))

                    if _is_empty_exp(exp_val):
                        ok = True
                    else:
                        ok = _near(usr_val, float(exp_val))

                    ok_cells.append(ok)

                flags.append((f"{exp['Fecha']} · {exp['Descripción']}", all(ok_cells)))

            aciertos = sum(1 for _, ok in flags if ok)
            st.metric("Aciertos por fila", f"{aciertos}/{len(flags)}")
            for label, ok in flags:
                st.write(("✅ " if ok else "❌ ") + label)

            if aciertos == len(flags):
                st.success("¡Excelente! Tu KARDEX coincide con el método y el tratamiento correcto de las devoluciones.")
            else:
                st.warning("Hay diferencias. Revisa cantidades, costos unitarios, devoluciones y el método aplicado en cada día.")

            if ask_ai:
                def _row_summary(idx):
                    r = edited.iloc[idx].to_dict()
                    def g(k):
                        v = _to_float(r.get(k, ""))
                        return "—" if v is None else f"{v:.2f}"
                    return (f"{edited.iloc[idx]['Fecha']} {edited.iloc[idx]['Descripción']}: "
                            f"E({g('Entrada_cant')},{g('Entrada_pu')},{g('Entrada_total')}) | "
                            f"S({g('Salida_cant')},{g('Salida_pu')},{g('Salida_total')}) | "
                            f"Saldo({g('Saldo_cant')},{g('Saldo_pu')},{g('Saldo_total')})")

                intento = "\n".join(_row_summary(i) for i in range(len(expected_rows)))

                final_exp = expected_rows[-1]
                exp_qtyF = final_exp["Saldo_cant"] if final_exp["Saldo_cant"] != "" else None
                exp_valF = final_exp["Saldo_total"] if final_exp["Saldo_total"] != "" else None
                exp_puF = final_exp["Saldo_pu"] if final_exp["Saldo_pu"] != "" else None

                sol_desc = (
                    f"Método: {ex_metodo}. "
                    f"Saldo final esperado: cant={exp_qtyF}, val={exp_valF}, pu={exp_puF}. "
                    f"Filas esperadas: {len(expected_rows)}."
                )

                with st.spinner("Generando retroalimentación de IA…"):
                    fb_txt = ia_feedback(
                        "Evalúa el KARDEX diligenciado por el estudiante (Días 1–5). " + sol_desc +
                        "\nEntradas del estudiante:\n" + intento +
                        "\nIndica: (1) si respeta el método (PP/PEPS/UEPS) en compra, venta y devoluciones, "
                        "(2) errores típicos (costo de devolución, promedio mal aplicado), "
                        "(3) un tip memotécnico breve sobre devoluciones al costo de salida original."
                    )
                with st.expander("💬 Retroalimentación de la IA"):
                    st.write(fb_txt)

    with tabs[3]:
        st.subheader("Evaluación final del Nivel 3")
        st.caption("Debes acertar **5 de 5** para aprobar y avanzar.")

        # =========================
        # Utilidades y helpers
        # =========================
        import pandas as _pd
        import time, re

        def _on_topic_fallback_q4() -> str:
            """Feedback on-topic si el proveedor no responde o se desvía."""
            return (
                "Para que las devoluciones no distorsionen los estados, deben registrarse con el "
                "costo correcto según el método:\n"
                "- En **Promedio Ponderado (PP)**: usa el **promedio vigente** al momento de la operación; "
                "ajusta el **CMV** (si es devolución de venta) o las **compras netas** (si es devolución de compra) y "
                "recalcula el **saldo y su costo unitario**.\n"
                "- En **PEPS/UEPS**: respeta la **capa** que corresponde (más antigua o más reciente). "
                "Promediar en estos métodos genera errores en CMV y en el valor del inventario.\n"
                "Consecuencia típica de hacerlo mal: **CMV**, **saldo** y **comparabilidad** quedan afectados, "
                "lo que sesga la utilidad y las decisiones de gestión."
            )

        def _sanitize_on_topic_q4(text: str) -> str:
            """Filtra contenido fuera de tema (debe/haber, IVA, asientos, etc.) y devuelve fallback si aparece."""
            if not text:
                return _on_topic_fallback_q4()
            banned = [
                "debe", "haber", "asiento", "apertura", "cierre", "diario",
                "iva repercutido", "iva soportado", "iva", "cuentas t",
                "resultado del ejercicio", "balance de comprobación"
            ]
            low = text.lower()
            if any(b in low for b in banned):
                return _on_topic_fallback_q4()
            return text

        def safe_ia_feedback(prompt: str, default: str = "", tries: int = 3, base_sleep: float = 0.8) -> str:
            """
            Envuelve ia_feedback con reintentos/backoff y salida segura.
            - Si hay 429/rate limit u otros errores → reintenta con backoff.
            - Si detecta 429 → marca st.session_state['n3_ai_rate_limited']=True y devuelve default.
            - Siempre retorna str; si no hay respuesta útil → default.
            """
            # limpiar flag de rate-limit al iniciar
            st.session_state.pop("n3_ai_rate_limited", None)
            for t in range(tries):
                try:
                    resp = ia_feedback(prompt)
                    if resp is None:
                        return default
                    if isinstance(resp, dict):
                        return str(resp.get("text", default))
                    return str(resp)
                except Exception as e:
                    msg = str(e)
                    # Detectar rate-limit explícito
                    if "429" in msg or "rate-limited" in msg.lower() or "rate limit" in msg.lower() or "rate" in msg.lower():
                        st.session_state["n3_ai_rate_limited"] = True
                        # devolvemos default (silencioso) para activar fallbacks
                        return default
                    # Otros errores temporales → backoff breve y reintentar
                    time.sleep(min(base_sleep * (2 ** t), 4.0))
                    continue
            return default

        # ===== ESCENARIO Q5 AJUSTADO (SIN DECIMALES RAROS) =====
        def q5_scenario():
            # Escenario EXACTO:
            # Día 1: 80 u @ 12
            # Día 2: compra 40 u @ 15
            # Día 3: venta 100 u
            # Día 4: devolución en compra 10 u
            # Día 5: devolución en venta 10 u
            return {
                "inv0_u":   80,    # Día 1: unidades iniciales
                "inv0_pu":  12.0,  # Día 1: precio unitario inicial
                "comp1_u":  40,    # Día 2: compra unidades
                "comp1_pu": 15.0,  # Día 2: compra precio unitario
                "venta_u":  100,   # Día 3: venta unidades
                "dev_comp": 10,    # Día 4: devolución en compras (unidades que salen)
                "dev_venta":10,    # Día 5: devolución en ventas (unidades que reingresan)
            }

        def _scenario_signature(sc: dict) -> str:
            return f'{sc["inv0_u"]}-{sc["inv0_pu"]}-{sc["comp1_u"]}-{sc["comp1_pu"]}-{sc["venta_u"]}-{sc["dev_comp"]}-{sc["dev_venta"]}'

        def _sum_layers(layers):
            q = sum(q for q, _ in layers)
            v = sum(q*p for q, p in layers)
            pu = (v/q) if q > 0 else 0.0
            return q, pu, v

        @st.cache_data(show_spinner=False)
        def build_expected_rows_q5_pp(sc: dict, sig: str):
            """
            Construye la solución esperada del KARDEX PP para Q5 siguiendo:
            - Día 1: saldo inicial solo como SALDO (entradas vacías).
            - Día 2: compra 1, saldo promediado.
            - Día 3: venta al costo promedio vigente.
            - Día 4: devolución de compra a **costo de compra** (15).
            - Día 5: devolución de venta a **costo de la venta original** (13).
            Las celdas no aplicables se marcan con "" para que el validador NO las exija.
            """
            inv0_u_ex  = sc["inv0_u"]
            inv0_pu_ex = sc["inv0_pu"]
            comp1_u    = sc["comp1_u"]
            comp1_pu   = sc["comp1_pu"]
            venta_ex_u = sc["venta_u"]
            dev_comp_u = sc["dev_comp"]
            dev_venta_u= sc["dev_venta"]

            rows = []

            # ===== Día 1: Saldo inicial =====
            layers = [[float(inv0_u_ex), float(inv0_pu_ex)]]
            s_q, s_p, s_v = _sum_layers(layers)
            rows.append({
                "Fecha": "Día 1", "Descripción": "Saldo inicial",
                "Entrada_cant": "", "Entrada_pu": "", "Entrada_total": "",
                "Salida_cant": "",  "Salida_pu": "",  "Salida_total": "",
                "Saldo_cant": s_q,  "Saldo_pu": round(s_p, 2), "Saldo_total": round(s_v, 2)
            })

            # ===== Día 2: Compra 1 (promediada) =====
            ent_tot = comp1_u * comp1_pu
            q_new = s_q + comp1_u
            v_new = s_v + ent_tot
            p_new = (v_new / q_new) if q_new > 0 else 0.0
            layers = [[q_new, p_new]]
            s_q, s_p, s_v = _sum_layers(layers)

            rows.append({
                "Fecha": "Día 2", "Descripción": "Compra 1",
                "Entrada_cant": comp1_u, "Entrada_pu": comp1_pu, "Entrada_total": ent_tot,
                "Salida_cant": "", "Salida_pu": "", "Salida_total": "",
                "Saldo_cant": s_q, "Saldo_pu": round(s_p, 2), "Saldo_total": round(s_v, 2)
            })

            # Costo unitario de la venta (para la devolución de venta)
            sale_unit_cost = p_new  # aquí será 13.0

            # ===== Día 3: Venta (al costo promedio vigente) =====
            sale_q  = min(venta_ex_u, s_q)
            sale_pu = sale_unit_cost
            sale_tot = sale_q * sale_pu
            q2 = s_q - sale_q
            v2 = s_v - sale_tot
            p2 = (v2 / q2) if q2 > 0 else 0.0
            layers = [[q2, p2]] if q2 > 0 else []
            s_q, s_p, s_v = _sum_layers(layers)

            rows.append({
                "Fecha": "Día 3", "Descripción": "Venta",
                "Entrada_cant": "", "Entrada_pu": "", "Entrada_total": "",
                "Salida_cant": sale_q, "Salida_pu": sale_pu, "Salida_total": sale_tot,
                "Saldo_cant": s_q, "Saldo_pu": round(s_p, 2), "Saldo_total": round(s_v, 2)
            })

            # ===== Día 4: Devolución de compra (a costo de compra) =====
            # Sacamos unidades desde el promedio, pero valoradas a 15 (costo de la compra original).
            take_q  = min(dev_comp_u, s_q)
            take_pu = comp1_pu          # 15.0
            take_val = take_q * take_pu
            q4 = max(s_q - take_q, 0)
            v4 = max(s_v - take_val, 0.0)
            p4 = (v4 / q4) if q4 > 0 else 0.0
            layers = [[q4, p4]] if q4 > 0 else []
            s_q, s_p, s_v = _sum_layers(layers)

            rows.append({
                "Fecha": "Día 4", "Descripción": "Devolución de compra",
                "Entrada_cant": "", "Entrada_pu": "", "Entrada_total": "",
                "Salida_cant": take_q, "Salida_pu": take_pu, "Salida_total": take_val,
                "Saldo_cant": q4, "Saldo_pu": round(p4, 2), "Saldo_total": round(v4, 2)
            })

            # ===== Día 5: Devolución de venta (reingreso a costo de venta) =====
            in_q  = dev_venta_u
            in_pu = sale_unit_cost      # 13.0
            in_val = in_q * in_pu
            q5 = s_q + in_q
            v5 = s_v + in_val
            p5 = (v5 / q5) if q5 > 0 else 0.0

            rows.append({
                "Fecha": "Día 5", "Descripción": "Devolución de venta (reingreso)",
                "Entrada_cant": in_q, "Entrada_pu": in_pu, "Entrada_total": in_val,
                "Salida_cant": "", "Salida_pu": "", "Salida_total": "",
                "Saldo_cant": q5, "Saldo_pu": round(p5, 2), "Saldo_total": round(v5, 2)
            })

            return rows

        @st.cache_data(show_spinner=False)
        def blank_df_for_editor(sig: str, expected_rows: list) -> _pd.DataFrame:
            """Devuelve un DF vacío (sin valores por defecto) según las filas esperadas (cacheado por firma)."""
            def _blank_like(r):
                return {
                    "Fecha": r["Fecha"], "Descripción": r["Descripción"],
                    "Entrada_cant": None, "Entrada_pu": None, "Entrada_total": None,
                    "Salida_cant": None,  "Salida_pu": None,  "Salida_total": None,
                    "Saldo_cant": None,   "Saldo_pu": None,   "Saldo_total": None
                }
            return _pd.DataFrame([_blank_like(r) for r in expected_rows])

        # =========================
        # Q1–Q3: Selección múltiple + Q4 abierta + Q5 ejercicio
        # =========================
        with st.form("n3_eval_form_mcq"):
            st.markdown("### Preguntas de selección múltiple")

            q1 = st.radio(
                "1) ¿Cuál de las siguientes afirmaciones describe correctamente una devolución en compras?",
                [
                    "a) Aumenta las compras brutas y el valor total del inventario.",
                    "b) Disminuye las compras brutas y el costo total disponible para la venta.",
                    "c) Aumenta el Costo de la Mercancía Vendida (CMV) del periodo.",
                    "d) No afecta el inventario ni las compras netas.",
                ],
                index=None,
                key="n3_q1"
            )

            q2 = st.radio(
                "2) Bajo el método PEPS (FIFO), ¿cómo deben valorarse las unidades que regresan al inventario por una devolución en ventas?",
                [
                    "a) Al costo promedio vigente del inventario.",
                    "b) Con el costo más reciente de compra.",
                    "c) Con los costos de las primeras unidades que salieron (más antiguas).",
                    "d) Con un costo determinado libremente por la empresa.",
                ],
                index=None,
                key="n3_q2"
            )

            q3 = st.radio(
                "3) ¿Cuál es el efecto de una devolución en ventas sobre el Costo de la Mercancía Vendida (CMV)?",
                [
                    "a) Incrementa el CMV neto del periodo.",
                    "b) No tiene ningún efecto sobre el CMV.",
                    "c) Disminuye el CMV neto del periodo porque parte del costo vendido regresa al inventario.",
                    "d) Aumenta el CMV bruto y reduce la utilidad.",
                ],
                index=None,
                key="n3_q3"
            )

            st.markdown("### Pregunta abierta")
            q4_text = st.text_area(
                "4) Explica por qué es importante registrar las devoluciones de forma coherente con el método (PP, PEPS o UEPS), y qué consecuencias tendría hacerlo mal en los estados financieros.",
                height=140,
                key="n3_q4"
            )

            ask_ai_q4 = st.checkbox("💬 Pedir feedback SOLO para la pregunta 4 (opcional)", key="n3_eval_ai_q4", value=False)

            st.markdown("### Ejercicio tipo práctica (Promedio Ponderado)")
            st.caption("**Q5**: Completa el KARDEX D1–D5. Tabla completamente en blanco. **Método: Promedio Ponderado**.")

            # ===== Escenario Q5 (mostrado al estudiante) =====
            _sc = q5_scenario()
            _sig = _scenario_signature(_sc)

            st.markdown(
                f"""
    **Escenario — Empresa “Comercial ABC S.A.S.”**

    La empresa comercializa bolsos y registró las siguientes transacciones durante la semana:

    - **Día 1:** La empresa reporta un saldo inicial de **{_sc['inv0_u']} unidades** de bolsos a **${_sc['inv0_pu']:.2f}** cada una.  
    - **Día 2:** Realizó una **compra** de **{_sc['comp1_u']} unidades** a **${_sc['comp1_pu']:.2f}** cada una.  
    - **Día 3:** Efectuó una **venta** de **{_sc['venta_u']} unidades**.  
    - **Día 4:** Se efectuó una **devolución en compra** de **{_sc['dev_comp']} unidades**, por defectos de fábrica.  
    - **Día 5:** Un cliente realizó una **devolución en venta** de **{_sc['dev_venta']} unidades**, que regresan al inventario.

    Completa el **KARDEX** registrando todas las operaciones en orden cronológico, aplicando el **método Promedio Ponderado**.
    """
            )

            # Filas esperadas (para validar luego)
            expected_rows_q5 = build_expected_rows_q5_pp(_sc, _sig)

            # DF vacío para el editor
            df_q5_blank = blank_df_for_editor(_sig, expected_rows_q5)

            tail_col_config = {
                "Fecha":        st.column_config.TextColumn(disabled=True),
                "Descripción":  st.column_config.TextColumn(disabled=True),
                "Entrada_cant": st.column_config.NumberColumn(step=1, help="Unidades que ingresan"),
                "Entrada_pu":   st.column_config.NumberColumn(step=0.01, help="Costo unitario de la entrada"),
                "Entrada_total":st.column_config.NumberColumn(step=0.01, help="Valor total de la entrada"),
                "Salida_cant":  st.column_config.NumberColumn(step=1, help="Unidades que salen"),
                "Salida_pu":    st.column_config.NumberColumn(step=0.01, help="Costo unitario de la salida"),
                "Salida_total": st.column_config.NumberColumn(step=0.01, help="Valor total de la salida"),
                "Saldo_cant":   st.column_config.NumberColumn(step=1, help="Unidades en saldo"),
                "Saldo_pu":     st.column_config.NumberColumn(step=0.01, help="Costo unitario del saldo"),
                "Saldo_total":  st.column_config.NumberColumn(step=0.01, help="Valor total del saldo"),
            }

            edited_q5 = st.data_editor(
                df_q5_blank,
                use_container_width=True,
                num_rows="fixed",
                column_config=tail_col_config,
                hide_index=True,
                key="n3_q5_editor_v3"
            )

            submitted = st.form_submit_button("🧪 Enviar evaluación")

        # =========================
        # Corrección y resultado
        # =========================
        if submitted:
            correct = {
                "n3_q1": "b) Disminuye las compras brutas y el costo total disponible para la venta.",
                "n3_q2": "c) Con los costos de las primeras unidades que salieron (más antiguas).",
                "n3_q3": "c) Disminuye el CMV neto del periodo porque parte del costo vendido regresa al inventario.",
            }
            answers = {"n3_q1": q1, "n3_q2": q2, "n3_q3": q3}
            q1_ok = (answers["n3_q1"] == correct["n3_q1"])
            q2_ok = (answers["n3_q2"] == correct["n3_q2"])
            q3_ok = (answers["n3_q3"] == correct["n3_q3"])

            # ---- Q4 abierta con IA ----
            def grade_open_q4(text: str):
                prompt = (
                    "Evalúa SOLO sobre devoluciones y coherencia con el método (PP/PEPS/UEPS).\n"
                    "Primera línea EXACTA: 'SCORE: 1' si el texto está en tema y cumple ≥2 de:\n"
                    "(1) costo correcto de la devolución según método; (2) efecto en CMV/compras netas;\n"
                    "(3) efecto en saldo/costo unitario; (4) impacto en estados/comparabilidad.\n"
                    "Si no, 'SCORE: 0'. Luego 2–4 líneas de feedback docente, sin emojis.\n"
                    "Prohibido: ecuación contable, pasivos/patrimonio, temas no relacionados.\n\n"
                    f"RESPUESTA:\n{text}"
                )

                # limpiar flag rate-limit y pedir IA de forma segura
                st.session_state.pop("n3_ai_rate_limited", None)
                raw = safe_ia_feedback(prompt, default="")
                sraw = str(raw or "")
                first = sraw.strip().splitlines()[0].strip() if sraw.strip() else ""
                score1 = 1 if first.upper().endswith("1") else 0
                fb = "\n".join(sraw.strip().splitlines()[1:]).strip()

                # fallback pedagógico si el modelo no devolvió nada útil o hubo rate-limit
                rate_limited = bool(st.session_state.get("n3_ai_rate_limited", False))
                if not fb or rate_limited:
                    fb = _on_topic_fallback_q4()

                # si el estudiante se desvió a ecuación/pasivos/patrimonio → fuerza 0 + feedback correcto
                banned_student = ["activo = pasivo + patrimonio", "ecuación contable", "pasivo", "patrimonio"]
                if any(b in (text or "").lower() for b in banned_student):
                    score1 = 0
                    fb = _on_topic_fallback_q4()

                return score1, fb

            q4_score1, q4_fb = grade_open_q4(q4_text or "")

            if ask_ai_q4:
                q4_fb = _sanitize_on_topic_q4(q4_fb)
            else:
                q4_fb = ""

            # ---- Q5 validación ----
            TOL = 0.5
            num_keys = [
                "Entrada_cant","Entrada_pu","Entrada_total",
                "Salida_cant","Salida_pu","Salida_total",
                "Saldo_cant","Saldo_pu","Saldo_total"
            ]

            def _to_float(x):
                try:
                    if x in ("", None): return None
                    return float(x)
                except:
                    return None

            def _near(a, b, tol=TOL):
                if a is None or b in ("", None):
                    return False
                try:
                    return abs(float(a) - float(b)) <= tol
                except:
                    return False

            ok_rows = []
            for i in range(len(expected_rows_q5)):
                user_row = edited_q5.iloc[i].to_dict()
                exp_row  = expected_rows_q5[i]
                ok_cells = []
                for k in num_keys:
                    exp_val = exp_row[k]
                    usr_val = _to_float(user_row.get(k, ""))
                    ok_cells.append(True if exp_val == "" else _near(usr_val, exp_val))
                ok_rows.append(all(ok_cells))
            q5_ok = all(ok_rows)

            total_hits = int(q1_ok) + int(q2_ok) + int(q3_ok) + int(q4_score1) + int(q5_ok)
            passed = (total_hits == 5)

            try:
                record_attempt(username, level=3, score=total_hits, passed=passed)
            except Exception:
                pass

            st.markdown("### Resultado")
            cA, cB = st.columns(2)
            with cA: st.metric("Aciertos", f"{total_hits}/5")
            with cB: st.metric("Estado", "APROBADO ✅" if passed else "NO APROBADO ❌")

            with st.expander("Detalle de corrección"):
                st.write(f"**Q1:** {'✅' if q1_ok else '❌'}")
                st.write(f"**Q2:** {'✅' if q2_ok else '❌'}")
                st.write(f"**Q3:** {'✅' if q3_ok else '❌'}")
                st.write(f"**Q4 (abierta):** {'✅' if q4_score1==1 else '❌'}")
                if ask_ai_q4 and q4_fb:
                    st.write("**Feedback Q4 (IA):**")
                    st.write(q4_fb)
                st.write(f"**Q5 (KARDEX PP):** {'✅' if q5_ok else '❌'}")
                if not q5_ok:
                    st.caption("Pistas: revisa el costo correcto en devoluciones y el saldo después de cada operación.")

            if passed:
                try:
                    set_level_passed(st.session_state.get('progress_col'), username, "level3", total_hits)
                except Exception:
                    pass
                st.success("🎉 ¡Aprobaste la Evaluación del Nivel 3! Avanzas al siguiente módulo.")
                try:
                    st.session_state["sidebar_next_select"] = "Nivel 4: Estado de Resultados"
                    start_celebration(
                        message_md=(
                            "<b>¡Nivel 3 dominado!</b> 🔁📦<br><br>"
                            "Manejaste devoluciones y su efecto en el KARDEX y CMV. "
                            "Ahora vamos al <b>Estado de Resultados</b>."
                        ),
                        next_label="Ir al Nivel 4",
                        next_key_value="Nivel 4: Estado de Resultados"
                    )
                except Exception:
                    pass
            else:
                st.error("No aprobado. Debes acertar 5/5. Repasa la lógica de devoluciones y vuelve a intentar.")

# ===========================
# NIVEL 4 (Estado de Resultados)
# ===========================
def page_level4(username):
    st.title("Nivel 4 · Construcción del Estado de Resultados (simplificado)")

    tabs = st.tabs(["🎧 Teoría", "🛠 Ejemplo guiado", "🎮 Práctica (IA)", "🏁 Evaluación final + Encuesta"])

    # =====================================================
    # TAB 1 — TEORÍA
    # =====================================================
    with tabs[0]:
        st.subheader("Teoría · Estado de Resultados (sistema perpetuo con devoluciones)")

        intro = """
    En una empresa comercial que utiliza **sistema perpetuo**, el **Estado de Resultados** resume el desempeño del periodo. 
    El **CMV** no se calcula con la fórmula periódica; **se determina directamente del KARDEX** según el método de inventario (**Promedio Ponderado, PEPS o UEPS**), 
    incluyendo el tratamiento de **devoluciones en compras** (ajustan el pool de costo) y **devoluciones en ventas** (reingreso que **disminuye el costo de la mercadería vendida** presentado en el Estado de Resultados).
        """
        st.markdown(intro)

        st.markdown("#### Estructura general")
        st.markdown(
            """
    - **Ventas netas**  
    - **Costo de la mercadería vendida (CMV)** → *desde KARDEX (PP/PEPS/UEPS), con devoluciones*  
    - **Utilidad bruta** = Ventas netas − CMV  
    - **Gastos operativos** (administración y ventas)  
    - **Resultado operativo** = Utilidad bruta − Gastos operativos  
    - **Otros ingresos** y **Otros egresos**  
    - **Utilidad antes de impuesto** = Resultado operativo + Otros ingresos − Otros egresos  
    - **Impuesto**  
    - **Utilidad neta** = Utilidad antes de impuesto − Impuesto
            """
        )

        st.markdown("---")
        st.markdown("### ¿Cómo se calcula cada rubro? (haz tu selección)")
        rubro = st.selectbox(
            "Selecciona un rubro para ver su fórmula y explicación",
            [
                "Ventas netas",
                "CMV",
                "Utilidad bruta",
                "Gastos operativos",
                "Resultado operativo",
                "Otros ingresos",
                "Otros egresos",
                "Utilidad antes de impuesto",
                "Impuesto",
                "Utilidad neta",
            ],
            index=None,
            placeholder="Elige un rubro…"
        )

        # Render dinámico por rubro
        if rubro == "Ventas netas":
            st.markdown("**Ventas netas**")
            st.latex(r"\text{Ventas Netas}=\text{Ventas Brutas}-\text{Devoluciones/Descuentos sobre Ventas}")
            st.caption(
                "Las devoluciones/descuentos sobre ventas **reducen** las ventas brutas para obtener las ventas netas. "
                "En el sistema perpetuo, estas partidas también impactan las cuentas de ingresos y la presentación del periodo."
            )

        elif rubro == "CMV":
            st.markdown("**Costo de la Mercadería Vendida (CMV) en sistema perpetuo**")
            st.markdown(
                """
    En **perpetuo**, el **CMV** que se presenta en el Estado de Resultados es el **costo neto** de la mercancía vendida,
    construido directamente desde el **KARDEX**, **según el método**:

    - **Promedio Ponderado (PP):** cada venta usa el **promedio vigente** del momento.  
      - La **devolución en compras** sale al proveedor al promedio vigente (reduce el pool de costo).  
      - La **devolución en ventas** reingresa al promedio vigente y **disminuye el CMV presentado**.

    - **PEPS (FIFO):** la venta consume **capas más antiguas** primero.  
      - La devolución en compras se registra contra la capa correspondiente.  
      - La devolución en ventas reingresa con el costo de las capas antiguas que habían salido.

    - **UEPS (LIFO):** la venta consume **capas más recientes** primero.  
      - La devolución en compras afecta las capas recientes.  
      - La devolución en ventas reingresa con el costo de las capas recientes que habían salido.

    En la práctica, el **CMV del Estado de Resultados** es el **costo definido para las ventas menos el costo de las unidades devueltas por clientes**.
    """
            )

        elif rubro == "Utilidad bruta":
            st.markdown("**Utilidad bruta**")
            st.latex(r"\text{Utilidad Bruta}=\text{Ventas Netas}-\text{CMV}")
            st.caption(
                "El CMV proviene del KARDEX (perpetuo), incorporando devoluciones y el método de valoración aplicado. En el ER solo se muestra un único rubro de CMV."
            )

        elif rubro == "Gastos operativos":
            st.markdown("**Gastos operativos**")
            st.caption(
                "Incluyen gastos de administración y ventas. No forman parte del CMV; se restan después para obtener el resultado operativo."
            )

        elif rubro == "Resultado operativo":
            st.markdown("**Resultado operativo**")
            st.latex(r"\text{Resultado Operativo}=\text{Utilidad Bruta}-\text{Gastos Operativos}")
            st.caption("Mide el desempeño del giro del negocio antes de partidas no operativas e impuestos.")

        elif rubro == "Otros ingresos":
            st.markdown("**Otros ingresos**")
            st.caption("Ingresos no operativos (p. ej., rendimientos financieros). Se **suman** al resultado operativo.")

        elif rubro == "Otros egresos":
            st.markdown("**Otros egresos**")
            st.caption("Egresos no operativos (p. ej., gastos financieros). Se **restan** del resultado operativo.")

        elif rubro == "Utilidad antes de impuesto":
            st.markdown("**Utilidad antes de impuesto**")
            st.latex(r"\text{UAI}=\text{Resultado Operativo}+\text{Otros Ingresos}-\text{Otros Egresos}")
            st.caption("Base previa al cálculo del impuesto del periodo.")

        elif rubro == "Impuesto":
            st.markdown("**Impuesto**")
            st.caption(
                "Carga tributaria del periodo según normativa aplicable. Puede modelarse como una tasa sobre la UAI o con reglas específicas."
            )

        elif rubro == "Utilidad neta":
            st.markdown("**Utilidad neta**")
            st.latex(r"\text{Utilidad Neta}=\text{UAI}-\text{Impuesto}")
            st.caption("Resultado final del periodo después de impuestos.")

        st.markdown("---")
        with st.expander("🔊 Escuchar explicación"):
            full_text = "\n\n".join([
                "Resumen: En sistema perpetuo, el CMV se obtiene del KARDEX y no con la fórmula periódica.",
                "Las devoluciones en compras reducen el pool de costo y las devoluciones en ventas disminuyen el costo de la mercadería vendida.",
                "Los métodos PP, PEPS y UEPS determinan el costo aplicado en cada movimiento.",
                "La estructura del estado de resultados incluye ventas netas, CMV, utilidad bruta, gastos operativos, resultado operativo, otros ingresos y egresos, UAI, impuesto y utilidad neta."
            ])
            speak_block(full_text, key_prefix="teo-n4", lang_hint="es")

    # =====================================================
    # TAB 2 — EJEMPLO GUIADO
    # =====================================================
    with tabs[1]:
        st.subheader("Ejemplo guiado: KARDEX + Estado de Resultados")

        import json as _json
        import streamlit.components.v1 as components

        # =========================
        # Parámetros del escenario (default)
        # =========================
        esc = {
            "inv0_u":   80,      # Día 1: inventario inicial (u)
            "inv0_pu":  10.0,    # Día 1: costo unitario
            "comp1_u":  40,      # Día 2: compra (u)
            "comp1_pu": 11.0,    # Día 2: compra ($/u)
            "venta_u":  90,      # Día 3: venta (u)
            "p_venta":  20.0,    # Día 3: precio de venta ($/u) para Ventas Brutas
            "dev_comp": 8,       # Día 4: devolución en compra (u) que salen a proveedor
            "dev_vent": 6,       # Día 5: devolución en venta (u) que regresan al inventario
            # PyG del periodo (no inventario): gastos y otros
            "gastos_operativos": [
                ("Publicidad y promociones", 120.0),
                ("Servicios (luz/agua/internet)", 80.0),
                ("Gastos administrativos", 150.0),
            ],
            "otros_ingresos": [
                ("Intereses ganados", 40.0),
            ],
            "otros_egresos": [
                ("Multas y sanciones", 20.0),
            ],
            "tasa_impuesto": 0.30,  # 30%
        }

        # =========================
        # Panel de ENTRADA (variables PyG)
        # =========================
        st.markdown("### Parámetros del Estado de Resultados (editables)")
        c1, c2 = st.columns(2)
        with c1:
            venta_u_in = st.number_input(
                "Unidades vendidas (Ventas brutas)",
                min_value=0, value=int(esc["venta_u"]), step=1, key="n4_venta_u"
            )
            dev_vent_in = st.number_input(
                "Devolución en ventas (unid.)",
                min_value=0, value=int(esc["dev_vent"]), step=1, key="n4_dev_vent"
            )
        with c2:
            p_venta_in = st.number_input(
                "Precio de venta ($/u)",
                min_value=0.0, value=float(esc["p_venta"]), step=0.5, key="n4_p_venta"
            )
            tasa_imp_in = st.slider(
                "Tasa de impuesto (%)",
                min_value=0, max_value=50, value=int(esc["tasa_impuesto"]*100), step=1, key="n4_tax"
            )

        st.markdown("#### Gastos operativos (desglosados)")
        gcol1, gcol2, gcol3 = st.columns(3)
        g1_name = gcol1.text_input("Ítem 1", value=esc["gastos_operativos"][0][0], key="n4_go_n1")
        g1_val  = gcol1.number_input("Valor 1", min_value=0.0, value=float(esc["gastos_operativos"][0][1]), step=10.0, key="n4_go_v1")
        g2_name = gcol2.text_input("Ítem 2", value=esc["gastos_operativos"][1][0], key="n4_go_n2")
        g2_val  = gcol2.number_input("Valor 2", min_value=0.0, value=float(esc["gastos_operativos"][1][1]), step=10.0, key="n4_go_v2")
        g3_name = gcol3.text_input("Ítem 3", value=esc["gastos_operativos"][2][0], key="n4_go_n3")
        g3_val  = gcol3.number_input("Valor 3", min_value=0.0, value=float(esc["gastos_operativos"][2][1]), step=10.0, key="n4_go_v3")

        st.markdown("#### Otros rubros")
        ocol1, ocol2 = st.columns(2)
        oi_val = ocol1.number_input("Otros ingresos (total)", min_value=0.0, value=float(esc["otros_ingresos"][0][1]), step=10.0, key="n4_oi")
        oe_val = ocol2.number_input("Otros egresos (total)", min_value=0.0, value=float(esc["otros_egresos"][0][1]), step=10.0, key="n4_oe")

        # Aplicar entradas al escenario
        esc["venta_u"] = int(venta_u_in)
        esc["p_venta"] = float(p_venta_in)
        esc["dev_vent"] = int(dev_vent_in)
        esc["tasa_impuesto"] = float(tasa_imp_in) / 100.0
        esc["gastos_operativos"] = [
            (g1_name.strip() or "Gasto 1", float(g1_val)),
            (g2_name.strip() or "Gasto 2", float(g2_val)),
            (g3_name.strip() or "Gasto 3", float(g3_val)),
        ]
        esc["otros_ingresos"] = [("Otros ingresos", float(oi_val))]
        esc["otros_egresos"] = [("Otros egresos", float(oe_val))]

        # =========================
        # Utilidades internas
        # =========================
        def _sum_layers(layers):
            q = sum(q for q, _ in layers)
            v = sum(q * p for q, p in layers)
            pu = (v / q) if q > 0 else 0.0
            return q, pu, v

        def _consume_layers_detail(layers, qty_out, fifo=True):
            """
            Consumo por capas.
            Retorna (detalles_de_venta, capas_restantes_en_orden_FIFO)
            detalles: [(q_take, pu_take, q_take*pu_take), ...]
            """
            order = layers[:] if fifo else layers[::-1]
            remaining = qty_out
            sale_details = []
            new_layers = []
            for q, pu in order:
                if remaining <= 0:
                    new_layers.append([q, pu]); continue
                take = min(q, remaining)
                if take > 0:
                    sale_details.append((take, pu, take * pu))
                    rest = q - take
                    remaining -= take
                    if rest > 0:
                        new_layers.append([rest, pu])
            # devolver las capas en orden FIFO natural
            final_layers = new_layers if fifo else new_layers[::-1]
            return sale_details, final_layers

        def pesos(v):
            try:
                return f"${v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
            except:
                return str(v)

        # =========================
        # Builder KARDEX + métricas PyG dependientes del método
        # =========================
        def build_kardex_and_metrics(metodo: str):
            inv0_u, inv0_pu = esc["inv0_u"], esc["inv0_pu"]
            c1_u, c1_pu     = esc["comp1_u"], esc["comp1_pu"]
            v_u, p_venta    = esc["venta_u"], esc["p_venta"]
            dcomp_u         = esc["dev_comp"]
            dvent_u         = esc["dev_vent"]

            fifo = True if "PEPS" in metodo else False if "UEPS" in metodo else None

            # --- Día 1: saldo inicial
            rows = []
            layers = [[float(inv0_u), float(inv0_pu)]] if inv0_u > 0 else []
            s_q, s_pu, s_v = _sum_layers(layers)
            rows.append({
                "Fecha":"Día 1", "Descripción":"Saldo inicial",
                "Entrada_cant":None, "Entrada_pu":None, "Entrada_total":None,
                "Salida_cant":None,  "Salida_pu":None,  "Salida_total":None,
                "Saldo_cant": int(s_q), "Saldo_pu": round(s_pu,2), "Saldo_total": round(s_v,2)
            })

            # --- Día 2: compra
            if metodo == "Promedio Ponderado":
                ent_tot = c1_u * c1_pu
                q_new = s_q + c1_u
                v_new = s_v + ent_tot
                p_new = (v_new / q_new) if q_new > 0 else 0.0
                layers = [[q_new, p_new]]
                s_q, s_pu, s_v = _sum_layers(layers)
                rows.append({
                    "Fecha":"Día 2", "Descripción":"Compra",
                    "Entrada_cant": c1_u, "Entrada_pu": round(c1_pu,2), "Entrada_total": round(ent_tot,2),
                    "Salida_cant":None, "Salida_pu":None, "Salida_total":None,
                    "Saldo_cant": int(s_q), "Saldo_pu": round(s_pu,2), "Saldo_total": round(s_v,2)
                })
            else:
                # PEPS / UEPS: agregamos una nueva capa sin fila "Saldo (día 1)"
                ent_tot = c1_u * c1_pu
                layers.append([float(c1_u), float(c1_pu)])  # nueva capa
                s_q, s_pu, s_v = _sum_layers(layers)
                rows.append({
                    "Fecha":"Día 2", "Descripción":"Compra",
                    "Entrada_cant": c1_u, "Entrada_pu": round(c1_pu,2), "Entrada_total": round(ent_tot,2),
                    "Salida_cant":None, "Salida_pu":None, "Salida_total":None,
                    "Saldo_cant": int(s_q), "Saldo_pu": round(s_pu,2), "Saldo_total": round(s_v,2)
                })

            # --- Día 3: venta
            if v_u > 0 and s_q > 0:
                if metodo == "Promedio Ponderado":
                    sale_q  = min(v_u, int(s_q))
                    sale_pu = layers[0][1] if layers else 0.0
                    sale_tot= sale_q * sale_pu
                    q2 = s_q - sale_q
                    v2 = s_v - sale_tot
                    p2 = (v2/q2) if q2 > 0 else 0.0
                    layers = [[q2, p2]] if q2 > 0 else []
                    s_q, s_pu, s_v = _sum_layers(layers)
                    rows.append({
                        "Fecha":"Día 3", "Descripción":"Venta",
                        "Entrada_cant":None, "Entrada_pu":None, "Entrada_total":None,
                        "Salida_cant": sale_q, "Salida_pu": round(sale_pu,2), "Salida_total": round(sale_tot,2),
                        "Saldo_cant": int(s_q), "Saldo_pu": round(s_pu,2), "Saldo_total": round(s_v,2)
                    })
                    cmv_bruto = sale_tot
                    sale_details = [(sale_q, sale_pu, sale_tot)]
                else:
                    sale_details, layers_after = _consume_layers_detail(layers, v_u, fifo=(fifo is True))
                    cmv_bruto = sum(t for _,_,t in sale_details)
                    running_layers = [l[:] for l in layers]
                    for i, (q_take, pu_take, tot_take) in enumerate(sale_details, start=1):
                        _, running_layers = _consume_layers_detail(running_layers, q_take, fifo=(fifo is True))
                        rq, rpu, rv = _sum_layers(running_layers)
                        rows.append({
                            "Fecha":"Día 3", "Descripción": f"Venta tramo {i} ({'PEPS' if fifo else 'UEPS'})",
                            "Entrada_cant":None, "Entrada_pu":None, "Entrada_total":None,
                            "Salida_cant": int(q_take), "Salida_pu": round(pu_take,2), "Salida_total": round(tot_take,2),
                            "Saldo_cant": int(rq), "Saldo_pu": round(rpu,2), "Saldo_total": round(rv,2)
                        })
                    layers = layers_after
                    s_q, s_pu, s_v = _sum_layers(layers)
            else:
                cmv_bruto = 0.0
                rows.append({
                    "Fecha":"Día 3", "Descripción":"Venta",
                    "Entrada_cant":None, "Entrada_pu":None, "Entrada_total":None,
                    "Salida_cant":None, "Salida_pu":None, "Salida_total":None,
                    "Saldo_cant": int(s_q), "Saldo_pu": round(s_pu,2), "Saldo_total": round(s_v,2)
                })
                sale_details = []

            # --- Día 4: devolución en compra (salida a proveedor)
            if metodo == "Promedio Ponderado":
                take_q  = min(esc["dev_comp"], s_q)
                take_pu = s_pu
                take_val= take_q * take_pu
                q4 = s_q - take_q
                v4 = s_v - take_val
                p4 = (v4/q4) if q4 > 0 else 0.0
                layers = [[q4, p4]] if q4 > 0 else []
                s_q, s_pu, s_v = _sum_layers(layers)
                rows.append({
                    "Fecha":"Día 4", "Descripción":"Devolución de compra (a proveedor)",
                    "Entrada_cant":None, "Entrada_pu":None, "Entrada_total":None,
                    "Salida_cant": int(take_q), "Salida_pu": round(take_pu,2), "Salida_total": round(take_val,2),
                    "Saldo_cant": int(s_q), "Saldo_pu": round(s_pu,2), "Saldo_total": round(s_v,2)
                })
                dev_comp_valor = take_val
            else:
                send_back = esc["dev_comp"]
                dev_comp_valor = 0.0
                rev = layers[::-1]
                new_rev = []
                for q, pu in rev:
                    if send_back <= 0:
                        new_rev.append([q, pu]); continue
                    take = min(q, send_back)
                    dev_comp_valor += take * pu
                    rest = q - take
                    send_back -= take
                    if rest > 0:
                        new_rev.append([rest, pu])
                layers = new_rev[::-1]
                s_q, s_pu, s_v = _sum_layers(layers)
                rows.append({
                    "Fecha":"Día 4", "Descripción":"Devolución de compra (a proveedor)",
                    "Entrada_cant":None, "Entrada_pu":None, "Entrada_total":None,
                    "Salida_cant": esc["dev_comp"], "Salida_pu":None, "Salida_total": round(dev_comp_valor,2),
                    "Saldo_cant": int(s_q), "Saldo_pu": round(s_pu,2), "Saldo_total": round(s_v,2)
                })

            # --- Día 5: devolución en venta (reingreso)
            if esc["dev_vent"] > 0:
                if metodo == "Promedio Ponderado":
                    in_q  = esc["dev_vent"]
                    in_pu = s_pu
                    in_val= in_q * in_pu
                    q5 = s_q + in_q
                    v5 = s_v + in_val
                    p5 = (v5/q5) if q5 > 0 else 0.0
                    layers = [[q5, p5]]
                    s_q, s_pu, s_v = _sum_layers(layers)
                    rows.append({
                        "Fecha":"Día 5", "Descripción":"Devolución de venta (reingreso)",
                        "Entrada_cant": in_q, "Entrada_pu": round(in_pu,2), "Entrada_total": round(in_val,2),
                        "Salida_cant":None, "Salida_pu":None, "Salida_total":None,
                        "Saldo_cant": int(s_q), "Saldo_pu": round(s_pu,2), "Saldo_total": round(s_v,2)
                    })
                    costo_dev_venta = in_val
                else:
                    devolver = esc["dev_vent"]
                    costo_dev_venta = 0.0
                    details = sale_details[:] if sale_details else []
                    if "PEPS" in metodo:
                        it = 0
                        while devolver > 0 and it < len(details):
                            q_take, pu_take, _ = details[it]
                            use = min(devolver, q_take)
                            costo_dev_venta += use * pu_take
                            layers.append([float(use), float(pu_take)])
                            devolver -= use
                            it += 1
                    else:
                        it = len(details) - 1
                        while devolver > 0 and it >= 0:
                            q_take, pu_take, _ = details[it]
                            use = min(devolver, q_take)
                            costo_dev_venta += use * pu_take
                            layers.append([float(use), float(pu_take)])
                            devolver -= use
                            it -= 1

                    s_q, s_pu, s_v = _sum_layers(layers)
                    rows.append({
                        "Fecha":"Día 5", "Descripción":"Devolución de venta (reingreso)",
                        "Entrada_cant": esc["dev_vent"], "Entrada_pu":None, "Entrada_total": round(costo_dev_venta,2),
                        "Salida_cant":None, "Salida_pu":None, "Salida_total":None,
                        "Saldo_cant": int(s_q), "Saldo_pu": round(s_pu,2), "Saldo_total": round(s_v,2)
                    })
            else:
                costo_dev_venta = 0.0
                rows.append({
                    "Fecha":"Día 5", "Descripción":"Devolución de venta (reingreso)",
                    "Entrada_cant":None, "Entrada_pu":None, "Entrada_total":None,
                    "Salida_cant":None, "Salida_pu":None, "Salida_total":None,
                    "Saldo_cant": int(s_q), "Saldo_pu": round(s_pu,2), "Saldo_total": round(s_v,2)
                })

            # =========================
            # Métricas PyG del periodo
            # =========================
            ventas_brutas       = esc["venta_u"] * esc["p_venta"]
            dev_ventas_brutas   = esc["dev_vent"] * esc["p_venta"]
            ventas_netas        = ventas_brutas - dev_ventas_brutas

            compras_brutas      = esc["comp1_u"] * esc["comp1_pu"]
            if metodo == "Promedio Ponderado":
                dev_compras_valor = 0.0
                for r in rows:
                    if r["Fecha"]=="Día 4" and "Devolución de compra" in r["Descripción"]:
                        dev_compras_valor = r["Salida_total"] if r["Salida_total"] != "" else 0.0
                        break
            else:
                # dev_comp_valor calculado en D4 para PEPS/UEPS
                try:
                    dev_compras_valor  # noqa
                except NameError:
                    dev_compras_valor = 0.0  # fallback (no debería ocurrir)

            compras_netas       = compras_brutas - dev_compras_valor

            # CMV neto: se descuenta explícitamente el costo de la devolución en ventas
            cmv_neto            = cmv_bruto - costo_dev_venta
            utilidad_bruta      = ventas_netas - cmv_neto

            gastos_op           = sum(v for _, v in esc["gastos_operativos"])
            resultado_operativo = utilidad_bruta - gastos_op

            otros_ingresos      = sum(v for _, v in esc["otros_ingresos"])
            otros_egresos       = sum(v for _, v in esc["otros_egresos"])
            utilidad_ai         = resultado_operativo + otros_ingresos - otros_egresos
            impuesto            = max(utilidad_ai, 0) * esc["tasa_impuesto"]
            utilidad_neta       = utilidad_ai - impuesto

            resumen = {
                "ventas_brutas": ventas_brutas,
                "dev_ventas_brutas": dev_ventas_brutas,
                "ventas_netas": ventas_netas,
                "compras_brutas": compras_brutas,
                "dev_compras_valor": dev_compras_valor,
                "compras_netas": compras_netas,
                "cmv_bruto": cmv_bruto,
                "costo_dev_venta": costo_dev_venta,
                "cmv_neto": cmv_neto,
                "utilidad_bruta": utilidad_bruta,
                "gastos_op": gastos_op,
                "resultado_operativo": resultado_operativo,
                "otros_ingresos": otros_ingresos,
                "otros_egresos": otros_egresos,
                "utilidad_ai": utilidad_ai,
                "impuesto": impuesto,
                "utilidad_neta": utilidad_neta,
            }
            return rows, resumen

        # =========================
        # UI: Selección de método
        # =========================
        metodo = st.selectbox(
            "Método de valoración para el ejemplo",
            ["Promedio Ponderado", "PEPS (FIFO)", "UEPS (LIFO)"],
            index=0,
            key="n4_demo_metodo"
        )

        kardex_rows, pyg = build_kardex_and_metrics(metodo)

        # =========================
        # HTML con panel + KARDEX + PYG + NARRACIÓN TTS (con PAUSA)
        # =========================
        html_template = """
        <style>
        .tbl { border-collapse: collapse; width: 100%; font-size:14px; margin:8px 0 }
        .tbl th, .tbl td { border:1px solid #eaeaea; padding:6px 8px; text-align:center }
        .tbl thead th { background:#f8fafc; font-weight:600 }
        .muted { color:#999 }
        .hi { background:#fff7e6; transition: background .3s }
        .controls { display:flex; gap:8px; align-items:center; margin:6px 0; flex-wrap: wrap }
        .btn { padding:6px 10px; border:1px solid #ddd; background:#fafafa; cursor:pointer; border-radius:6px; }
        .btn:hover { background:#f0f0f0 }
        .badge { display:inline-block; background:#eef; border:1px solid #dde; padding:2px 8px; border-radius:12px; font-size:12px; }
        #narr { margin:6px 0 2px 0; font-size:15px }
        .chips { display:grid; grid-template-columns: repeat(3, minmax(0,1fr)); gap:8px; margin:10px 0 6px 0 }
        .chip { background:#f7fafc; border:1px solid #e5e7eb; padding:8px 10px; border-radius:10px; text-align:left; font-size:13px }
        .chip b { display:block; font-size:12px; color:#555; margin-bottom:2px }
        .head { display:flex; align-items:center; justify-content:space-between; margin-top:8px }
        .ratewrap { display:flex; align-items:center; gap:8px; font-size:12px; color:#666 }
        </style>

        <div class="controls">
        <button id="play" class="btn">▶️ Reproducir</button>
        <button id="pause" class="btn">⏸️ Pausa</button>
        <button id="reset" class="btn">⏹️ Detener/Reiniciar</button>
        <span class="badge">__METODO__</span>
        <span class="ratewrap">
            Velocidad voz <input id="rate" type="range" min="0.7" max="1.3" step="0.1" value="1">
        </span>
        </div>

        <div class="head">
        <h4 style="margin:0">🧾 Datos de entrada del Estado de Resultados</h4>
        <small class="muted">Estos valores se editan arriba y alimentan la narración paso a paso</small>
        </div>
        <div class="chips" id="chips"></div>

        <h4>🧮 KARDEX (D1–D5)</h4>
        <table class="tbl" id="kdx">
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

        <h4>📑 Estado de Resultados (en blanco → se completará paso a paso)</h4>
        <table class="tbl" id="pyg">
        <thead>
            <tr><th>Rubro</th><th>Valor</th></tr>
        </thead>
        <tbody>
            <tr><td>Ventas brutas</td><td id="pyg_vb" class="muted"></td></tr>
            <tr><td>(-) Devoluciones en ventas</td><td id="pyg_dv" class="muted"></td></tr>
            <tr><td><b>Ventas netas</b></td><td id="pyg_vn" class="muted"></td></tr>

            <tr><td>Costos de mercancía vendida brutos</td><td id="pyg_cmvb" class="muted"></td></tr>
            <tr><td>(-) Costo devolución en ventas</td><td id="pyg_cdv" class="muted"></td></tr>
            <tr><td><b>Costos de mercancía vendida netos</b></td><td id="pyg_cmvn" class="muted"></td></tr>

            <tr><td><b>Utilidad bruta</b></td><td id="pyg_ub" class="muted"></td></tr>
            <tr><td>Gastos operativos</td><td id="pyg_go" class="muted"></td></tr>
            <tr><td><b>Resultado operativo</b></td><td id="pyg_ro" class="muted"></td></tr>

            <tr><td>Otros ingresos</td><td id="pyg_oi" class="muted"></td></tr>
            <tr><td>Otros egresos</td><td id="pyg_oe" class="muted"></td></tr>
            <tr><td><b>Utilidad antes de impuesto</b></td><td id="pyg_uai" class="muted"></td></tr>
            <tr><td>Impuesto</td><td id="pyg_imp" class="muted"></td></tr>
            <tr><td><b>Utilidad neta</b></td><td id="pyg_un" class="muted"></td></tr>
        </tbody>
        </table>

        <script>
        (function(){
        const rows = __ROWS__;
        const pyg = __PYG__;
        const params = __PARAMS__;
        const narr = document.getElementById("narr");
        const tb = document.getElementById("kbody");
        const btnPlay = document.getElementById("play");
        const btnPause = document.getElementById("pause");
        const btnReset = document.getElementById("reset");
        const chips = document.getElementById("chips");
        const rateCtl = document.getElementById("rate");
        const synth = window.speechSynthesis;

        function pesos(v){
            try{ return new Intl.NumberFormat('es-CO',{style:'currency',currency:'COP'}).format(v); }
            catch(e){ return '$'+(Math.round(v*100)/100).toLocaleString('es-CO'); }
        }
        function fmt(x){
            if(x===null||x===undefined||x==="") return "";
            if(typeof x==="number") return Number.isInteger(x)? x.toString() : (Math.round(x*100)/100).toString().replace(".",",");
            return x;
        }

        function buildChips(){
            const items = [
            ["Unidades vendidas", params.venta_u],
            ["Precio de venta", pesos(params.p_venta)],
            ["Devolución en ventas (u)", params.dev_vent],
            ["Gastos operativos (Σ)", pesos(params.gastos_op)],
            ["Otros ingresos", pesos(params.otros_ingresos)],
            ["Otros egresos", pesos(params.otros_egresos)],
            ["Tasa de impuesto", (params.tasa*100).toFixed(0) + "%"]
            ];
            chips.innerHTML = "";
            items.forEach(([k,v])=>{
            const div = document.createElement("div");
            div.className = "chip";
            div.innerHTML = "<b>"+k+"</b>"+v;
            chips.appendChild(div);
            });
        }

        function buildKardex(){
            tb.innerHTML = "";
            rows.forEach((r,i)=>{
            const tr = document.createElement("tr");
            tr.id = "r"+i;
            tr.innerHTML = `
                <td>${r.Fecha}</td><td>${r.Descripción}</td>
                <td>${fmt(r.Entrada_cant)}</td><td>${(r.Entrada_pu==null||r.Entrada_pu==="")?"":pesos(r.Entrada_pu)}</td><td>${(r.Entrada_total==null||r.Entrada_total==="")?"":pesos(r.Entrada_total)}</td>
                <td>${fmt(r.Salida_cant)}</td><td>${(r.Salida_pu==null||r.Salida_pu==="")?"":pesos(r.Salida_pu)}</td><td>${(r.Salida_total==null||r.Salida_total==="")?"":pesos(r.Salida_total)}</td>
                <td>${fmt(r.Saldo_cant)}</td><td>${(r.Saldo_pu==null||r.Saldo_pu==="")?"":pesos(r.Saldo_pu)}</td><td>${(r.Saldo_total==null||r.Saldo_total==="")?"":pesos(r.Saldo_total)}</td>
            `;
            tb.appendChild(tr);
            });
        }

        function clearPYG(){
            ["vb","dv","vn","cmvb","cdv","cmvn","ub","go","ro","oi","oe","uai","imp","un"].forEach(id=>{
            const el = document.getElementById("pyg_"+id);
            el.textContent = ""; el.classList.add("muted");
            });
            narr.textContent = "";
        }

        function fill(id,val){
            const el = document.getElementById("pyg_"+id);
            if(!el) return;
            el.classList.remove("muted");
            el.textContent = pesos(val);
            el.parentElement.classList.add("hi");
            setTimeout(()=>el.parentElement.classList.remove("hi"),300);
        }

        // --- Estado para pausa / reanudar ---
        let voices = [];
        let isRunning = false;
        let isPaused = false;
        let shouldStop = false;

        function pickSpanishVoice(){
            const prefer = v => v.lang && /^es(-CO)?/i.test(v.lang);
            const prefer2 = v => v.lang && /^es(-MX|-ES)?/i.test(v.lang);
            return voices.find(prefer) || voices.find(prefer2) || voices.find(v=>/^es/i.test(v.lang)) || null;
        }

        function speak(text, {rate=1}={}){
            return new Promise((resolve)=>{
            if(!('speechSynthesis' in window)){
                console.warn("SpeechSynthesis no soportado");
                resolve(); return;
            }
            const u = new SpeechSynthesisUtterance(text);
            const v = pickSpanishVoice();
            if(v) u.voice = v;
            u.lang = (v && v.lang) ? v.lang : "es-ES";
            u.rate = rate;
            u.onend = ()=>resolve();
            u.onerror = ()=>resolve();
            synth.speak(u);
            });
        }

        function loadVoices(){
            voices = synth.getVoices() || [];
        }
        loadVoices();
        if(typeof speechSynthesis !== "undefined"){
            speechSynthesis.onvoiceschanged = loadVoices;
        }

        function stopTTS(){
            try{ synth.cancel(); }catch(e){}
        }

        async function waitWhilePaused(){
            while(isPaused && !shouldStop){
            await new Promise(res=>setTimeout(res,150));
            }
        }

        function stepTexts(p){
            const vb = pesos(pyg.ventas_brutas);
            const dv = pesos(pyg.dev_ventas_brutas);
            const vn = pesos(pyg.ventas_netas);
            const cmvb = pesos(pyg.cmv_bruto);
            const cdv = pesos(pyg.costo_dev_venta);
            const cmvn = pesos(pyg.cmv_neto);
            const ub = pesos(pyg.utilidad_bruta);
            const go = pesos(pyg.gastos_op);
            const ro = pesos(pyg.resultado_operativo);
            const oi = pesos(pyg.otros_ingresos);
            const oe = pesos(pyg.otros_egresos);
            const uai = pesos(pyg.utilidad_ai);
            const imp = pesos(pyg.impuesto);
            const un = pesos(pyg.utilidad_neta);

            const u = p.venta_u;
            const pu = pesos(p.p_venta);
            const dvu = p.dev_vent;
            const tasa = Math.round(p.tasa*100);

            return [
            ["vb", pyg.ventas_brutas, `Iniciamos con las ventas brutas. Tomamos ${u} unidades vendidas y las multiplicamos por el precio de venta ${pu}. El resultado es ${vb}.`],
            ["dv", pyg.dev_ventas_brutas, `A continuación, restamos las devoluciones en ventas. Volvieron ${dvu} unidades, valorizadas al mismo precio de venta ${pu}. Esto equivale a ${dv}.`],
            ["vn", pyg.ventas_netas, `Las ventas netas resultan de ventas brutas menos devoluciones en ventas. Obtenemos ${vn}.`],

            ["cmvb", pyg.cmv_bruto, `Ahora pasamos a los costos de mercancía vendida brutos. Este valor proviene directamente del KARDEX según el método de inventario aplicado. Su valor es ${cmvb}.`],
            ["cdv", pyg.costo_dev_venta, `Luego reconocemos el costo de las unidades devueltas por los clientes. Ese costo se resta de los costos de mercancía vendida brutos y asciende a ${cdv}.`],
            ["cmvn", pyg.cmv_neto, `Los costos de mercancía vendida netos resultan de restar el costo de las devoluciones en ventas a los costos de mercancía vendida brutos. Obtenemos un valor neto de ${cmvn}.`],

            ["ub", pyg.utilidad_bruta, `La utilidad bruta es ventas netas menos los costos de mercancía vendida netos. Esto nos da ${ub}.`],
            ["go", pyg.gastos_op, `Luego restamos los gastos operativos parametrizados. En total suman ${go}.`],
            ["ro", pyg.resultado_operativo, `El resultado operativo es la utilidad bruta menos los gastos operativos. Obtenemos ${ro}.`],

            ["oi", pyg.otros_ingresos, `Sumamos otros ingresos por ${oi}.`],
            ["oe", pyg.otros_egresos, `Restamos otros egresos por ${oe}.`],
            ["uai", pyg.utilidad_ai, `Llegamos a la utilidad antes de impuesto, que asciende a ${uai}.`],
            ["imp", pyg.impuesto, `Aplicamos la tasa de impuesto del ${tasa} por ciento. El impuesto calculado es ${imp}.`],
            ["un", pyg.utilidad_neta, `Finalmente, la utilidad neta del período es ${un}.`]
            ];
        }

        async function run(){
            // Si ya está corriendo y está en pausa → reanudar
            if (isRunning && isPaused){
            isPaused = false;
            btnPause.textContent = "⏸️ Pausa";
            try{ synth.resume(); }catch(e){}
            return;
            }
            // Si está corriendo y no está en pausa → ignorar
            if (isRunning) return;

            // Inicio de una nueva narración
            shouldStop = false;
            isPaused = false;
            isRunning = true;
            btnPause.textContent = "⏸️ Pausa";

            stopTTS();
            clearPYG();
            buildChips();
            const rate = parseFloat(rateCtl.value || "1");
            const steps = stepTexts(params);

            for(const [id,val,txt] of steps){
            if (shouldStop) break;
            await waitWhilePaused();
            if (shouldStop) break;

            narr.innerHTML = txt;
            fill(id, val);
            const pVoice = speak(txt, {rate});
            await pVoice;
            if (shouldStop) break;
            await waitWhilePaused();
            if (shouldStop) break;
            await new Promise(r=>setTimeout(r,180));
            }

            if (!shouldStop){
            narr.innerHTML = "✅ Estado de Resultados completado a partir de los datos parametrizados y el KARDEX.";
            }

            isRunning = false;
            isPaused = false;
            btnPause.textContent = "⏸️ Pausa";
        }

        buildChips();
        buildKardex();

        // Botón Reproducir / Reanudar
        btnPlay.onclick = run;

        // Botón Pausa / Reanudar
        btnPause.onclick = ()=>{
            if (!isRunning) return;
            if (!isPaused){
            isPaused = true;
            btnPause.textContent = "▶️ Reanudar";
            try{ synth.pause(); }catch(e){}
            } else {
            isPaused = false;
            btnPause.textContent = "⏸️ Pausa";
            try{ synth.resume(); }catch(e){}
            }
        };

        // Botón Reset (detener todo y limpiar)
        btnReset.onclick = ()=>{
            shouldStop = true;
            isPaused = false;
            isRunning = false;
            btnPause.textContent = "⏸️ Pausa";
            stopTTS();
            buildChips(); buildKardex(); clearPYG();
        };
        })();
        </script>
        """

        # Adaptar filas del KARDEX a la estructura HTML
        def _as_row(r):
            return {
                "Fecha": r["Fecha"],
                "Descripción": r["Descripción"],
                "Entrada_cant": r.get("Entrada_cant",""),
                "Entrada_pu": r.get("Entrada_pu",""),
                "Entrada_total": r.get("Entrada_total",""),
                "Salida_cant": r.get("Salida_cant",""),
                "Salida_pu": r.get("Salida_pu",""),
                "Salida_total": r.get("Salida_total",""),
                "Saldo_cant": r.get("Saldo_cant",""),
                "Saldo_pu": r.get("Saldo_pu",""),
                "Saldo_total": r.get("Saldo_total",""),
            }

        # Serializar y reemplazar placeholders
        rows_json = _json.dumps([_as_row(r) for r in kardex_rows], ensure_ascii=False)
        pyg_json  = _json.dumps(pyg, ensure_ascii=False)
        params_json = _json.dumps({
            "venta_u": esc["venta_u"],
            "p_venta": esc["p_venta"],
            "dev_vent": esc["dev_vent"],
            "gastos_op": sum(v for _, v in esc["gastos_operativos"]),
            "otros_ingresos": sum(v for _, v in esc["otros_ingresos"]),
            "otros_egresos": sum(v for _, v in esc["otros_egresos"]),

            "tasa": esc["tasa_impuesto"],
        }, ensure_ascii=False)

        html = (
            html_template
            .replace("__METODO__", metodo)
            .replace("__ROWS__", rows_json)
            .replace("__PYG__",  pyg_json)
            .replace("__PARAMS__", params_json)
        )

        # Render principal
        components.html(html, height=860, scrolling=True)

    # =====================================================
    # TAB 3 — PRÁCTICA IA
    # =====================================================
    with tabs[2]:
        st.subheader("Práctica IA: Estado de Resultados (Nivel 4)")
        st.caption("Genera un escenario, observa el KARDEX de referencia y completa el Estado de Resultados. Valida y recibe retroalimentación.")

        # ========= Prefijo de claves (namespacing) =========
        KP = "lvl4_"
        K = lambda name: f"{KP}{name}"

        # =========================
        # Estado y escenario (defaults + aleatorio)
        # =========================
        def _n4_ensure_default_state():
            ss = st.session_state
            ss.setdefault(K("metodo"), "Promedio Ponderado")
            ss.setdefault(K("inv0_u"), 80)
            ss.setdefault(K("inv0_pu"), 10.0)
            ss.setdefault(K("comp1_u"), 40)
            ss.setdefault(K("comp1_pu"), 11.0)
            ss.setdefault(K("venta_u"), 90)
            ss.setdefault(K("p_venta"), 20.0)
            ss.setdefault(K("dev_comp_u"), 8)
            ss.setdefault(K("dev_vent_u"), 6)
            ss.setdefault(K("go_1_name"), "Publicidad y promociones")
            ss.setdefault(K("go_1_val"), 120.0)
            ss.setdefault(K("go_2_name"), "Servicios (luz/agua/internet)")
            ss.setdefault(K("go_2_val"), 80.0)
            ss.setdefault(K("go_3_name"), "Gastos administrativos")
            ss.setdefault(K("go_3_val"), 150.0)
            ss.setdefault(K("otros_ing"), 40.0)
            ss.setdefault(K("otros_egr"), 20.0)
            ss.setdefault(K("tasa"), 0.30)
            # método para visualizar KARDEX de referencia
            ss.setdefault(K("kdx_view_metodo"), ss[K("metodo")])

        def _n4_randomize_scenario():
            import random
            inv0_u  = random.choice([60, 80, 100, 120, 150])
            inv0_pu = random.choice([8.0, 9.0, 10.0, 11.0, 12.0])
            comp1_u = random.choice([30, 40, 50, 60, 70])
            comp1_pu= random.choice([inv0_pu - 1, inv0_pu, inv0_pu + 1, inv0_pu + 2])
            venta_u = random.choice([40, 60, 90, 110, 130])
            p_venta = random.choice([16.0, 18.0, 20.0, 22.0, 24.0])
            dev_comp_u  = max(0, min(comp1_u, random.choice([4, 6, 8, 10, 12, 15])))
            dev_vent_u  = max(0, min(venta_u, random.choice([2, 4, 6, 8, 10, 12])))
            go_vals = sorted(random.sample([80.0, 90.0, 100.0, 120.0, 140.0, 150.0], 3))
            otros_ing = random.choice([20.0, 30.0, 40.0, 50.0])
            otros_egr = random.choice([10.0, 15.0, 20.0, 25.0])
            tasa = random.choice([0.19, 0.25, 0.30])

            ss = st.session_state
            ss[K("inv0_u")]  = inv0_u
            ss[K("inv0_pu")] = float(max(1.0, round(inv0_pu, 2)))
            ss[K("comp1_u")] = comp1_u
            ss[K("comp1_pu")] = float(max(1.0, round(comp1_pu, 2)))
            ss[K("venta_u")] = venta_u
            ss[K("p_venta")] = float(p_venta)
            ss[K("dev_comp_u")]  = dev_comp_u
            ss[K("dev_vent_u")]  = dev_vent_u
            ss[K("go_1_name")], ss[K("go_2_name")], ss[K("go_3_name")] = "Gasto A", "Gasto B", "Gasto C"
            ss[K("go_1_val")], ss[K("go_2_val")], ss[K("go_3_val")] = go_vals
            ss[K("otros_ing")] = otros_ing
            ss[K("otros_egr")] = otros_egr
            ss[K("tasa")] = float(tasa)

        def _n4_request_random():
            st.session_state[K("rand_req")] = True

        _n4_ensure_default_state()

        # ====== CONTROLES SUPERIORES: Método + Aleatorio (ANTES del KARDEX) ======
        ctop1, ctop2 = st.columns([1.3, 1])
        with ctop1:
            st.selectbox(
                "Método de valoración (afecta CMV y devoluciones de venta)",
                ["Promedio Ponderado", "PEPS (FIFO)", "UEPS (LIFO)"],
                key=K("metodo")
            )
        with ctop2:
            st.button("🎲 Generar escenario aleatorio", on_click=_n4_request_random, key=K("rand_btn"))

        if st.session_state.get(K("rand_req"), False):
            _n4_randomize_scenario()
            st.session_state.pop(K("rand_req"), None)
            st.rerun()

        # =========================
        # Escenario visible (inputs)
        # =========================
        st.markdown("#### 🎯 Escenario del ejercicio")
        colA, colB, colC = st.columns(3)
        with colA:
            st.number_input("Día 1: inventario inicial (u)", min_value=0, step=1, key=K("inv0_u"))
            st.number_input("Día 1: costo unitario inicial", min_value=0.0, step=0.1, key=K("inv0_pu"))
        with colB:
            st.number_input("Día 2: compra (u)", min_value=0, step=1, key=K("comp1_u"))
            st.number_input("Día 2: costo unitario compra", min_value=0.0, step=0.1, key=K("comp1_pu"))
        with colC:
            st.number_input("Día 3: venta (u)", min_value=0, step=1, key=K("venta_u"))
            st.number_input("Precio de venta ($/u)", min_value=0.0, step=0.5, key=K("p_venta"))

        colD, colE, colF = st.columns(3)
        with colD:
            st.number_input("Día 4: devolución en compra (u)", min_value=0, step=1, key=K("dev_comp_u"))
        with colE:
            st.number_input("Día 5: devolución en venta (u)", min_value=0, step=1, key=K("dev_vent_u"))
        with colF:
            st.slider("Tasa de impuesto (%)", min_value=0, max_value=50,
                    value=int(st.session_state[K("tasa")]*100), step=1, key=K("tasa_pct"))
        st.session_state[K("tasa")] = float(st.session_state[K("tasa_pct")]) / 100.0

        st.markdown("##### Gastos operativos (edita los tres ítems)")
        g1, g2, g3 = st.columns(3)
        with g1:
            st.text_input("Ítem 1", key=K("go_1_name"))
            st.number_input("Valor 1", min_value=0.0, step=10.0, key=K("go_1_val"))
        with g2:
            st.text_input("Ítem 2", key=K("go_2_name"))
            st.number_input("Valor 2", min_value=0.0, step=10.0, key=K("go_2_val"))
        with g3:
            st.text_input("Ítem 3", key=K("go_3_name"))
            st.number_input("Valor 3", min_value=0.0, step=10.0, key=K("go_3_val"))

        o1, o2 = st.columns(2)
        with o1:
            st.number_input("Otros ingresos (total)", min_value=0.0, step=10.0, key=K("otros_ing"))
        with o2:
            st.number_input("Otros egresos (total)", min_value=0.0, step=10.0, key=K("otros_egr"))

        # =========================
        # Helpers comunes (inventario)
        # =========================
        def _sum_layers(layers):
            q = sum(q for q, _ in layers)
            v = sum(q * p for q, p in layers)
            pu = (v / q) if q > 0 else 0.0
            return q, pu, v

        def _consume_layers_detail(layers, qty_out, fifo=True):
            order = layers[:] if fifo else layers[::-1]
            remaining = qty_out
            sale_details = []
            new_layers = []
            for q, pu in order:
                if remaining <= 0:
                    new_layers.append([q, pu]); continue
                take = min(q, remaining)
                if take > 0:
                    sale_details.append((take, pu, take * pu))
                    rest = q - take
                    remaining -= take
                    if rest > 0:
                        new_layers.append([rest, pu])
            final_layers = new_layers if fifo else new_layers[::-1]
            return sale_details, final_layers

        # =========================
        # KARDEX de referencia (según método)
        # =========================
        def _build_kardex_expected(method_name):
            ss = st.session_state
            inv0_u, inv0_pu = ss[K("inv0_u")], ss[K("inv0_pu")]
            c1_u, c1_pu     = ss[K("comp1_u")], ss[K("comp1_pu")]
            v_u             = ss[K("venta_u")]
            dcomp_u         = ss[K("dev_comp_u")]
            dvent_u         = ss[K("dev_vent_u")]

            rows = []
            layers = [[float(inv0_u), float(inv0_pu)]] if inv0_u > 0 else []
            s_q, s_p, s_v = _sum_layers(layers)
            rows.append({"Fecha":"Día 1","Descripción":"Saldo inicial",
                        "Entrada_cant":None, "Entrada_pu":None, "Entrada_total":None,
                        "Salida_cant":None, "Salida_pu":None, "Salida_total":None,
                        "Saldo_cant": s_q, "Saldo_pu": round(s_p,2), "Saldo_total": round(s_v,2)})

            if method_name == "Promedio Ponderado":
                ent_tot = c1_u * c1_pu
                q_new = s_q + c1_u
                v_new = s_v + ent_tot
                p_new = (v_new / q_new) if q_new > 0 else 0.0
                layers = [[q_new, p_new]]
                s_q, s_p, s_v = _sum_layers(layers)
                rows.append({"Fecha":"Día 2","Descripción":"Compra",
                            "Entrada_cant": c1_u, "Entrada_pu": round(c1_pu,2), "Entrada_total": round(ent_tot,2),
                            "Salida_cant":None, "Salida_pu":None, "Salida_total":None,
                            "Saldo_cant": s_q, "Saldo_pu": round(s_p,2), "Saldo_total": round(s_v,2)})
            else:
                rows.append({"Fecha":"Día 2","Descripción":"Saldo (día 1)",
                            "Entrada_cant":None, "Entrada_pu":None, "Entrada_total":None,
                            "Salida_cant":None, "Salida_pu":None, "Salida_total":None,
                            "Saldo_cant": s_q, "Saldo_pu": round(s_p,2), "Saldo_total": round(s_v,2)})
                ent_tot = c1_u * c1_pu
                layers.append([float(c1_u), float(c1_pu)])
                rows.append({"Fecha":"Día 2","Descripción":"Compra",
                            "Entrada_cant": c1_u, "Entrada_pu": round(c1_pu,2), "Entrada_total": round(ent_tot,2),
                            "Salida_cant":None, "Salida_pu":None, "Salida_total":None,
                            "Saldo_cant": c1_u, "Saldo_pu": round(c1_pu,2), "Saldo_total": round(ent_tot,2)})
                s_q, s_p, s_v = _sum_layers(layers)

            if v_u > 0 and s_q > 0:
                if method_name == "Promedio Ponderado":
                    sale_q  = min(v_u, int(s_q))
                    sale_pu = layers[0][1] if layers else 0.0
                    sale_tot= sale_q * sale_pu
                    q2 = s_q - sale_q
                    v2 = s_v - sale_tot
                    p2 = (v2/q2) if q2 > 0 else 0.0
                    layers = [[q2, p2]] if q2 > 0 else []
                    s_q, s_p, s_v = _sum_layers(layers)
                    rows.append({"Fecha":"Día 3","Descripción":"Venta",
                                "Entrada_cant":None, "Entrada_pu":None, "Entrada_total":None,
                                "Salida_cant": sale_q, "Salida_pu": round(sale_pu,2), "Salida_total": round(sale_tot,2),
                                "Saldo_cant": s_q, "Saldo_pu": round(s_p,2), "Saldo_total": round(s_v,2)})
                else:
                    fifo = (method_name == "PEPS (FIFO)")
                    sale_details, layers_after = _consume_layers_detail(layers, v_u, fifo=fifo)
                    running_layers = [l[:] for l in layers]
                    tag = "PEPS" if fifo else "UEPS"
                    for i,(q_take, pu_take, tot_take) in enumerate(sale_details, start=1):
                        _, running_layers = _consume_layers_detail(running_layers, q_take, fifo=fifo)
                        rq, rpu, rv = _sum_layers(running_layers)
                        rows.append({"Fecha":"Día 3","Descripción": f"Venta tramo {i} ({tag})",
                                    "Entrada_cant":None, "Entrada_pu":None, "Entrada_total":None,
                                    "Salida_cant": q_take, "Salida_pu": round(pu_take,2), "Salida_total": round(tot_take,2),
                                    "Saldo_cant": rq, "Saldo_pu": round(rpu,2), "Saldo_total": round(rv,2)})
                    layers = layers_after
                    s_q, s_p, s_v = _sum_layers(layers)
            else:
                rows.append({"Fecha":"Día 3","Descripción":"Venta",
                            "Entrada_cant":None, "Entrada_pu":None, "Entrada_total":None,
                            "Salida_cant":None, "Salida_pu":None, "Salida_total":None,
                            "Saldo_cant": s_q, "Saldo_pu": round(s_p,2), "Saldo_total": round(s_v,2)})

            # D4 devolución compra
            if method_name == "Promedio Ponderado":
                take_q  = min(st.session_state[K("dev_comp_u")], s_q)
                take_pu = s_p
                take_val= take_q * take_pu
                q4 = s_q - take_q
                v4 = s_v - take_val
                p4 = (v4/q4) if q4 > 0 else 0.0
                layers = [[q4, p4]] if q4 > 0 else []
                s_q, s_p, s_v = _sum_layers(layers)
                rows.append({"Fecha":"Día 4","Descripción":"Devolución de compra",
                            "Entrada_cant":None, "Entrada_pu":None, "Entrada_total":None,
                            "Salida_cant": int(take_q), "Salida_pu": round(take_pu,2), "Salida_total": round(take_val,2),
                            "Saldo_cant": int(s_q), "Saldo_pu": round(s_p,2), "Saldo_total": round(s_v,2)})
            else:
                send_back = st.session_state[K("dev_comp_u")]
                rev = layers[::-1]
                new_rev = []
                dev_val = 0.0
                take_q_total = 0
                for q, pu in rev:
                    if send_back <= 0:
                        new_rev.append([q, pu]); continue
                    take = min(q, send_back)
                    take_q_total += take
                    dev_val += take * pu
                    rest = q - take
                    send_back -= take
                    if rest > 0:
                        new_rev.append([rest, pu])
                layers = new_rev[::-1]
                s_q, s_p, s_v = _sum_layers(layers)
                rows.append({"Fecha":"Día 4","Descripción":"Devolución de compra",
                            "Entrada_cant":None, "Entrada_pu":None, "Entrada_total":None,
                            "Salida_cant": int(take_q_total), "Salida_pu":None, "Salida_total": round(dev_val,2),
                            "Saldo_cant": int(s_q), "Saldo_pu": round(s_p,2), "Saldo_total": round(s_v,2)})

            # D5 devolución venta
            if st.session_state[K("dev_vent_u")] > 0:
                dvent = st.session_state[K("dev_vent_u")]
                in_pu = s_p  # referencia para mostrar
                in_val= dvent * in_pu
                layers.append([float(dvent), float(in_pu)])
                s_q, s_p, s_v = _sum_layers(layers)
                rows.append({"Fecha":"Día 5","Descripción":"Devolución de venta (reingreso)",
                            "Entrada_cant": int(dvent), "Entrada_pu": round(in_pu,2), "Entrada_total": round(in_val,2),
                            "Salida_cant":None, "Salida_pu":None, "Salida_total":None,
                            "Saldo_cant": int(s_q), "Saldo_pu": round(s_p,2), "Saldo_total": round(s_v,2)})
            else:
                rows.append({"Fecha":"Día 5","Descripción":"Devolución de venta (reingreso)",
                            "Entrada_cant":None, "Entrada_pu":None, "Entrada_total":None,
                            "Salida_cant":None, "Salida_pu":None, "Salida_total":None,
                            "Saldo_cant": int(s_q), "Saldo_pu": round(s_p,2), "Saldo_total": round(s_v,2)})

            return rows

        # =========================
        # PyG esperado (para el validador del ER)
        # =========================
        def _build_pyg_expected_from_kardex():
            ss = st.session_state
            method_name = ss[K("metodo")]
            rows = _build_kardex_expected(method_name)

            cmvb = 0.0
            cdev = 0.0
            for r in rows:
                desc = r["Descripción"]
                if "Venta" in desc:
                    val = r.get("Salida_total","")
                    cmvb += float(val) if val != "" else 0.0
                if "Devolución de venta" in desc:
                    val_in = r.get("Entrada_total","")
                    cdev += float(val_in) if val_in != "" else 0.0

            ventas_brutas     = ss[K("venta_u")] * ss[K("p_venta")]
            dev_ventas_brutas = ss[K("dev_vent_u")] * ss[K("p_venta")]
            compras_brutas    = ss[K("comp1_u")] * ss[K("comp1_pu")]

            dev_compras_valor = 0.0
            for r in rows:
                if r["Fecha"]=="Día 4" and "Devolución de compra" in r["Descripción"]:
                    val = r.get("Salida_total","")
                    dev_compras_valor = float(val) if val != "" else 0.0
                    break

            ventas_netas  = ventas_brutas - dev_ventas_brutas
            compras_netas = compras_brutas - dev_compras_valor
            cmv_neto      = cmvb - cdev
            utilidad_bruta= ventas_netas - cmv_neto

            go_vals = [ss[K("go_1_val")], ss[K("go_2_val")], ss[K("go_3_val")]]
            gastos_op = sum(go_vals)

            resultado_operativo = utilidad_bruta - gastos_op
            utilidad_ai = resultado_operativo + float(ss[K("otros_ing")]) - float(ss[K("otros_egr")])
            impuesto = max(utilidad_ai, 0.0) * float(ss[K("tasa")])
            utilidad_neta = utilidad_ai - impuesto

            return {
                "Ventas brutas": ventas_brutas,
                "(-) Devoluciones en ventas": dev_ventas_brutas,
                "Ventas netas": ventas_netas,
                "CMV": cmv_neto,
                "Utilidad bruta": utilidad_bruta,
                "Gastos operativos": gastos_op,
                "Resultado operativo": resultado_operativo,
                "Otros ingresos": float(ss[K("otros_ing")]),
                "Otros egresos": float(ss[K("otros_egr")]),
                "Utilidad antes de impuesto": utilidad_ai,
                "Impuesto": impuesto,
                "Utilidad neta": utilidad_neta
            }

        # =========================
        # Mostrar KARDEX de referencia (desplegable por método)
        # =========================
        st.markdown("---")
        st.markdown("### 🧮 KARDEX de referencia")
        st.selectbox(
            "Visualizar KARDEX con método:",
            ["Promedio Ponderado", "PEPS (FIFO)", "UEPS (LIFO)"],
            key=K("kdx_view_metodo")
        )
        df_kdx_ref = pd.DataFrame(_build_kardex_expected(st.session_state[K("kdx_view_metodo")]))
        st.dataframe(df_kdx_ref, use_container_width=True)

        # =========================
        # Estado de Resultados (editor + validador)
        # =========================
        st.markdown("---")
        st.markdown("### 📑 Estado de Resultados — completa los valores")

        pyg_expected = _build_pyg_expected_from_kardex()

        order_rows = [
            "Ventas brutas",
            "(-) Devoluciones en ventas",
            "Ventas netas",
            "CMV",
            "Utilidad bruta",
            "Gastos operativos",
            "Resultado operativo",
            "Otros ingresos",
            "Otros egresos",
            "Utilidad antes de impuesto",
            "Impuesto",
            "Utilidad neta"
        ]
        df_blank_pyg = pd.DataFrame({"Rubro": order_rows, "Valor": [""]*len(order_rows)})

        st.caption("Ingresa los valores numéricos. El validador exige coherencia total con el escenario y el método seleccionado.")
        edited_pyg = st.data_editor(
            df_blank_pyg,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Rubro": st.column_config.TextColumn(disabled=True),
                "Valor": st.column_config.NumberColumn(step=0.01, help="Valor monetario (usa números)"),
            },
            key=K("pyg_student_editor"),
            num_rows="fixed",
            disabled=False
        )

        with st.form(K("pyg_check_form")):
            ask_ai = st.checkbox("💬 Retroalimentación de IA (opcional)", value=False, key=K("ai_cb"))
            submitted = st.form_submit_button("✅ Validar mi Estado de Resultados")

        if submitted:
            tol = 0.5
            def _to_float_or_none(x):
                try:
                    if x in (None,""): return None
                    return float(x)
                except: return None
            def _near(a,b):
                if a is None or b is None: return False
                return abs(a-b) <= tol

            checks = []
            correct_rows = 0
            for i, rubro in enumerate(order_rows):
                usr_val = _to_float_or_none(edited_pyg.iloc[i]["Valor"])
                exp_val = float(pyg_expected[rubro])
                ok = _near(usr_val, exp_val)
                checks.append((rubro, usr_val, exp_val, ok))
                if ok: correct_rows += 1

            st.metric("ER — renglones correctos", f"{correct_rows}/{len(order_rows)}")
            for rubro, usr, exp, ok in checks:
                badge = "✅" if ok else "❌"
                st.write(f"{badge} **{rubro}** — tu valor: {('—' if usr is None else f'{usr:.2f}')} | esperado: {exp:.2f}")

            if correct_rows == len(order_rows):
                st.success("¡Excelente! Tu Estado de Resultados es consistente con el escenario y el método.")
            else:
                st.warning("Hay diferencias. Revisa la secuencia y los vínculos con el KARDEX (ventas, CMV, devoluciones e impuesto).")

            if ask_ai:
                try:
                    intento_txt = "\n".join([f"{r}: {('—' if v is None else f'{v:.2f}')}" for r, v, _, _ in checks])
                    esperado_txt = "\n".join([f"{r}: {float(pyg_expected[r]):.2f}" for r in order_rows])
                    metodo = st.session_state[K("metodo")]
                    tasa_pct = int(st.session_state[K("tasa")]*100)
                    prompt_fb = (
                        "Evalúa el Estado de Resultados diligenciado por el estudiante.\n"
                        f"Método de valoración: {metodo}. Tasa de impuesto: {tasa_pct}%.\n"
                        "Usa como base el KARDEX del período (saldo inicial, compra, venta, devoluciones) y los rubros del período.\n\n"
                        "Valores del estudiante:\n" + intento_txt + "\n\n"
                        "Valores esperados:\n" + esperado_txt + "\n\n"
                        "Indica: (1) errores por renglón, (2) explicación paso a paso, "
                        "(3) tips para no confundir ventas netas, CMV, utilidad bruta y cálculo del impuesto."
                    )
                    with st.spinner("Generando retroalimentación de IA…"):
                        fb_text = ia_feedback(prompt_fb)  # si no existe, caerá al except
                    with st.expander("💬 Retroalimentación de la IA (Estado de Resultados)"):

                        st.write(fb_text)
                except Exception as e:
                    st.info("La retroalimentación de IA no está disponible en este entorno.")
                    st.caption(f"Detalle técnico: {e}")

    # =====================================================
    # TAB 4 — EVALUACIÓN FINAL
    # =====================================================
    with tabs[3]:
        st.subheader("Evaluación final del Nivel 4")
        st.caption("Debes acertar **5 de 5** para aprobar y avanzar.")

        # ---------- Namespacing (CLAVES ÚNICAS) ----------
        KP = "n4_final_"
        K = lambda name: f"{KP}{name}"

        # =====================================================
        # Escenario base (SOLO Promedio Ponderado)
        # =====================================================
        def exam_scenario_q5():
            """
            Este escenario define TODAS las cifras necesarias para:
            - Construir el KARDEX PP de referencia
            - Calcular el Estado de Resultados esperado
            Nota: En la interfaz, SOLO se muestran las variables clave solicitadas.
            """
            return {
                "inv0_u":   90,     # Día 1: inventario inicial (u)
                "inv0_pu":  10.0,   # Día 1: costo unitario
                "comp1_u":  50,     # Día 2: compra (u)
                "comp1_pu": 12.0,   # Día 2: $/u compra
                "venta_u":  100,    # Día 3: cantidad vendida (u)
                "p_venta":  20.0,   # Día 3: precio de venta ($/u) — visible
                "dev_comp": 6,      # Día 4: devolución en compra (u)
                "dev_venta":8,      # Día 5: devolución en venta (u)
                "gastos_operativos": [("Publicidad",120.0), ("Servicios",80.0), ("Administrativos",150.0)],
                "otros_ing": 40.0,  # visible
                "otros_egr": 20.0,  # visible
                "tasa": 0.30,       # visible: 30%
                "metodo": "Promedio Ponderado"
            }

        # =====================================================
        # Helpers: KARDEX & PyG en Promedio Ponderado
        # =====================================================
        def _sum_layers(layers):
            q = sum(q for q, _ in layers)
            v = sum(q * p for q, p in layers)
            pu = (v / q) if q > 0 else 0.0
            return q, pu, v

        def _kardex_rows_pp(sc: dict):
            """
            Construye el KARDEX D1–D5 (SOLO PP) para mostrarlo como referencia.
            """
            inv0_u, inv0_pu = sc["inv0_u"], sc["inv0_pu"]
            c1_u, c1_pu     = sc["comp1_u"], sc["comp1_pu"]
            v_u             = sc["venta_u"]
            dcomp_u         = sc["dev_comp"]
            dvent_u         = sc["dev_venta"]

            rows = []

            # D1 saldo inicial
            layers = [[float(inv0_u), float(inv0_pu)]]
            s_q, s_p, s_v = _sum_layers(layers)
            rows.append({
                "Fecha":"Día 1","Descripción":"Saldo inicial",
                "Entrada_cant":None, "Entrada_pu":None, "Entrada_total":None,
                "Salida_cant":None,  "Salida_pu":None,  "Salida_total":None,
                "Saldo_cant": int(s_q), "Saldo_pu": round(s_p,2), "Saldo_total": round(s_v,2)
            })

            # D2 compra (promediada)
            ent_tot = c1_u * c1_pu
            q2 = s_q + c1_u
            v2 = s_v + ent_tot
            p2 = (v2 / q2) if q2 > 0 else 0.0
            layers = [[q2, p2]]
            s_q, s_p, s_v = _sum_layers(layers)
            rows.append({
                "Fecha":"Día 2","Descripción":"Compra",
                "Entrada_cant": int(c1_u), "Entrada_pu": round(c1_pu,2), "Entrada_total": round(ent_tot,2),
                "Salida_cant":None, "Salida_pu":None, "Salida_total":None,
                "Saldo_cant": int(s_q), "Saldo_pu": round(s_p,2), "Saldo_total": round(s_v,2)
            })

            # D3 venta (al promedio vigente)
            sale_q  = min(v_u, s_q)
            sale_pu = layers[0][1] if layers else 0.0
            sale_tot= sale_q * sale_pu
            q3 = s_q - sale_q
            v3 = s_v - sale_tot
            p3 = (v3 / q3) if q3 > 0 else 0.0
            layers = [[q3, p3]] if q3 > 0 else []
            s_q, s_p, s_v = _sum_layers(layers)
            rows.append({
                "Fecha":"Día 3","Descripción":"Venta",
                "Entrada_cant":None, "Entrada_pu":None, "Entrada_total":None,
                "Salida_cant": int(sale_q), "Salida_pu": round(sale_pu,2), "Salida_total": round(sale_tot,2),
                "Saldo_cant": int(s_q), "Saldo_pu": round(s_p,2), "Saldo_total": round(s_v,2)
            })

            # D4 devolución compra (sale al promedio vigente)
            take_q  = min(dcomp_u, s_q)
            take_pu = s_p
            take_val= take_q * take_pu
            q4 = s_q - take_q
            v4 = s_v - take_val
            p4 = (v4 / q4) if q4 > 0 else 0.0
            layers = [[q4, p4]] if q4 > 0 else []
            s_q, s_p, s_v = _sum_layers(layers)
            rows.append({
                "Fecha":"Día 4","Descripción":"Devolución de compra",
                "Entrada_cant":None, "Entrada_pu":None, "Entrada_total":None,
                "Salida_cant": int(take_q), "Salida_pu": round(take_pu,2), "Salida_total": round(take_val,2),
                "Saldo_cant": int(s_q), "Saldo_pu": round(s_p,2), "Saldo_total": round(s_v,2)
            })

            # D5 devolución venta (reingresa al promedio vigente)
            in_q  = dvent_u
            in_pu = s_p
            in_val= in_q * in_pu
            q5 = s_q + in_q
            v5 = s_v + in_val
            p5 = (v5 / q5) if q5 > 0 else 0.0
            layers = [[q5, p5]]
            s_q, s_p, s_v = _sum_layers(layers)
            rows.append({
                "Fecha":"Día 5","Descripción":"Devolución de venta (reingreso)",
                "Entrada_cant": int(in_q), "Entrada_pu": round(in_pu,2), "Entrada_total": round(in_val,2),
                "Salida_cant":None, "Salida_pu":None, "Salida_total":None,
                "Saldo_cant": int(s_q), "Saldo_pu": round(s_p,2), "Saldo_total": round(s_v,2)
            })

            return rows

        def _pyg_expected_from_scenario_pp(sc: dict):
            """
            Calcula el Estado de Resultados esperado (SOLO PP) a partir del escenario.
            Devuelve un dict con el orden EXACTO de los renglones que mostrará el editor.
            """
            inv0_u, inv0_pu = sc["inv0_u"], sc["inv0_pu"]
            c1_u, c1_pu     = sc["comp1_u"], sc["comp1_pu"]
            v_u, p_venta    = sc["venta_u"], sc["p_venta"]
            dcomp_u         = sc["dev_comp"]
            dvent_u         = sc["dev_venta"]
            go_vals         = [v for _, v in sc["gastos_operativos"]]
            otros_ing       = float(sc["otros_ing"])
            otros_egr       = float(sc["otros_egr"])
            tasa            = float(sc["tasa"])

            # D1
            layers = [[float(inv0_u), float(inv0_pu)]]
            s_q, s_p, s_v = _sum_layers(layers)

            # D2
            ent_tot = c1_u * c1_pu
            q2 = s_q + c1_u
            v2 = s_v + ent_tot
            p2 = (v2 / q2) if q2 > 0 else 0.0
            layers = [[q2, p2]]
            s_q, s_p, s_v = _sum_layers(layers)

            # D3 venta
            sale_q  = min(v_u, s_q)
            sale_pu = layers[0][1] if layers else 0.0
            sale_tot= sale_q * sale_pu
            q3 = s_q - sale_q
            v3 = s_v - sale_tot
            p3 = (v3 / q3) if q3 > 0 else 0.0
            layers = [[q3, p3]] if q3 > 0 else []
            s_q, s_p, s_v = _sum_layers(layers)
            cmv_bruto = sale_tot

            # D4 devolución compra (reduce pool al promedio vigente)
            take_q  = min(dcomp_u, s_q)
            take_pu = s_p
            take_val= take_q * take_pu
            q4 = s_q - take_q
            v4 = s_v - take_val
            p4 = (v4 / q4) if q4 > 0 else 0.0
            layers = [[q4, p4]] if q4 > 0 else []
            s_q, s_p, s_v = _sum_layers(layers)
            dev_compras_valor = take_val

            # D5 devolución venta (reingresa al promedio vigente)
            in_q  = dvent_u
            in_pu = s_p
            in_val= in_q * in_pu
            q5 = s_q + in_q
            v5 = s_v + in_val
            p5 = (v5 / q5) if q5 > 0 else 0.0
            layers = [[q5, p5]]
            s_q, s_p, s_v = _sum_layers(layers)
            costo_dev_venta = in_val

            # PyG
            ventas_brutas       = v_u * p_venta
            dev_ventas_brutas   = dvent_u * p_venta
            ventas_netas        = ventas_brutas - dev_ventas_brutas

            compras_brutas      = c1_u * c1_pu
            compras_netas       = compras_brutas - dev_compras_valor

            cmv_neto            = cmv_bruto - costo_dev_venta
            utilidad_bruta      = ventas_netas - cmv_neto

            gastos_op           = sum(go_vals)
            resultado_operativo = utilidad_bruta - gastos_op

            utilidad_ai         = resultado_operativo + otros_ing - otros_egr
            impuesto            = max(utilidad_ai, 0.0) * tasa
            utilidad_neta       = utilidad_ai - impuesto

            # Orden EXACTO que verá el editor
            return {
                "Ventas brutas": ventas_brutas,
                "(-) Devoluciones en ventas": dev_ventas_brutas,
                "Ventas netas": ventas_netas,
                "CMV": cmv_neto,
                "Utilidad bruta": utilidad_bruta,
                "Gastos operativos": gastos_op,
                "Resultado operativo": resultado_operativo,
                "Otros ingresos": otros_ing,
                "Otros egresos": otros_egr,
                "Utilidad antes de impuesto": utilidad_ai,
                "Impuesto": impuesto,
                "Utilidad neta": utilidad_neta
            }

        # =====================================================
        # IA Helpers (opcionales y seguros)
        # =====================================================
        def _on_topic_fallback_open() -> str:
            return (
                "En sistema perpetuo, el **CMV** se determina desde el **KARDEX** según el método (PP/PEPS/UEPS), "
                "incluyendo devoluciones: las **de compras** reducen el pool de costo; las **de ventas** reingresan "
                "unidades y reducen el **CMV** presentado. Registrar de forma incoherente distorsiona **Utilidad Bruta**, "
                "**Resultado Operativo** y **Utilidad Neta**, afectando comparabilidad y decisiones."
            )

        def _sanitize_on_topic(text: str) -> str:
            if not text:
                return _on_topic_fallback_open()
            banned = [
                "debe", "haber", "asiento", "apertura", "cierre", "diario",
                "iva repercutido", "iva soportado", "iva", "cuentas t",
                "resultado del ejercicio", "balance de comprobación",
                "pasivo", "patrimonio", "ecuación contable", "activo = pasivo + patrimonio"
            ]
            low = text.lower()
            if any(b in low for b in banned):
                return _on_topic_fallback_open()
            return text

        def safe_ia_feedback(prompt: str, default: str = "", tries: int = 3, base_sleep: float = 0.8) -> str:
            st.session_state.pop(K("ai_rate_limited"), None)
            for t in range(tries):
                try:
                    resp = ia_feedback(prompt)  # noqa: F821 (si no existe, cae al except)
                    if resp is None:
                        return default
                    if isinstance(resp, dict):
                        return str(resp.get("text", default))
                    return str(resp)
                except Exception as e:
                    msg = str(e)
                    if "429" in msg or "rate-limit" in msg.lower() or "rate limit" in msg.lower():
                        st.session_state[K("ai_rate_limited")] = True
                        return default
                    time.sleep(min(base_sleep * (2 ** t), 4.0))
                    continue
            return default

        # =====================================================
        # FORM — 2 MCQ + 2 abiertas + 1 ejercicio (tipo práctica)
        # =====================================================
        with st.form(K("eval_form")):
            st.markdown("### Preguntas de selección múltiple")

            q1 = st.radio(
                "1) En sistema perpetuo, ¿cómo se obtiene el Costo de la Mercancía Vendida (CMV) para el Estado de Resultados?",
                [
                    "a) Con la fórmula periódica: saldo inicial + compras – saldo final.",
                    "b) Directamente del KARDEX según el método (PP/PEPS/UEPS), incluyendo devoluciones.",
                    "c) Sumando todas las compras netas del período.",
                    "d) Con el promedio de precios de venta y compra.",
                ],
                index=None,
                key=K("q1")
            )

            q2 = st.radio(
                "2) ¿Cuál es el efecto de una devolución en ventas sobre el CMV y las ventas netas?",
                [
                    "a) Aumenta el CMV y aumenta las ventas netas.",
                    "b) Disminuye el CMV y disminuye las ventas netas.",
                    "c) No afecta el CMV y aumenta las ventas netas.",
                    "d) Aumenta el CMV y disminuye las ventas netas.",
                ],
                index=None,
                key=K("q2")
            )

            st.markdown("### Preguntas abiertas")
            open1 = st.text_area(
                "3) Explica la relación entre el KARDEX y el Estado de Resultados en sistema perpetuo, "
                "indicando específicamente cómo influye el método de inventario en el cálculo del CMV.",
                height=140,
                key=K("open1")
            )
            ask_ai_open1 = st.checkbox("💬 Pedir feedback para la pregunta 3 (opcional)", key=K("ai_open1"), value=False)

            open2 = st.text_area(
                "4) Bajo el método Promedio Ponderado, describe paso a paso cómo impactan en el Estado de Resultados: "
                "(i) una devolución en compras y (ii) una devolución en ventas.",
                height=140,
                key=K("open2")
            )
            ask_ai_open2 = st.checkbox("💬 Pedir feedback para la pregunta 4 (opcional)", key=K("ai_open2"), value=False)

            # ---------- Q5: Estructura tipo PRÁCTICA ----------
            st.markdown("### Ejercicio tipo práctica — Estado de Resultados (Promedio Ponderado)")
            _sc = exam_scenario_q5()
            total_gastos = sum(v for _, v in _sc["gastos_operativos"])

            # (1) Escenario — Solo variables clave visibles
            st.markdown("#### **Escenario — Empresa “Mercantil XYZ S.A.S.”**")

            # Formateo con separador de miles y 2 decimales
            valores_escenario = {
                "Cantidad vendida": f"{_sc['venta_u']} unidades",
                "Precio de venta": f"${_sc['p_venta']:,.2f}",
                "Gastos operacionales totales": f"${total_gastos:,.2f}",
                "Otros ingresos": f"${_sc['otros_ing']:,.2f}",
                "Otros egresos": f"${_sc['otros_egr']:,.2f}",
                "Tasa de impuesto": f"{int(_sc['tasa']*100)} %",
                "Método": _sc["metodo"],
            }

            # Mostrar como viñetas (markdown)
            st.markdown(
                "\n".join([f"- **{k}:** {v}" for k, v in valores_escenario.items()])
            )

            # (2) KARDEX de referencia (solo PP)
            st.markdown("#### 🧮 KARDEX de referencia (Promedio Ponderado)")
            df_kdx_pp = _pd.DataFrame(_kardex_rows_pp(_sc))
            st.dataframe(df_kdx_pp, use_container_width=True)

            # (3) ✍️ Completa tu Estado de Resultados
            order_rows = [
                "Ventas brutas",
                "(-) Devoluciones en ventas",
                "Ventas netas",
                "CMV",
                "Utilidad bruta",
                "Gastos operativos",
                "Resultado operativo",
                "Otros ingresos",
                "Otros egresos",
                "Utilidad antes de impuesto",
                "Impuesto",
                "Utilidad neta"
            ]
            df_er_blank = _pd.DataFrame({"Rubro": order_rows, "Valor": [""]*len(order_rows)})

            st.markdown("#### ✍️ Completa el **Estado de Resultados**")
            st.caption("Diligencia los valores numéricos. El validador verificará rubro por rubro con tolerancia ±0.5.")
            edited_er = st.data_editor(
                df_er_blank,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Rubro": st.column_config.TextColumn(disabled=True),
                    "Valor": st.column_config.NumberColumn(step=0.01, help="Valor monetario (usa números)"),
                },
                key=K("er_editor"),
                num_rows="fixed",
                disabled=False
            )

            # Feedback IA específico del ejercicio (opcional)
            ask_ai_q5 = st.checkbox("💬 Retroalimentación de IA para el ejercicio (opcional)", key=K("ai_q5"), value=False)

            submitted = st.form_submit_button("🧪 Enviar evaluación")

        # =====================================================
        # Corrección y resultado
        # =====================================================
        if submitted:
            # --- MCQ ---
            correct_mcq = {
                K("q1"): "b) Directamente del KARDEX según el método (PP/PEPS/UEPS), incluyendo devoluciones.",
                K("q2"): "b) Disminuye el CMV y disminuye las ventas netas.",
            }
            q1_ok = (st.session_state.get(K("q1")) == correct_mcq[K("q1")])
            q2_ok = (st.session_state.get(K("q2")) == correct_mcq[K("q2")])

            # --- Abiertas ---
            def grade_open_generic(text: str, focus: str):
                prompt = (
                    "Evalúa la respuesta del estudiante (máx. 4 líneas de feedback). "
                    "Primera línea EXACTA debe ser 'SCORE: 1' si el texto está en tema y cubre al menos dos puntos relevantes; "
                    "si no, 'SCORE: 0'. Prohibido desviarse a asientos/IVA/ecuación contable.\n"
                    f"TEMA: {focus}\n"
                    f"RESPUESTA:\n{text}"
                )
                raw = safe_ia_feedback(prompt, default="")
                sraw = str(raw or "")
                first = sraw.strip().splitlines()[0].strip() if sraw.strip() else ""
                score1 = 1 if first.upper().endswith("1") else 0
                fb = "\n".join(sraw.strip().splitlines()[1:]).strip()

                rate_limited = bool(st.session_state.get(K("ai_rate_limited"), False))
                if not fb or rate_limited:
                    fb = _on_topic_fallback_open()

                banned_student = [
                    "activo = pasivo + patrimonio", "ecuación contable", "asiento", "iva", "debe", "haber",
                    "balance de comprobación", "pasivo", "patrimonio"
                ]
                if any(b in (text or "").lower() for b in banned_student):
                    score1 = 0
                    fb = _on_topic_fallback_open()

                return score1, fb

            open1_text = st.session_state.get(K("open1"), "") or ""
            open2_text = st.session_state.get(K("open2"), "") or ""

            q3_score1, q3_fb = grade_open_generic(
                open1_text,
                "Relación KARDEX ↔ Estado de Resultados en sistema perpetuo; impacto del método de inventario en el CMV."
            )
            q4_score1, q4_fb = grade_open_generic(
                open2_text,
                "Efecto en el Estado de Resultados de devoluciones en compras y en ventas bajo Promedio Ponderado."
            )

            if not st.session_state.get(K("ai_open1"), False):
                q3_fb = ""
            else:
                q3_fb = _sanitize_on_topic(q3_fb)

            if not st.session_state.get(K("ai_open2"), False):
                q4_fb = ""
            else:
                q4_fb = _sanitize_on_topic(q4_fb)

            # --- Q5: Validación del Estado de Resultados (rubro por rubro) ---
            expected_pyg = _pyg_expected_from_scenario_pp(exam_scenario_q5())

            tol = 0.5
            def _to_float_or_none(x):
                try:
                    if x in (None, ""): return None
                    return float(x)
                except:
                    return None

            def _near(a, b, tol=tol):
                if a is None or b is None:
                    return False
                try:
                    return abs(float(a) - float(b)) <= tol
                except:
                    return False

            order_rows = list(expected_pyg.keys())
            er_checks = []
            er_correct_rows = 0

            try:
                edited_er_df = st.session_state[K("er_editor")]
            except Exception:
                edited_er_df = None

            if edited_er_df is None or not isinstance(edited_er_df, _pd.DataFrame):
                q5_ok = False
            else:
                for i, rubro in enumerate(order_rows):
                    usr_val = _to_float_or_none(edited_er_df.iloc[i]["Valor"])
                    exp_val = float(expected_pyg[rubro])
                    ok = _near(usr_val, exp_val)
                    er_checks.append((rubro, usr_val, exp_val, ok))
                    if ok:
                        er_correct_rows += 1
                q5_ok = (er_correct_rows == len(order_rows))

            # --- Feedback IA específico Q5 (opcional) ---
            q5_fb = ""
            if st.session_state.get(K("ai_q5"), False):
                intento_lines = []
                for rubro, usr_val, exp_val, ok in er_checks:
                    uv = "—" if usr_val is None else f"{usr_val:.2f}"
                    intento_lines.append(f"{rubro}: {uv}")
                intento_txt = "\n".join(intento_lines)
                exp_txt = "\n".join([f"{k}: {v:.2f}" for k, v in expected_pyg.items()])

                prompt_q5 = (
                    "Evalúa el Estado de Resultados diligenciado por el estudiante. "
                    "Centra el feedback en: coherencia con KARDEX PP (CMV y devoluciones), "
                    "ventas netas, CMV, y derivación de utilidades e impuesto. "
                    "Primera línea EXACTA: 'SCORE: 1' si ≥80% de rubros están correctos y no hay errores conceptuales graves; "
                    "si no, 'SCORE: 0'. Luego 3–5 líneas con correcciones puntuales.\n\n"
                    f"INTENTO (valores del estudiante):\n{intento_txt}\n\n"
                    f"ESPERADO (PP):\n{exp_txt}"
                )
                q5_fb_raw = safe_ia_feedback(prompt_q5, default="")
                q5_fb = _sanitize_on_topic(q5_fb_raw)

            # --- Resultado global ---
            total_hits = int(q1_ok) + int(q2_ok) + int(q3_score1) + int(q4_score1) + int(q5_ok)
            passed = (total_hits == 5)

            # Registro (si tienes estas funciones y 'username')
            try:
                record_attempt(username, level=4, score=total_hits, passed=passed)  # noqa: F821
            except Exception:
                pass

            st.markdown("### Resultado")
            cA, cB = st.columns(2)
            with cA: st.metric("Aciertos", f"{total_hits}/5")
            with cB: st.metric("Estado", "APROBADO ✅" if passed else "NO APROBADO ❌")

            # Detalle estilo PRÁCTICA
            with st.expander("Detalle de corrección"):
                st.write(f"**Q1 (MCQ):** {'✅' if q1_ok else '❌'}")
                st.write(f"**Q2 (MCQ):** {'✅' if q2_ok else '❌'}")
                st.write(f"**Q3 (abierta):** {'✅' if q3_score1==1 else '❌'}")
                if q3_fb:
                    st.write("**Feedback Q3 (IA):**")
                    st.write(q3_fb)
                st.write(f"**Q4 (abierta):** {'✅' if q4_score1==1 else '❌'}")
                if q4_fb:
                    st.write("**Feedback Q4 (IA):**")
                    st.write(q4_fb)

                st.write(f"**Q5 (Estado de Resultados PP):** {'✅' if q5_ok else '❌'}")
                st.metric("Rubros correctos en Q5", f"{er_correct_rows}/{len(order_rows)}")
                for rubro, usr_val, exp_val, ok in er_checks:
                    uv = "—" if usr_val is None else f"{usr_val:.2f}"
                    st.write(("✅ " if ok else "❌ ") + f"**{rubro}** — tu valor: {uv} | esperado: {exp_val:.2f}")

                if q5_fb:
                    st.write("**Feedback Q5 (IA):**")
                    st.write(q5_fb)

                if not q5_ok:
                    st.caption(
                        "Pistas: (1) Ventas netas = Ventas brutas − Dev. ventas; "
                        "(2) El CMV que se presenta en el ER ya descuenta el costo de las unidades devueltas; "
                        "(3) Utilidad bruta = Ventas netas − CMV; "
                        "(4) Resultado operativo, UAI, Impuesto y UN en ese orden."
                    )

            # Celebración / avance (si tienes estas funciones)
            if passed:
                try:
                    set_level_passed(st.session_state.get('progress_col'), username, "level4", total_hits)  # noqa: F821
                except Exception:
                    pass
                st.success("🎉 ¡Aprobaste la Evaluación del Nivel 4! Avanzas al siguiente módulo.")
                try:
                    start_celebration(  # noqa: F821
                        message_md=(
                            "<b>¡Nivel 4 dominado!</b> 📑💼<br><br>"
                            "Construiste y validaste el Estado de Resultados en sistema perpetuo con devoluciones."
                        ),
                        next_label="Volver al menú",
                        next_key_value="🏠 Inicio"
                    )
                except Exception:
                    pass
            else:
                st.error("No aprobado. Debes acertar 5/5. Repasa la integración KARDEX ↔ ER, el tratamiento de devoluciones y el CMV único del Estado de Resultados.")


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
    # Dos columnas: izquierda (login) / derecha (imagen)
    col1, col2 = st.columns([1, 1])

    # -------- Columna izquierda: formulario de login --------
    with col1:
        st.header("Iniciar Sesión")
        with st.form("login_form"):
            st.text_input("Usuario", key="login_raw_user")
            st.text_input("Contraseña", type="password", key="login_password")
            st.form_submit_button("Ingresar", on_click=do_login)

        if st.session_state.get("login_error"):
            st.error(st.session_state["login_error"])

        st.markdown("---")
        st.caption("Ingresa para comenzar la experiencia gamificada de inventarios. 🎮📊")

    # -------- Columna derecha: imagen de la herramienta --------
    with col2:
        st.markdown("## ")

        st.markdown(
            """
            <div style="display: flex; justify-content: center;">
                <img src="https://i.ibb.co/9mh1LX4B/Logotipo-Herramienta.png" 
                    width="600">
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div style="text-align: center;">
                <h3>Herramienta Contable – Inventarios Gamificados</h3>
                <p>Aprende inventarios con escenarios, niveles y retroalimentación interactiva.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

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

