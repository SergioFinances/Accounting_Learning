# -*- coding: utf-8 -*-
# =========================================================
#   Herramienta Contable - Inventarios Gamificados (con Mongo + Fallback Data API)
#   Niveles por pestaña (desbloqueo progresivo)
#   Pantalla de celebración aparte (confeti + globos + botón)
#   IA DeepSeek vía OpenRouter para feedback
#   Admin con CRUD desde MongoDB (users)
#   Fecha: 2025-10-07 (fallback Atlas Data API)
# =========================================================

import os
import ssl
import random
from datetime import datetime, timezone

import numpy as np
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
    (Web Speech API del navegador)
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
# Pantalla de Celebración (aparte)
# ===========================
def confetti_block(duration_ms: int = 6000, height_px: int = 340):
    """
    Confeti y 'globos' simples 100% inline (sin CDNs).
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
      const pick = (arr)=>Math.floor(Math.random()*arr.length);

      const pieces = [];
      const N = 180;
      for (let i=0;i<N;i++) {{
        pieces.push({{
          type: Math.random()<0.4 ? 'tri' : 'rect',
          x: Math.random()*canvas.width,
          y: (Math.random()*-canvas.height),
          w: 6 + Math.random()*6,
          h: 8 + Math.random()*10,
          r: Math.random()*Math.PI*2,
          vr: (Math.random()-0.5)*0.2,
          vx: (Math.random()-0.5)*1.2,
          vy: 1.8 + Math.random()*1.4,
          color: ['#ff6b6b','#ffd93d','#6BCB77','#4D96FF','#845EC2','#FF9671','#FFC75F'][pick(['a','b','c','d','e','f','g'])],
          alpha: 0.85 + Math.random()*0.15
        }});
      }}

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
    confetti_block(duration_ms=6500, height_px=360)
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
from pymongo.errors import ServerSelectionTimeoutError, ConfigurationError
import certifi

# ===== Data API Adapter (fallback HTTPS) =====
import json
import requests

class DataApiCollectionAdapter:
    """
    Adapter mínimo para simular métodos de PyMongo Collection usando MongoDB Atlas Data API (HTTPS).
    Implementa: find_one, find, insert_one, update_one, delete_one, count_documents.
    """
    def __init__(self, base_url, api_key, data_source, database, collection):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.data_source = data_source
        self.database = database
        self.collection = collection
        self.headers = {
            "Content-Type": "application/json",
            "api-key": self.api_key
        }

    def _action(self, name, payload):
        url = f"{self.base_url}/action/{name}"
        body = {
            "dataSource": self.data_source,
            "database": self.database,
            "collection": self.collection,
            **payload
        }
        r = requests.post(url, headers=self.headers, data=json.dumps(body), timeout=20)
        r.raise_for_status()
        return r.json()

    def find_one(self, filter: dict, projection: dict | None = None):
        payload = {"filter": filter}
        if projection:
            payload["projection"] = projection
        res = self._action("findOne", payload)
        return res.get("document")

    def find(self, filter: dict, projection: dict | None = None, sort: list | None = None, limit: int = 100):
        payload = {"filter": filter, "limit": limit}
        if projection:
            payload["projection"] = projection
        if sort:
            payload["sort"] = dict(sort)
        res = self._action("find", payload)
        return res.get("documents", [])

    def insert_one(self, document: dict):
        res = self._action("insertOne", {"document": document})
        return res

    def update_one(self, filter: dict, update: dict):
        res = self._action("updateOne", {"filter": filter, "update": {"$set": update}})
        return res

    def delete_one(self, filter: dict):
        res = self._action("deleteOne", {"filter": filter})
        return res

    def count_documents(self, filter: dict):
        res = self._action("countDocuments", {"filter": filter})
        return res.get("count", 0)

class RepoHandle:
    """
    Mantiene referencias users_col y progress_col ya sea de PyMongo o del Adapter Data API.
    """
    def __init__(self, users_col, progress_col, backend: str):
        self.users_col = users_col
        self.progress_col = progress_col
        self.backend = backend  # 'pymongo' o 'data_api'

# Hashing: preferimos passlib[bcrypt]; si no está, caemos a SHA256 (uso educativo)
try:
    from passlib.context import CryptContext
    _pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")
    def hash_password(p: str) -> str:
        return _pwd_ctx.hash(p)
    def verify_password(p: str, h: str) -> bool:
        try:
            return _pwd_ctx.verify(p, h)
        except Exception:
            return False
except Exception:
    import hashlib
    st.warning("Passlib no disponible: usando SHA256 solo para pruebas (no apto producción).")
    def hash_password(p: str) -> str:
        return hashlib.sha256(("pepper123::"+p).encode("utf-8")).hexdigest()
    def verify_password(p: str, h: str) -> bool:
        return hash_password(p) == h

from passlib.context import CryptContext
pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")

def _mk_client(uri: str, insecure: bool = False) -> MongoClient:
    """
    Crea un cliente MongoDB con parámetros consistentes.
    insecure=True activa tlsAllowInvalidCertificates (fallback).
    """
    common_kwargs = {
        "server_api": ServerApi('1'),
        "retryWrites": True,
        "connectTimeoutMS": 20000,
        "socketTimeoutMS": 20000,
        "appname": "AccountingLearningStreamlit",
    }
    tls_kwargs = {"tls": True, "tlsCAFile": certifi.where()}
    if insecure:
        tls_kwargs.update({"tlsAllowInvalidCertificates": True})
    return MongoClient(uri, **common_kwargs, **tls_kwargs)

def repo_init():
    """
    1) Intenta PyMongo (driver). Si funciona -> backend 'pymongo'.
    2) Si falla por TLS/selección de servidor -> Fallback Data API (HTTPS) -> backend 'data_api'.
    Requiere vars Data API: DATA_API_URL, DATA_API_KEY, MONGODB_DATA_SOURCE.
    """
    # ---- 1) Intento con driver ----
    uri = os.environ.get("MONGODB_URI")
    if not uri:
        # como backup, intenta secrets
        try:
            uri = st.secrets["mongodb"]["uri"]
        except Exception:
            uri = None

    driver_error = None
    if uri:
        try:
            client = _mk_client(uri, insecure=(os.getenv("MONGODB_TLS_INSECURE","")=="1"))
            db = client["accounting_app"]
            users_col = db["users"]
            progress_col = db["progress"]
            client.admin.command("ping")
            return RepoHandle(users_col, progress_col, backend="pymongo")
        except (ssl.SSLError, ServerSelectionTimeoutError) as e:
            driver_error = e  # continuar a Data API
        except Exception as e:
            driver_error = e  # continuar a Data API

    # ---- 2) Fallback Data API ----
    data_api_url = os.getenv("DATA_API_URL")  # p.ej: https://us-east-1.aws.data.mongodb-api.com/app/<APP_ID>/endpoint/data/v1
    data_api_key = os.getenv("DATA_API_KEY")
    data_source  = os.getenv("MONGODB_DATA_SOURCE")  # p.ej: Cluster0
    database     = os.getenv("MONGODB_DB", "accounting_app")

    if not all([data_api_url, data_api_key, data_source]):
        missing = []
        if not data_api_url: missing.append("DATA_API_URL")
        if not data_api_key: missing.append("DATA_API_KEY")
        if not data_source:  missing.append("MONGODB_DATA_SOURCE")
        raise RuntimeError(
            "Error conectando a MongoDB.\n\n"
            f"Detalles driver: {driver_error}\n\n"
            "No fue posible usar el driver. Para el fallback HTTPS (Atlas Data API) faltan variables: "
            + ", ".join(missing)
        )

    users_adapter    = DataApiCollectionAdapter(data_api_url, data_api_key, data_source, database, "users")
    progress_adapter = DataApiCollectionAdapter(data_api_url, data_api_key, data_source, database, "progress")

    # Prueba rápida de reachability: countDocuments vacío
    try:
        _ = users_adapter.count_documents({})
    except Exception as e:
        raise RuntimeError(
            "No fue posible conectarse mediante Data API (HTTPS). "
            f"Revisa DATA_API_URL/DATA_API_KEY/MONGODB_DATA_SOURCE. Detalle: {e}"
        )

    st.warning("Usando **Atlas Data API (HTTPS)** como respaldo. No se usan sockets TLS al puerto 27017.")
    return RepoHandle(users_adapter, progress_adapter, backend="data_api")

def verify_credentials(users_col, username: str, password: str):
    if users_col is None:
        st.error("Colección de usuarios no inicializada.")
        return None
    # Data API adapter y PyMongo comparten interfaz usada aquí:
    doc = users_col.find_one({"username": username})
    if not doc:
        return None
    password_hash = doc.get("password_hash")
    try:
        if password_hash and pwd_ctx.verify(password, password_hash):
            return doc
    except Exception:
        return None
    return None

def create_user(users_col, progress_col, username: str, password: str, role: str = "user"):
    users_col.insert_one({
        "username": username.strip().lower(),
        "password_hash": hash_password(password),
        "role": role,
        "created_at": datetime.now(timezone.utc).isoformat()
    })
    # crear doc de progreso base si no existe
    existing = progress_col.find_one({"username": username.strip().lower()})
    if not existing:
        progress_col.insert_one({"username": username.strip().lower(), "levels": {}})

def update_user(users_col, username: str, new_password: str | None, new_role: str | None):
    update = {}
    if new_password:
        update["password_hash"] = hash_password(new_password)
    if new_role:
        update["role"] = new_role
    if update:
        # PyMongo: update_one({"username":..}, {"$set": update})
        # Data API adapter hace lo mismo internamente
        users_col.update_one({"username": username}, update)

def delete_user(users_col, progress_col, username: str):
    users_col.delete_one({"username": username})
    progress_col.delete_one({"username": username})

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
    doc = verify_credentials(users_col, user, pwd)
    if doc:
        st.session_state.authenticated = True
        st.session_state.username = user
        st.session_state.login_error = ""
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

    options = [
        "Nivel 1: Introducción a Inventarios",
        "Nivel 2: Métodos (PP/PEPS/UEPS)",
        "Nivel 3: Devoluciones",
        "Nivel 4: Estado de Resultados",
        "Encuesta de satisfacción",
    ]

    # rol desde Mongo
    current_user_role = "user"
    users_col = st.session_state.get("users_col")
    if users_col and username:
        doc = users_col.find_one({"username": username}, {"role": 1, "_id": 0}) if hasattr(users_col, "find_one") else None
        if not doc and hasattr(users_col, "find"):
            # Adapter find con proyección
            docs = users_col.find({"username": username}, {"role": 1, "_id": 0}, limit=1)
            doc = docs[0] if docs else {}
        current_user_role = (doc or {}).get("role", "user")

    if current_user_role == "admin":
        options.append(ADMIN_OPTION)

    # limpia selección inválida si cambiara el menú
    if "sidebar_level_select" in st.session_state and st.session_state.sidebar_level_select not in options:
        del st.session_state["sidebar_level_select"]

    sel = st.sidebar.radio("Ir a:", options, key="sidebar_level_select")

    st.sidebar.markdown("---")
    st.sidebar.caption(f"Usuario: **{username}** · Rol: **{current_user_role}**")
    st.sidebar.button("Cerrar Sesión", on_click=logout, key="logout_btn")
    return sel

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
            "Valorar inventarios es asignar un **costo monetario** a las existencias que mantiene una empresa para vender. "
            "Ese costo aparece como **activo** (Inventarios) y determina el **Costo de Ventas (COGS)** en el estado de resultados, "
            "afectando la **utilidad bruta**. En un **sistema periódico**, no actualizas inventarios con cada venta: "
            "acumulas durante el período y cierras con la fórmula base:\n\n"
            "  **COGS = Inventario Inicial + Compras - Devoluciones - Inventario Final**\n\n"
            "- **InvI:** lo que tenías al empezar.\n"
            "- **Compras:** adquisiciones del período (incluso costos necesarios para dejar el inventario disponible).\n"
            "- **Devoluciones:** restan a Compras cuando devuelves a proveedor.\n"
            "- **InvF:** lo que queda al cierre; su **valoración** depende del método (PP/PEPS/UEPS del Nivel 2).\n\n"
            "Regla mental: imagina una **mochila de costo**. Entra InvI y Compras; si devuelves, sacas parte (Devoluciones). "
            "Al final miras qué queda dentro (InvF). **Lo que salió** para vender es el **COGS**."
        )
        st.write(teoria)
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
            st.write(f"**1) InvI + Compras** → {peso(inv0)} + {peso(compras)} = **{peso(inv0+compras)}**")
            st.write(f"**2) − Devoluciones**  → {peso(inv0+compras)} − {peso(devol)} = **{peso(inv0+compras-devol)}**")
            st.write(f"**3) − InvF**          → {peso(inv0+compras-devol)} − {peso(invf)} = **{peso(inv0+compras-devol-invf)}**")
            cogs = inv0 + compras - devol - invf
            st.success(f"**COGS (Costo de Ventas)** = {peso(cogs)}")
            st.caption("Interpretación: la ‘mochila de costo’ se llenó con InvI y Compras; devolviste parte (Devoluciones) "
                       "y lo que quedó al cierre (InvF) no salió a ventas. El resto es COGS.")

        st.markdown("—")
        st.write("**Mini reto**: explica qué pasaría con el COGS si **no hubiera devoluciones** y el **Inventario Final fuera muy pequeño**.")
        razonamiento = st.text_area("Tu razonamiento (opcional, la IA te comenta):", key="n1_ex_raz")

        if st.button("💬 Comentar con IA (opcional)", key="n1_ex_fb"):
            prompt = (
                "Evalúa si el razonamiento es coherente con COGS = InvI + Compras - Devoluciones - InvF. "
                f"Datos: InvI={inv0}, Compras={compras}, Devoluciones={devol}, InvF={invf}. "
                f"Texto del estudiante: {razonamiento}"
            )
            fb = ia_feedback(prompt)
            st.info(fb)

    # Práctica interactiva (IA)
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
        user_comment = st.text_area("Justifica brevemente (opcional, la IA lo comenta mejor):", key="n1_practice_comment")

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
            fb = ia_feedback(prompt)
            with st.expander("💬 Feedback de la IA"):
                st.write(fb)

    # Evaluación final
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
                f"Aciertos: {score}/3. Escribe un feedback breve y amable (máx 6 líneas)."
            )
            fb = ia_feedback(prompt)

            if passed:
                st.success(f"¡Aprobado! Aciertos {score}/3 🎉 Se habilitará el Nivel 2 en el menú.")
                start_celebration(
                    message_md=(
                        "<b>¡Nivel 1 superado!</b> 🏆<br><br>"
                        "Dominaste la fórmula del <b>COGS</b> y entendiste el sistema periódico. "
                        "Ahora sí: pasemos a los <b>métodos de valoración</b>."
                    ),
                    next_label="Nivel 2",
                    next_key_value="Nivel 2: Métodos (PP/PEPS/UEPS)"
                )
            else:
                st.error(f"No aprobado. Aciertos {score}/3. Repasa la teoría y vuelve a intentar.")
                with st.expander("💬 Feedback de la IA"):
                    st.write(fb)

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

            st.write(f"**Costo Promedio** = ({peso(inv0_val)} + {peso(comp_val)}) / ({inv0_u} + {comp_u}) = **{peso(prom)}**/u")
            st.write(f"**COGS** por venta de {venta_u} u = {venta_u} × {peso(prom)} = **{peso(cogs)}**")
            st.success(f"**Saldo final**: {saldo_u} u × {peso(prom)} = **{peso(saldo_val)}**")

        st.markdown("---")
        st.subheader("Ejemplo FIFO vs LIFO (comparación rápida)")
        inv = [(100, 10.0), (50, 12.0)]  # (u, $/u)
        venta = 120
        # FIFO
        fifo_cogs = 0.0; remaining = venta; inv_fifo = inv.copy()
        for u, pu in inv_fifo:
            use = min(remaining, u)
            fifo_cogs += use * pu
            remaining -= use
            if remaining <= 0: break
        # LIFO
        lifo_cogs = 0.0; remaining = venta; inv_lifo = inv.copy()[::-1]
        for u, pu in inv_lifo:
            use = min(remaining, u)
            lifo_cogs += use * pu
            remaining -= use
            if remaining <= 0: break

        st.write(f"Venta: {venta} u. Inventario: 100u @10; 50u @12")
        st.info(f"**FIFO COGS** ≈ {peso(fifo_cogs)} · **LIFO COGS** ≈ {peso(lifo_cogs)}  → (LIFO mayor COGS con precios al alza)")

    with tabs[2]:
        st.subheader("Práctica: elige el método correcto")
        st.caption("Completa el cálculo según el método seleccionado.")

        metodo = st.selectbox("Método", ["Promedio Ponderado", "PEPS (FIFO)", "UEPS (LIFO)"], key="n2_pract_met")
        inv0_u = random.randint(50, 150)
        inv0_pu = random.choice([10.0, 11.0, 12.0])
        comp_u = random.randint(50, 200)
        comp_pu = random.choice([12.0, 13.0, 14.0])
        venta_u = random.randint(60, inv0_u + comp_u)

        st.write(f"Inv0: {inv0_u} u @ {peso(inv0_pu)} | Compra: {comp_u} u @ {peso(comp_pu)} | Venta: {venta_u} u")

        ans_cogs = st.number_input("Tu COGS", min_value=0.0, value=0.0, step=10.0, key="n2_prac_cogs")
        if st.button("✅ Validar práctica N2", key="n2_prac_btn"):
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

        q1 = st.radio("1) En inflación, ¿cuál suele dar mayor COGS?",
                      ["PEPS", "UEPS", "Promedio Ponderado"], index=None, key="n2_eval_q1")
        q2 = st.radio("2) En PEPS, ¿con qué costos se valora el inventario final?",
                      ["Con los más antiguos", "Con los más recientes"], index=None, key="n2_eval_q2")
        q3 = st.radio("3) El Promedio Ponderado:",
                      ["Usa costo del último lote", "Mezcla costos para un único costo unitario"], index=None, key="n2_eval_q3")

        if st.button("🧪 Validar evaluación N2", key="n2_eval_btn"):
            correct = {
                "n2_eval_q1": "UEPS",
                "n2_eval_q2": "Con los más recientes",
                "n2_eval_q3": "Mezcla costos para un único costo unitario"
            }
            answers = {"n2_eval_q1": q1, "n2_eval_q2": q2, "n2_eval_q3": q3}
            score = sum(1 for k,v in answers.items() if v == correct[k])
            passed = score >= 2

            fb = ia_feedback(
                f"Nivel 2 evaluación. Respuestas estudiante: {answers}. Correctas: {correct}. "
                f"Aciertos: {score}/3. Da feedback amable y breve."
            )

            if passed:
                st.success(f"¡Aprobado! Aciertos {score}/3 🎉 Se habilitará el Nivel 3 en el menú.")
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
        inv0 = random.randint(500, 1500)
        prom0 = random.choice([15.0, 16.0, 17.0])
        comp = random.randint(500, 2000)
        comp_pu = random.choice([17.0, 18.0, 19.0])
        dev_comp = random.randint(0, int(comp*0.2))
        venta_u = random.randint(200, inv0 + comp)
        dev_venta_u = random.randint(0, int(venta_u*0.2))

        st.write(
            f"Inv0: {inv0} u @ {peso(prom0)} | Compra: {comp} u @ {peso(comp_pu)} | "
            f"Devol. compra: {peso(dev_comp)} (resta $) | Venta: {venta_u} u | Devol. venta: {dev_venta_u} u"
        )

        ans_cogs = st.number_input("Tu COGS estimado (PP)", min_value=0.0, value=0.0, step=10.0, key="n3_prac_cogs")
        if st.button("✅ Validar práctica N3", key="n3_prac_btn"):
            inv0_val = inv0 * prom0
            comp_val = comp * comp_pu
            comp_net_val = comp_val - dev_comp
            total_val = inv0_val + comp_net_val
            total_u = inv0 + comp
            prom = total_val / total_u
            venta_neta_u = max(venta_u - dev_venta_u, 0)
            correct = venta_neta_u * prom

            ok = abs(ans_cogs - correct) <= 0.5
            if ok:
                st.success(f"COGS (venta neta) ≈ {peso(correct)} con PP")
            else:
                st.error(f"COGS esperado ≈ {peso(correct)}")
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

        q1 = st.radio("1) La devolución de compra...",
                      ["Aumenta las compras", "Disminuye las compras", "No afecta las compras"], index=None, key="n3_eval_q1")
        q2 = st.radio("2) La devolución de venta (PP) reingresa unidades con costo...",
                      ["Del último lote", "Promedio vigente", "Más antiguo"], index=None, key="n3_eval_q2")
        q3 = st.radio("3) En términos de COGS, una devolución de venta...",
                      ["Disminuye el COGS neto", "Aumenta el COGS neto", "No lo afecta"], index=None, key="n3_eval_q3")

        if st.button("🧪 Validar evaluación N3", key="n3_eval_btn"):
            correct = {
                "n3_eval_q1": "Disminuye las compras",
                "n3_eval_q2": "Promedio vigente",
                "n3_eval_q3": "Disminuye el COGS neto"
            }
            answers = {"n3_eval_q1": q1, "n3_eval_q2": q2, "n3_eval_q3": q3}
            score = sum(1 for k,v in answers.items() if v == correct[k])
            passed = score >= 2

            fb = ia_feedback(
                f"Nivel 3 evaluación. Respuestas estudiante: {answers}. Correctas: {correct}. "
                f"Aciertos: {score}/3. Da feedback breve y amable."
            )

            if passed:
                st.success(f"¡Aprobado! Aciertos {score}/3 🎉 Se habilitará el Nivel 4 en el menú.")
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
        ventas = random.randint(8000, 20000)
        dev_vtas = random.randint(0, 1200)
        cogs = random.randint(4000, 12000)
        gastos = random.randint(1000, 5000)

        st.write(
            f"Ventas brutas={peso(ventas)}, Devol/Desc Ventas={peso(dev_vtas)}, "
            f"COGS={peso(cogs)}, Gastos Op.={peso(gastos)}"
        )
        ans_util_oper = st.number_input("Tu Utilidad Operativa", min_value=-100000.0, value=0.0, step=50.0, key="n4_prac_uop")

        if st.button("✅ Validar práctica N4", key="n4_prac_btn"):
            vtas_net = ventas - dev_vtas
            util_bruta = vtas_net - cogs
            correct = util_bruta - gastos
            if abs(ans_util_oper - correct) <= 0.5:
                st.success(f"¡Correcto! Utilidad operativa = {peso(correct)}")
            else:
                st.error(f"Utilidad operativa esperada = {peso(correct)}")
            fb = ia_feedback(
                f"N4 práctica EERR. Datos: Ventas={ventas}, DevVtas={dev_vtas}, COGS={cogs}, Gastos={gastos}. "
                f"UO_est={ans_util_oper}, UO_correcta={correct}. Explica pasos y da truco memotécnico."
            )
            with st.expander("💬 Feedback de la IA"):
                st.write(fb)

    with tabs[3]:
        st.subheader("Evaluación final del Nivel 4")
        st.caption("Necesitas acertar **2 de 3** para terminar el curso.")

        q1 = st.radio("1) Ventas netas se calculan como:",
                      ["Ventas brutas + Devoluciones", "Ventas brutas − Devoluciones/Descuentos", "Ventas brutas"], index=None, key="n4_eval_q1")
        q2 = st.radio("2) Utilidad bruta =",
                      ["Ventas netas − COGS", "Ventas netas − Gastos operativos", "Ventas brutas − COGS"], index=None, key="n4_eval_q2")
        q3 = st.radio("3) Utilidad operativa =",
                      ["Utilidad bruta − Gastos operativos", "Ventas netas − COGS − Gastos financieros", "COGS − Gastos operativos"], index=None, key="n4_eval_q3")

        if st.button("🧪 Validar evaluación N4", key="n4_eval_btn"):
            correct = {
                "n4_eval_q1": "Ventas brutas − Devoluciones/Descuentos",
                "n4_eval_q2": "Ventas netas − COGS",
                "n4_eval_q3": "Utilidad bruta − Gastos operativos"
            }
            answers = {"n4_eval_q1": q1, "n4_eval_q2": q2, "n4_eval_q3": q3}
            score = sum(1 for k,v in answers.items() if v == correct[k])
            passed = score >= 2

            fb = ia_feedback(
                f"Nivel 4 evaluación. Respuestas estudiante: {answers}. Correctas: {correct}. "
                f"Aciertos: {score}/3. Feedback amable y breve."
            )

            if passed:
                st.success(f"¡Felicidades! Aciertos {score}/3 🎓 Has completado los 4 niveles.")
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
# Módulo: Administrador de Usuarios (Mongo)
# ===========================
def admin_page():
    st.title("⚙️ Administrador de Usuarios")

    users_col = st.session_state.get("users_col")
    progress_col = st.session_state.get("progress_col")
    if not users_col:
        st.error("No hay conexión con la base de datos.")
        return

    # ---- Tabla de usuarios ----
    st.subheader("Usuarios actuales")
    # Compatibilidad adapter/driver
    rows = []
    try:
        # PyMongo: cursor .find
        rows = list(users_col.find({}, {"_id": 0, "username": 1, "role": 1, "created_at": 1}))
    except Exception:
        # Adapter Data API
        rows = users_col.find({}, {"_id": 0, "username": 1, "role": 1, "created_at": 1}, sort=[("username", 1)], limit=500)
    st.dataframe(rows, use_container_width=True)

    st.markdown("---")

    # ---- Crear usuario ----
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
            else:
                # evitar duplicados
                exists = users_col.find_one({"username": new_user})
                if exists:
                    st.error("El usuario ya existe.")
                else:
                    create_user(users_col, progress_col, new_user, new_pass, new_role)
                    st.success(f"Usuario '{new_user}' creado como {new_role}.")

    st.markdown("---")

    # ---- Editar usuario ----
    st.subheader("Editar usuario")
    try:
        usernames = [u["username"] for u in users_col.find({}, {"username": 1, "_id": 0})]
    except Exception:
        docs = users_col.find({}, {"username": 1, "_id": 0}, sort=[("username", 1)], limit=500)
        usernames = [d["username"] for d in docs]
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
                    try:
                        admin_count = users_col.count_documents({"role": "admin"})
                    except Exception:
                        # Adapter Data API
                        admin_count = users_col.count_documents({"role": "admin"})
                    if curr_role == "admin" and new_role_opt == "user" and admin_count <= 1:
                        st.error("No puedes quitar el último administrador del sistema.")
                    else:
                        update_user(users_col, edit_user, new_pass_opt or None, new_role_opt)
                        st.success(f"Usuario '{edit_user}' actualizado.")
    else:
        st.info("No hay usuarios para editar.")

    st.markdown("---")

    # ---- Eliminar usuario ----
    st.subheader("Eliminar usuario")
    try:
        usernames = [u["username"] for u in users_col.find({}, {"username": 1, "_id": 0})]
    except Exception:
        docs = users_col.find({}, {"username": 1, "_id": 0}, sort=[("username", 1)], limit=500)
        usernames = [d["username"] for d in docs]
    if usernames:
        del_user = st.selectbox("Selecciona el usuario a eliminar", usernames, key="admin_del_select")
        if st.button("🗑️ Eliminar usuario seleccionado"):
            if del_user == st.session_state.username:
                st.error("No puedes eliminar tu propia cuenta en esta vista.")
                return
            # proteger 'admin' si aplica
            doc = users_col.find_one({"username": del_user}, {"role": 1, "_id": 0}) or {}
            if doc.get("role") == "admin":
                try:
                    admin_count = users_col.count_documents({"role": "admin"})
                except Exception:
                    admin_count = users_col.count_documents({"role": "admin"})
                if admin_count <= 1:
                    st.error("No puedes eliminar el último administrador del sistema.")
                    return
            delete_user(users_col, progress_col, del_user)
            st.success(f"Usuario '{del_user}' eliminado.")
    else:
        st.info("No hay usuarios para eliminar.")

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
        # Última comprobación de rol por seguridad
        users_col = st.session_state.get("users_col")
        role = "user"
        if users_col:
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

    # Inicializa conexión y colecciones
    try:
        repo = repo_init()
        # Normaliza en session_state
        st.session_state["users_col"] = repo.users_col
        st.session_state["progress_col"] = repo.progress_col
        st.session_state["db_backend"] = repo.backend  # info útil
    except Exception as e:
        st.error(f"Error conectando a la base de datos.\n\nDetalles: {e}\n\n"
                 "Si ves errores de TLS con el driver y no se habilita el fallback, "
                 "asegúrate de definir las variables de Atlas Data API:\n"
                 "• DATA_API_URL  (endpoint base /endpoint/data/v1)\n"
                 "• DATA_API_KEY  (API Key de Data API)\n"
                 "• MONGODB_DATA_SOURCE (ej. Cluster0)\n"
                 "• MONGODB_DB=accounting_app (opcional, por defecto)")
        st.stop()

    # Flujo principal
    if not st.session_state.get("authenticated"):
        login_screen()
    else:
        main_app()

if __name__ == "__main__":
    main()
