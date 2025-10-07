# repo.py
from datetime import datetime
from pymongo import ASCENDING
from passlib.context import CryptContext
from db_connection import get_mongo_client

pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")

DB_NAME = "accounting_app"
USERS_COL = "users"
PROG_COL  = "progress"

def default_progress_doc(username: str):
    now = datetime.utcnow()
    return {
        "username": username,
        "level1": {"passed": False, "date": None, "score": None, "time_sec": 0},
        "level2": {"passed": False, "date": None, "score": None, "time_sec": 0},
        "level3": {"passed": False, "date": None, "score": None, "time_sec": 0},
        "level4": {"passed": False, "date": None, "score": None, "time_sec": 0},
        "survey_unlocked": False,
        "updated_at": now,
        "created_at": now,
    }

def repo_init():
    """
    Crea conexión, índices y garantiza un admin por defecto.
    Llama a esta función una vez al arrancar tu app (en main()).
    """
    client = get_mongo_client()
    db = client[DB_NAME]
    users = db[USERS_COL]
    progress = db[PROG_COL]

    # Índices únicos
    users.create_index([("username", ASCENDING)], unique=True)
    progress.create_index([("username", ASCENDING)], unique=True)

    # Admin por defecto
    admin_user = "admin"
    admin_pass = "AdminSeguro#2025"  # cámbiala luego desde el panel
    if not users.find_one({"username": admin_user}):
        users.insert_one({
            "username": admin_user,
            "password_hash": pwd_ctx.hash(admin_pass),
            "role": "admin",
            "created_at": datetime.utcnow()
        })
        progress.insert_one(default_progress_doc(admin_user))

    return db, users, progress

# -------- Operaciones de usuarios (CRUD) --------

def create_user(users_col, progress_col, username: str, password: str, role: str = "user"):
    users_col.insert_one({
        "username": username,
        "password_hash": pwd_ctx.hash(password),
        "role": role,
        "created_at": datetime.utcnow()
    })
    if not progress_col.find_one({"username": username}):
        progress_col.insert_one(default_progress_doc(username))

def update_user(users_col, username: str, new_password: str | None, new_role: str | None):
    update = {"updated_at": datetime.utcnow()}
    if new_password:
        update["password_hash"] = pwd_ctx.hash(new_password)
    if new_role:
        update["role"] = new_role
    users_col.update_one({"username": username}, {"$set": update})

def delete_user(users_col, progress_col, username: str):
    users_col.delete_one({"username": username})
    progress_col.delete_one({"username": username})

def verify_credentials(users_col, username: str, plain_password: str) -> dict | None:
    """
    Devuelve el doc de usuario si la contraseña es válida; en caso contrario None.
    """
    doc = users_col.find_one({"username": username})
    if not doc:
        return None
    if pwd_ctx.verify(plain_password, doc.get("password_hash", "")):
        return doc
    return None
