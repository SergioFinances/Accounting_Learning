# db_connection.py
import streamlit as st
from pymongo import MongoClient
from pymongo.server_api import ServerApi

@st.cache_resource
def get_mongo_client():
    """
    Retorna un MongoClient cacheado para toda la sesión de la app.
    Lee la URI desde .streamlit/secrets.toml
    """
    uri = st.secrets["mongodb"]["uri"]  # ya lo tienes configurado
    client = MongoClient(uri, server_api=ServerApi('1'), serverSelectionTimeoutMS=8000)
    # Verificación rápida
    client.admin.command("ping")
    return client