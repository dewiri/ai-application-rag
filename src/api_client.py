import os
from pathlib import Path
from groq import Groq

# 🔁 Nur lokal .env laden – auf Streamlit gibt es keine
if not os.getenv("GROQ_API_KEY"):  # falls nicht bereits vorhanden (z. B. durch Secrets)
    env_path = Path(__file__).resolve().parent.parent / "env" / ".env"
    if env_path.exists():
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=env_path)

# 🔑 API-Key holen
groq_api_key = os.getenv("GROQ_API_KEY")

# 🧪 Validierung
if not groq_api_key or not groq_api_key.startswith("gsk_"):
    raise ValueError("❌ GROQ_API_KEY fehlt oder ist ungültig.")

# ✅ Client initialisieren
client = Groq(api_key=groq_api_key)

# 🐛 Debug-Log (optional, lokal)
if os.getenv("ENV", "LOCAL") == "LOCAL":
    print("✅ GROQ API Key geladen:", groq_api_key[:10], "...")