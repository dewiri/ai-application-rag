from dotenv import load_dotenv
import os
from groq import Groq
from pathlib import Path

# Dynamisch relativ zum Projektpfad laden
env_path = Path(__file__).resolve().parent.parent / "env" / ".env"
load_dotenv(dotenv_path=env_path)

groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key or not groq_api_key.startswith("gsk_"):
    raise ValueError("❌ GROQ_API_KEY fehlt oder ist ungültig.")

client = Groq(api_key=groq_api_key)

# Optional Debug
print("✅ GROQ API Key geladen:", groq_api_key[:10], "...")