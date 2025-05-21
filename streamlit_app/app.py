import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from src.retrieval import retrieve
from src.api_client import client
import base64

# ✅ Streamlit-Seitenkonfiguration
st.set_page_config(page_title="Catan Rule Expert", page_icon="🧱")

# ✅ Hintergrundbild laden und einbetten
def get_base64_image(image_path):
    with open(image_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

bg_image = get_base64_image("streamlit_app/assets/catan_bg.jpg")

# ✅ Stildefinition (CSS)
st.markdown(
    f"""
    <style>
    html, body {{
        background-image: url("data:image/jpg;base64,{bg_image}");
        background-size: cover;
        background-attachment: fixed;
        color: #111111;
    }}
    .stApp {{
        background-color: rgba(250, 235, 215, 0.82);
        padding: 2rem;
        border-radius: 1rem;
        max-width: 850px;
        margin: auto;
        box-shadow: 0 0 30px rgba(0,0,0,0.2);
    }}
    h1 {{
        color: #111111;
        text-align: center;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }}
    .stSelectbox, .stTextInput {{
        background-color: #111111;
        border-radius: 0.5rem;
        padding: 0.75rem 1rem;
        font-size: 1rem;
    }}
    .stMarkdown {{
        font-size: 1.1rem;
        color: #111111;
    }}
    details.st-expander {{
        border: 1px solid #111111 !important;
        border-radius: 8px;
        padding: 0.5rem;
        color: #111111;
    }}
    details.st-expander summary {{
        color: #111111;
        font-weight: 600;
        font-size: 1rem;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# ✅ App-Überschrift
st.markdown("""
    <h1 style='text-align: center; color: #B22222;'>🧱 Catan Regel-Chatbot</h1>
""", unsafe_allow_html=True)

st.markdown("Stelle mir Fragen zu den Spielregeln von **Catan** (inkl. Erweiterungen).")

# ✅ Modellwahl
model = st.selectbox(" - Modell", ["llama3-70b-8192", "mixtral-8x7b-32768"])

# ✅ Texteingabe
query = st.text_input(" ❓ Deine Frage")

# ✅ Verarbeitung & Antwort
if query:
    with st.spinner("🔄 Denke nach ..."):
        docs = retrieve(query, top_k=5)
        context = "\n\n".join(docs)

        prompt = f"""Beantworte die folgende Frage basierend auf dem gegebenen Kontext.

==================== Kontext =====================
{context}

==================== Frage =====================
{query}

==================== Antwort ===================="""

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Du bist ein hilfreicher Catan-Regel-Experte."},
                {"role": "user", "content": prompt}
            ]
        )

        answer = response.choices[0].message.content.strip()

        # ✅ Antwort anzeigen
        st.markdown("### 💬 Antwort")
        st.write(answer)

        # ✅ Kontext anzeigen mit gestyltem HTML-Block
        with st.expander("🔍 Kontext anzeigen"):
            st.markdown(
                f"<div style='background-color: rgba(0, 0, 0, 0.05); "
                f"padding: 1rem; border-radius: 8px; color: #111111; "
                f"font-family: sans-serif; font-size: 0.95rem;'>{context.replace(chr(10), '<br>')}</div>",
                unsafe_allow_html=True
            )