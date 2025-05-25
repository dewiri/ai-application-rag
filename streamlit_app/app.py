import sys
import os
import base64
import re
import streamlit as st

# Pfad-Anpassung
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.retrieval import retrieve
from src.api_client import client

# --- Streamlit Setup ---
st.set_page_config(page_title="Catan Rule Expert", page_icon="🎲")

# Hintergrundbild laden
def get_base64_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

bg_image = get_base64_image("streamlit_app/assets/catan_bg.jpg")

# Styling
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
        background-color: rgba(250, 235, 215, 0.88);
        padding: 2rem;
        border-radius: 1rem;
        max-width: 850px;
        margin: auto;
        box-shadow: 0 0 30px rgba(0,0,0,0.2);
    }}
    h1, label {{
        color: #111111 !important;
    }}
    .example-header {{
        text-align: center;
        font-weight: bold;
        margin-top: 2rem;
        margin-bottom: 0.3rem;
    }}
    .example-container {{
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 0.3rem;
        margin-bottom: 1.5rem;
    }}
    .stButton > button {{
        background-color: rgba(255,255,255,0.05) !important;
        border: none !important;
        color: #111111 !important;
        padding: 0.35rem 0.75rem !important;
        font-size: 0.9rem !important;
        border-radius: 6px !important;
        width: fit-content;
        margin: 0 auto;
        display: block;
    }}
    .stButton > button:hover {{
        background-color: rgba(255,255,255,0.15) !important;
    }}
    mark {{
        background-color: #ffff66;
        font-weight: 600;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# --- Session State Initialisieren ---
if "query_input" not in st.session_state:
    st.session_state.query_input = ""

# --- Header ---
st.markdown("<h1 style='text-align: center;'>Catan Rule Chatbot</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 1.1rem;'>Ask any question about the rules of Catan (including expansions).</p>", unsafe_allow_html=True)

# --- Modellanzeige zuerst ---
model = "llama3-70b-8192"
st.text_input("Model", value=model, disabled=True)

# --- Spielvariante auswählen ---
game_versions = {
    "Base Game": "basegame",
    "Seafarers": "seafarers",
    "Cities & Knights": "cities_knights",
    "Traders & Barbarians": "traders_barbarians",
    "Explorers & Pirates": "explorers_pirates"
}
selected_label = st.selectbox("Game version", list(game_versions.keys()))
selected_variant = game_versions[selected_label]

# --- Beispiel-Fragen ---
examples = [
    "Can I build a settlement directly next to another one?",
    "What happens if I roll a 7 and have too many cards?",
    "What do I need to build a city?"
]

st.markdown("<div class='example-header'>Example sentences:</div>", unsafe_allow_html=True)
st.markdown("<div class='example-container'>", unsafe_allow_html=True)
for idx, example in enumerate(examples):
    if st.button(example, key=f"ex_{idx}"):
        st.session_state.query_input = example
st.markdown("</div>", unsafe_allow_html=True)

# --- Eingabe ---
query = st.text_input("Your question", key="query_input")

# --- Markierungsfunktion ---
def highlight_relevant_sentences(answer: str, context: str) -> str:
    context_sentences = re.split(r'(?<=[.!?])\s+', context)
    answer_keywords = set(re.findall(r'\b\w{5,}\b', answer.lower()))
    highlighted = []
    for sentence in context_sentences:
        lowered = sentence.lower()
        overlap = sum(1 for word in answer_keywords if word in lowered)
        if overlap >= 2:
            highlighted.append(f"<mark>{sentence.strip()}</mark>")
        else:
            highlighted.append(sentence.strip())
    return "<br>".join(highlighted)

# --- Antwort generieren ---
if query:
    with st.spinner("Generating answer..."):
        docs = retrieve(query, top_k=5, variant=selected_variant)
        context = "\n\n".join(docs)

        prompt = f"""Answer the following question based on the given context.

==================== Context =====================
{context}

==================== Question =====================
{query}

==================== Answer ===================="""

        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful expert on the rules of Catan. "
                        "Always prioritize using the rules and context from the currently selected game variant. "
                        "If no matching context is found, you may draw from other Catan rulebooks, "
                        "but make clear that you are referencing a different variant."
                        "Answer in a Catan-Lover tone"
                    )
                },
                {"role": "user", "content": prompt}
            ]
        )

        answer = response.choices[0].message.content.strip()

        st.markdown("### Answer")
        st.write(answer)

        with st.expander("Show retrieved context"):
            highlighted = highlight_relevant_sentences(answer, context)
            st.markdown(
                f"<div style='background-color: rgba(0, 0, 0, 0.05); "
                f"padding: 1rem; border-radius: 8px; color: #111111; "
                f"font-family: sans-serif; font-size: 0.95rem;'>{highlighted}</div>",
                unsafe_allow_html=True
            )