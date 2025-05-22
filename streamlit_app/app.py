import sys
import os
import base64
import streamlit as st

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.retrieval import retrieve
from src.api_client import client

# Page setup
st.set_page_config(page_title="Catan Rule Expert", page_icon="🎲")

# Background image
def get_base64_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

bg_image = get_base64_image("streamlit_app/assets/catan_bg.jpg")

# Custom styling
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
        margin: 0 auto;  /* ✅ zentriert den Button horizontal */
        display: block;  /* ✅ nötig für zentrierung mit margin */
    }}
    .stButton > button:hover {{
        background-color: rgba(255,255,255,0.15) !important;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Session state
if "query_input" not in st.session_state:
    st.session_state.query_input = ""

# Title and subtitle
st.markdown("<h1 style='text-align: center;'>Catan Rule Chatbot</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; font-size: 1.1rem;'>Ask any question about the rules of Catan (including expansions).</p>",
    unsafe_allow_html=True
)

# Example questions
examples = [
    "Can I build a settlement directly next to another one?",
    "What happens if I roll a 7 and have too many cards?",
    "How do knights work in Cities & Knights?"
]

# Display example buttons centered
st.markdown("<div class='example-header'>Example sentences:</div>", unsafe_allow_html=True)
st.markdown("<div class='example-container'>", unsafe_allow_html=True)
for idx, example in enumerate(examples):
    if st.button(example, key=f"ex_{idx}"):
        st.session_state.query_input = example
st.markdown("</div>", unsafe_allow_html=True)

# Fixed model display
model = "llama3-70b-8192"
st.text_input("Model", value=model, disabled=True)

# Input field
query = st.text_input("Your question", value=st.session_state.query_input, key="query_input")

# Answer generation
if query:
    with st.spinner("Generating answer..."):
        docs = retrieve(query, top_k=5)
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
                {"role": "system", "content": "You are a helpful expert on the rules of Catan."},
                {"role": "user", "content": prompt}
            ]
        )

        answer = response.choices[0].message.content.strip()

        st.markdown("### Answer")
        st.write(answer)

        with st.expander("Show retrieved context"):
            st.markdown(
                f"<div style='background-color: rgba(0, 0, 0, 0.05); "
                f"padding: 1rem; border-radius: 8px; color: #111111; "
                f"font-family: sans-serif; font-size: 0.95rem;'>{context.replace(chr(10), '<br>')}</div>",
                unsafe_allow_html=True
            )