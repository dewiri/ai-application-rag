import sys
import os
import base64
import streamlit as st
from src.retrieval import retrieve
from src.api_client import client

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

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
    </style>
    """,
    unsafe_allow_html=True
)

# Title
st.markdown("<h1 style='text-align: center;'>Catan Rule Chatbot</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; font-size: 1.1rem;'>Ask any question about the rules of Catan (including expansions).</p>",
    unsafe_allow_html=True
)   

# Fixed model display (not editable)
model = "llama3-70b-8192"
st.text_input("Model", value=model, disabled=True)

# Input field
query = st.text_input("Your question")

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

        # Output
        st.markdown("### Answer")
        st.write(answer)

        # Context viewer
        with st.expander("Show retrieved context"):
            st.markdown(
                f"<div style='background-color: rgba(0, 0, 0, 0.05); "
                f"padding: 1rem; border-radius: 8px; color: #111111; "
                f"font-family: sans-serif; font-size: 0.95rem;'>{context.replace(chr(10), '<br>')}</div>",
                unsafe_allow_html=True
            )