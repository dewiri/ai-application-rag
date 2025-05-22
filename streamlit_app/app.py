import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from src.retrieval import retrieve
from src.api_client import client
import base64

# Page config
st.set_page_config(page_title="Catan Rule Expert", page_icon=None)

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
    label, .stTextInput label {{
        color: #111111 !important;
        font-weight: 600;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Title
st.markdown("<h1 style='text-align: center;'>Catan Rule Chatbot</h1>", unsafe_allow_html=True)
st.markdown("Ask any question about the rules of Catan (including expansions).")

# Fixed model display (not editable)
st.text_input("Model", value="llama3-70b-8192", disabled=True)

# Input field
query = st.text_input("Your question")

# Most asked questions under input
example_questions = [
    "Can I build a settlement directly next to another one?",
    "What happens when the robber is moved?",
    "How many victory points do I need to win?",
    "Can I trade development cards?",
    "What is the longest road and how do I get it?",
    "Can I build multiple roads in one turn?",
    "How do harbors work in Seafarers?",
    "What happens if I roll a 7 and have too many cards?",
    "Do cities count as two settlements for building restrictions?",
    "How do I use knights in Cities & Knights?"
]

selected_example = st.selectbox("Most asked questions (optional):", [""] + example_questions)
if selected_example:
    query = selected_example
    st.experimental_rerun()

# Fixed model
model = "llama3-70b-8192"

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

        # Answer output
        st.markdown("### Answer")
        st.write(answer)

        # Context display
        with st.expander("Show retrieved context"):
            st.markdown(
                f"<div style='background-color: rgba(0, 0, 0, 0.05); "
                f"padding: 1rem; border-radius: 8px; color: #111111; "
                f"font-family: sans-serif; font-size: 0.95rem;'>{context.replace(chr(10), '<br>')}</div>",
                unsafe_allow_html=True
            )