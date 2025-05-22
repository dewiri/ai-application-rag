import streamlit as st

st.set_page_config(page_title="Most Asked Questions", page_icon="❓")

st.markdown("<h1 style='text-align: center;'>Most Asked Questions</h1>", unsafe_allow_html=True)
st.markdown("Here are some frequently asked questions about the rules of Catan:")

questions = [
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

for q in questions:
    st.markdown(f"• **{q}**")