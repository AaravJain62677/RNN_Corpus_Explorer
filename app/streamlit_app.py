import streamlit as st

from src.generate import generate_text


st.title("RNN Corpus Explorer")

prompt = st.text_input(
    "Enter starting text"
)

length = st.slider(
    "Generation Length",
    100,
    1000,
    300
)

temperature = st.slider(
    "Temperature",
    0.1,
    2.0,
    0.8
)

if st.button("Generate"):

    text = generate_text(
        start_text=prompt,
        length=length,
        temperature=temperature
    )

    st.write(text)