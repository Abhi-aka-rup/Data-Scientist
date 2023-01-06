import streamlit as st

st.title("Our Streamlit App")

from PIL import Image

st.subheader("Total Data Science")

image = Image.open('tdslogo.png')
st.image(image, use_column_width=True)
