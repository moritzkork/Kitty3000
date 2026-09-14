# MK: Router. st.navigation gives us a proper top nav bar instead of the ugly sidebar page list.
# MK: The actual pages live in app/views/ ; each is a normal Streamlit script.

import streamlit as st

st.set_page_config(page_title="Kitty3000", page_icon="🐱", layout="wide")

home = st.Page("views/home.py", title="Home", icon="🐱", default=True)
analysis = st.Page("views/analysis.py", title="Analysis", icon="🔬")

nav = st.navigation([home, analysis], position="top")
nav.run()
