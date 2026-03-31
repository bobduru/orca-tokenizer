import os
import streamlit as st

APP_ENV = os.environ.get("APP_ENV", "dev")

st.set_page_config(
    page_title="Orca Call Catalogue",
    page_icon="🐋",
    layout="wide"
)

catalogue = st.Page("pages/0_Catalogue.py", title="Catalogue", icon="🐋", default=True)

if APP_ENV == "prod":
    pages = [catalogue]
else:
    pages = [
        catalogue,
        st.Page("pages/1_Label_Calls.py", title="Label Calls", icon="🏷️"),
        st.Page("pages/2_Call_Query.py", title="Call Query", icon="🔍"),
    ]

st.navigation(pages).run()
