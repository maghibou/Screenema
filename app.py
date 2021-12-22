import streamlit as st
import home as home
from PIL import Image
import os

page_icon = st.image(os.path.join('assets', "Icon plus+.jpg"))


st.set_page_config(
    page_title='SCREENEMA',
    page_icon=page_icon,
    layout='wide'
)

# pages_dict = {
#     'home': home.render
# }

# selected_page = st.sidebar.selectbox("Go to page:", options=pages_dict)

home.render()
# pages_dict[selected_page]()