import streamlit as st
import home
from PIL import Image

page_icon = Image.open('assets/istockphoto-1268952827-170667a.jpg')


st.set_page_config(
    page_title='My Cool App',
    page_icon=page_icon,
    layout='wide'
)

# pages_dict = {
#     'home': home.render
# }

# selected_page = st.sidebar.selectbox("Go to page:", options=pages_dict)

home.render()
# pages_dict[selected_page]()