import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates
from PIL import ImageDraw, Image

st.set_page_config(
    page_title = "Sudoku digitiser",
    page_icon = "🔢",
    layout="wide"
)

if "corner" not in st.session_state:
    st.session_state["corner"] = {"top-left": None, "bottom-left": None, "bottom-right": None, "top-right": None, "cursor": None}

def draw_cross(draw: ImageDraw.Draw, coords: tuple[int,int]):
    cross_radius = 10
    draw.line((coords[0]-10, coords[1], coords[0]+10, coords[1]), width = 2, fill = "black")
    draw.line((coords[0], coords[1]-10, coords[0], coords[1]+10), width = 2, fill = "black")

col1, col2 = st.columns([0.7,0.3])

with col1:
    with Image.open("sudoku_uncropped.jpg") as img:

        draw = ImageDraw.Draw(img)
        corners = st.session_state["corner"]

        cursor = corners["cursor"]
        if cursor is not None:
            draw_cross(draw, cursor)

        value = streamlit_image_coordinates(img, key = "pil")
        
        if value is not None:
            raw_value = st.session_state["pil"]
            point = raw_value["x"], raw_value["y"]
            st.session_state["corner"]["cursor"] = point

with col2:
    if st.button("top left"):
        corners["top left"] = corners["cursor"]
    if st.button("top right"):
        corners["top right"] = corners["cursor"]
    if st.button("bottom left"):
        corners["bottom left"] = corners["cursor"]
    if st.button("bottom right"):
        corners["bottom right"] = corners["cursor"]

# if st.button("crop sudoku"):
#     crop_sudoku()

# if st.button("digitise sudoku"):
#     #digitise that shit

# if st