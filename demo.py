import streamlit as st
import numpy as np
from streamlit_image_coordinates import streamlit_image_coordinates
from PIL import ImageDraw, Image
from sudoku_digitalisation.features.image_operations import ImageCropper, ImageConverter
from sudoku_digitalisation.features.sudoku_preprocessing import DatasetPreprocessor
from sudoku_digitalisation.features.dataset_handler import DatasetHandler
from sudoku_digitalisation.scripts.run_training import get_model
from sudoku_digitalisation.scripts.run_prediction import make_prediction

st.set_page_config(
    page_title = "Sudoku digitiser",
    page_icon = "🔢",
    layout="wide"
)

if "corner" not in st.session_state:
    st.session_state["corner"] = [None, None, None, None, None]
if "cropped" not in st.session_state:
    st.session_state["cropped"] = False

def draw_cross(draw: ImageDraw.Draw, coords: tuple[int,int], cursor: bool):
    cross_radius = 10
    draw.line((coords[0]-10, coords[1], coords[0]+10, coords[1]), width = 2, fill = "grey" if cursor else "black")
    draw.line((coords[0], coords[1]-10, coords[0], coords[1]+10), width = 2, fill = "grey" if cursor else "black")

col1, col2 = st.columns([0.7,0.3])

with Image.open("sudoku_uncropped.jpg") as img:
    with col1:

        draw = ImageDraw.Draw(img)
        corners = st.session_state["corner"]

        cursor = corners[-1]
        if cursor is not None:
            draw_cross(draw, cursor, True)
        for point in corners[0:4]:
            if point:
                draw_cross(draw, point, False)

        value = streamlit_image_coordinates(img, key = "pil")
        
        if value is not None:
            raw_value = st.session_state["pil"]
            point = raw_value["x"], raw_value["y"]
            st.session_state["corner"][-1] = point

    with col2:
        if st.button("top left"):
            corners[0] = corners[4]
        if st.button("top right"):
            corners[1] = corners[4]
        if st.button("bottom left"):
            corners[2] = corners[4]
        if st.button("bottom right"):
            corners[3] = corners[4]

    if st.button("crop sudoku"):
        st.session_state["cropped"] = True
    
    if st.session_state["cropped"] == True:
        cropper = ImageCropper(450)
        converter = ImageConverter(3)
        corners_array = np.array([
                [corners[0][0], corners[0][1]],  # top left
                [corners[2][0], corners[2][1]],  # bottom left
                [corners[3][0], corners[3][1]],  # bottom right
                [corners[1][0], corners[1][1]]   # top right
            ], dtype=np.float32)
        clahe_img = converter.apply_clahe(img)
        cropped_img = cropper.crop_to_box(image=clahe_img, bounding_box=corners_array)
        st.image(cropped_img)

        if st.button("digitise sudoku"):
            handler = DatasetHandler()
            preprocessor = DatasetPreprocessor(handler)
            cnn = get_model(False, "cnn", "32_64_128", preprocessor)
            digitised_sudoku = make_prediction(cropped_img, preprocessor, cnn)
            st.write(digitised_sudoku)
