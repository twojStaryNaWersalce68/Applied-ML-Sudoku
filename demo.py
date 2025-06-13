import streamlit as st
from PIL import Image
from sudoku_digitalisation.features.image_operations import ImageConverter
from sudoku_digitalisation.features.sudoku_preprocessing import DatasetPreprocessor
from sudoku_digitalisation.features.dataset_handler import DatasetHandler
from sudoku_digitalisation.scripts.run_training import get_model
from sudoku_digitalisation.scripts.run_prediction import make_prediction

st.set_page_config(
    page_title = "Sudoku digitiser",
    page_icon = "🔢",
    layout="wide"
)

if "uploaded" not in st.session_state:
    st.session_state["uploaded"] = None


st.title("Sudoku Extractor")

st.write("Upload an image of your sudoku, select where the corners are and then we will extract the sudoku so you can copy the 9x9 matrix.")

st.session_state["uploaded"] = st.file_uploader("upload your sudoku here, file type: png, jpg, jpeg")

sudoku_file = st.session_state["uploaded"]

col1, col2 = st.columns([0.7,0.3])
if sudoku_file:
    with Image.open(sudoku_file) as img:
        with col1:
            st.image(img)

        with col2:
            converter = ImageConverter(3)
            clahe_img = converter.apply_clahe(img)

            if st.button("extract sudoku"):
                handler = DatasetHandler()
                preprocessor = DatasetPreprocessor(handler)
                cnn = get_model(False, "cnn", "32_64_128", preprocessor)
                large_digits, candidate_digits = make_prediction(clahe_img, preprocessor, cnn)
                for i in range(9):
                    st.markdown(candidate_digits[i*9:i*9+9])

# streamlit run demo.py
