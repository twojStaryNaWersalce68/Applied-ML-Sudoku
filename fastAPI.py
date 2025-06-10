import os
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
from io import BytesIO
from sudoku_digitalisation.features.sudoku_preprocessing import SudokuPreprocessor
from sudoku_digitalisation.models.CNN import CNN
from pydantic import BaseModel
from typing import List

from sudoku_digitalisation.scripts.run_prediction import make_prediction

app = FastAPI(
    title = "Sudoku digitizer",
    summary = "an API that takes in a sudoku and give the corresponding 9x9 representing the sudoku",
    description = """
# Model usage
The model is a CNN was trained on a combination of digits in different fonts and some handwritten.
It takes these images by cropping sudokus that were fed to it.
The API takes in an already cropped image of a sudoku and returns a 9x9 matrix representing the numbers inside the sudoku, 0 means an empty square.

## Limitations
The model cannot predict the small numbers that are used in sudokus, 
however these should not impact the prediction.""",
version = "alpha"
)

# Constants
MODEL_NAME = "32_64_128"
OUTPUT_SIZE = 450

class SudokuPredictions(BaseModel):
    predictions: List[List[int]]


def load_model(model_name=MODEL_NAME, output_size=OUTPUT_SIZE):
    print("Loading pre-trained model...")
    sudoku_height = output_size // 9
    cnn = CNN(input_shape=(sudoku_height, sudoku_height, 1), num_classes=10)
    cnn.load(model_name)
    return cnn


def predict_sudoku(cnn, image: Image.Image, output_size=OUTPUT_SIZE):
    print("Preprocessing and predicting...")
    preprocessor = SudokuPreprocessor(clip_limit=3, output_size=output_size)
    sudoku_labels = make_prediction(image, preprocessor, cnn)

    return sudoku_labels


# Load the model once when FastAPI starts
cnn_model = load_model()


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    print("Received request")

    try:
        image_bytes = await file.read()
        print(f"Image bytes read: {len(image_bytes)} bytes")

        image = Image.open(BytesIO(image_bytes)).convert("L")
        print("Image successfully opened")
    except Exception as e:
        print(f"Image load error: {e}")
        raise HTTPException(status_code=400, detail=f"Image error: {e}")

    return {"message": "Image loaded successfully"}

# Run with: uvicorn fastAPI:app --reload