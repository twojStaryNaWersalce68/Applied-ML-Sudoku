from PIL import Image
from io import BytesIO
from pydantic import BaseModel
from typing import List
from fastapi.responses import JSONResponse
from fastapi import FastAPI, File, UploadFile, HTTPException
from sudoku_digitalisation.scripts.run_prediction import make_prediction
from sudoku_digitalisation.scripts.run_training import get_model
from sudoku_digitalisation.features.sudoku_preprocessing import get_preprocessor

app = FastAPI(
    title = "Sudoku digitizer",
    summary = "an API that takes in a sudoku and give the corresponding 9x9 representing the sudoku",
    description = """
# Model usage
The model is a CNN was trained on a combination of digits in different fonts and some handwritten.
It takes these images by cropping sudokus that were fed to it.
The API takes in an already cropped image of a sudoku and returns a 9x9 matrix representing the numbers inside the sudoku, 0 means an empty square.

## Limitations
The sudoku goes through the process of edge detection, and predictions with 
the CNN, and as there are 81 square in a sudoku it is pretty hard to get fully
correct sudokus. Either the edge detection goes wrong, or there is one digit
the CNN doesn't classify properly.""",
version = "alpha"
)

# Constants
MODEL_NAME = "32_64_128"
OUTPUT_SIZE = 450

class SudokuPredictions(BaseModel):
    predictions: List[List[int]]

# Load the model once when FastAPI starts
preprocessor = get_preprocessor(output_size=OUTPUT_SIZE, is_preprocessed=True)
cnn_model = get_model(False, 'cnn', MODEL_NAME, preprocessor)


@app.post("/predict/", description = "Sudoku digitizer endpoint. Upload picture of already cropped sudoku."
                                    " Picture has to be .png, .jpg or .jpeg."
                                    " Returns list of lists, where each list represents a cell.",
                        response_model = SudokuPredictions,
                        response_description = "Digitised version of uploaded sudoku, in the form of a 9x9 matrix.")
async def predict(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".png", ".jpg", ".jpeg")):
        raise HTTPException(status_code=400, detail="Only image files (.png, .jpg, .jpeg) are accepted")

    try:
        image_bytes = await file.read()
        image = Image.open(BytesIO(image_bytes)).convert("L")  # convert to grayscale

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process image: {e}")

    try:
        large_digits, candidate_digits = make_prediction(image, preprocessor, cnn_model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    return JSONResponse(content={"sudoku_grid": candidate_digits})

# Run with: uvicorn fastAPI:app --reload
