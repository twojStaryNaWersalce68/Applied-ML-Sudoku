import os
from sudoku_digitalisation.features.sudoku_preprocessing import DatasetPreprocessor
from sudoku_digitalisation.features.dataset_handler import load_sudoku_dataset
from sudoku_digitalisation.models.CNN import train_cnn, evaluate_cnn

if __name__ == "__main__":
    # # IF THE DATASET HAS NOT BEEN PREPROCESSED YET, USE:
    # handler = load_sudoku_dataset("Lexski/sudoku-image-recognition", hugface=True)
    # preprocessor = DatasetPreprocessor(handler, clip_limit=3, output_size=450)
    # dataset_dict, digit_dataset_dict = preprocessor.dataset_preprocessing()
    # preprocessor.handler.save_all_datasets()

    # IF THE DATASET HAS ALREADY BEEN PREPROCESSED AND SAVED, USE:
    handler = load_sudoku_dataset()
    digit_dataset= handler.datasets['digits']

    cnn, history = train_cnn(digit_dataset['train'], digit_dataset['validation'])
    evaluate_cnn(cnn, history, digit_dataset['test'])

    save_path = os.path.join("sudoku_digitalisation", "data", "models")
    os.makedirs(save_path, exist_ok=True)
    cnn.save(save_path)