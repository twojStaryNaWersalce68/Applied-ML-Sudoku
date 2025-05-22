import os
from sudoku_digitalisation.features.sudoku_preprocessing import DatasetPreprocessor
from sudoku_digitalisation.features.dataset_handler import load_sudoku_dataset
from sudoku_digitalisation.models.CNN import train_cnn, evaluate_cnn
from sudoku_digitalisation.models.SVM import SVM

if __name__ == "__main__":
    # # IF THE DATASET HAS NOT BEEN PREPROCESSED YET, USE:
    # handler = load_sudoku_dataset("Lexski/sudoku-image-recognition", hugface=True)
    # preprocessor = DatasetPreprocessor(handler, clip_limit=3, output_size=252)
    # dataset_dict, digit_dataset = preprocessor.dataset_preprocessing()
    # preprocessor.handler.save_all_datasets()

    # IF THE DATASET HAS ALREADY BEEN PREPROCESSED AND SAVED, USE:
    handler = load_sudoku_dataset()
    preprocessor = DatasetPreprocessor(handler, clip_limit=3, output_size=252)
    digit_dataset= handler.datasets['digits']

    X_train = digit_dataset['train']['image']
    y_train = digit_dataset['train']['label']

    X_test = digit_dataset['test']['image']
    y_test = digit_dataset['test']['label']

    svm = SVM()
    svm.train(X_train[:8000], y_train[:8000])
    svm.evaluate(X_test, y_test)

    cnn, history = train_cnn(digit_dataset['train'], digit_dataset['validation'])
    evaluate_cnn(cnn, history, digit_dataset['test'])

    folder_path = os.path.join("sudoku_digitalisation", "data", "models")
    os.makedirs(folder_path, exist_ok=True)
    model_name = "cnn_test.keras"
    save_path = os.path.join(folder_path, model_name)
    cnn.save(save_path)