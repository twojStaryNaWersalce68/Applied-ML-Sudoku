from keras import Sequential
from keras.src.ops import threshold

from sudoku_digitalisation.features.cell_splitter import CellSplitter
from sudoku_digitalisation.features.sudoku_preprocessing import DatasetPreprocessor, SudokuPreprocessor
from sudoku_digitalisation.features.dataset_handler import load_sudoku_dataset
from sudoku_digitalisation.models.CNN import CNN
from sudoku_digitalisation.models.SVM import SVM
import matplotlib.pyplot as plt
import numpy as np
import argparse
from PIL import Image
import cv2


def get_preprocessed_dataset(is_preprocessed: bool = False):
    print("Getting preprocessed dataset")
    # fetches the dataset from local source, if it is already preprocessed
    if is_preprocessed:
        handler = load_sudoku_dataset()
        preprocessor = DatasetPreprocessor(handler, clip_limit=3, output_size=252)
    else:
        handler = load_sudoku_dataset("Lexski/sudoku-image-recognition", hugface=True)
        preprocessor = DatasetPreprocessor(handler, clip_limit=3, output_size=252)
        preprocessor.handler.save_all_datasets()
    return preprocessor


def get_model(preprocessor: DatasetPreprocessor, train_model: bool = False):
    print("Getting model")
    digit_dataset = preprocessor.handler.datasets['digits']
    if train_model:
        X_train = digit_dataset['train']['image']
        y_train = digit_dataset['train']['label']

        X_val = digit_dataset['validation']['image']
        y_val = digit_dataset['validation']['label']

        X_test = digit_dataset['test']['image']
        y_test = digit_dataset['test']['label']

        sudoku_height = preprocessor.cropper.output_size // 9

        svm = SVM(input_shape=(sudoku_height, sudoku_height), verbose=True)
        svm.train(X_train[:10000], y_train[:10000])
        svm.evaluate(X_test, y_test)

        cnn = CNN(input_shape=(sudoku_height, sudoku_height, 1), num_classes=10)
        cnn.train(X_train, y_train, X_val, y_val, verbose=1)
        cnn.evaluate(X_test, y_test)
        cnn.save("sudoku_cnn")
    else:
        sudoku_height = preprocessor.cropper.output_size // 9
        cnn = CNN(input_shape=(sudoku_height, sudoku_height, 1), num_classes=10)
        cnn.load("sudoku_cnn")
    return cnn


def predict_sudoku(cnn: Sequential, sudoku_sample: Image.Image):
    print("Predicting")
    preprocessor = SudokuPreprocessor(clip_limit=3, output_size=252)
    _, digit_dataset = preprocessor.sudoku_preprocessing(sudoku_sample)

    predictions = cnn.predict(digit_dataset)
    sudoku_labels = []
    dimension = int(np.sqrt(len(digit_dataset)))
    for i in range(dimension):
        row = []
        for j in range(dimension):
            label = np.argmax(predictions[j + (dimension * i)])
            row.append(int(label))
        sudoku_labels.append(row)

    for row in sudoku_labels:
        print(row)

    plt.imshow(sudoku_sample, cmap="gray")
    plt.show()


if __name__ == "__main__":
    preprocessor = get_preprocessed_dataset(is_preprocessed=True) # change this to False if the dataset hasn't been processed and saved
    cnn = get_model(preprocessor, False)

    sudoku_sample = Image.open(r"C:\Users\Tabea\Pictures\sudoku_small_numbers1.jpg")
    plt.imshow(sudoku_sample, cmap="gray")
    plt.show()
    preprocessor = SudokuPreprocessor(clip_limit=3, output_size=450)
    _, digit_dataset = preprocessor.sudoku_preprocessing(sudoku_sample)

    for idx in range(81):
    #idx = 3
        tiny_digits = CellSplitter.get_candidate_digits(digit_dataset[idx])
            #get_tiny_digits(digit_dataset[idx])) # 6 3 8/16 1 1

        if len(tiny_digits) > 0:
            predictions = cnn.predict(tiny_digits)
            cell_labels_binary = [0] * 10  # format for computer
            for i, prediction in enumerate(predictions):
                label = np.argmax(prediction)
                #print(label)
                #plt.imshow(tiny_digits[i], cmap="gray")
                #plt.title(label)
                #plt.show()
                if 0 <= label <= 9:
                    cell_labels_binary[label] = 1
            #print(cell_labels_binary)

            cell_labels = []  # format for humans
            for i in range(len(cell_labels_binary)):
                if cell_labels_binary[i] == 1:
                    cell_labels.append(i)
            print(idx, cell_labels)
            plt.imshow(digit_dataset[idx], cmap="gray")
            plt.title(f"{idx}: {cell_labels}")
            plt.show()

        else:
            print("no digits found")