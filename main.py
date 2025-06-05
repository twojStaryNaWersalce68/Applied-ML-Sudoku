from keras import Sequential
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


def get_tiny_digits():
    sudoku_sample = Image.open(r"C:\Users\Tabea\Pictures\sudoku_small_numbers.jpg")
    plt.imshow(sudoku_sample, cmap="gray")
    #plt.show()
    preprocessor = SudokuPreprocessor(clip_limit=3, output_size=450)
    _, digit_dataset = preprocessor.sudoku_preprocessing(sudoku_sample)

    # focus on one cell for now
    cell_sample = digit_dataset[6]  # 6 3 8/16 1
    cell_sample = np.array(cell_sample)
    plt.imshow(cell_sample, cmap="gray")
    plt.title("Cell Raw")
    plt.show()

    h, w = cell_sample.shape
    print(cell_sample.shape)
    margin = 2
    cell_sample = cell_sample[margin:(h-margin), margin:(w-margin)]
    cell_sample = cv2.copyMakeBorder(src=cell_sample, top=margin, bottom=margin, left=margin, right=margin, borderType=cv2.BORDER_CONSTANT, value=(255, 255, 255))

    print(cell_sample.shape)

    plt.imshow(cell_sample, cmap="gray")
    plt.title("Cell Cropped")
    plt.show()

    # find contours of the small numbers
    cell_img = cell_sample.copy()
    _, binary_sample = cv2.threshold(cell_sample, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, hierarchy = cv2.findContours(binary_sample, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print("num of contours", len(contours))
    bounding_box = []
    print("areas")
    for contour in contours:
        area = cv2.contourArea(contour)
        print(area)
        if 25 <= area <= 300:
            x, y, w, h = cv2.boundingRect(contour)
            bounding_box.append([x, y, w, h])
            cv2.rectangle(cell_sample, (x, y), (x + w, y + h), (0, 255, 0), 1)
    print("num of bb", len(bounding_box))
    # print(bounding_box)

    plt.imshow(binary_sample, cmap="gray")
    plt.show()

    plt.imshow(cell_sample, cmap="gray")
    plt.show()

    tiny_digits = []
    for bbox in bounding_box:
        x, y, w, h = bbox
        centre_x = x + (w / 2)
        centre_y = y + (h / 2)
        size = max(w, h)
        x = int(centre_x - size / 2)
        y = int(centre_y - size / 2)
        cropped_image = cell_img[y:y + size, x:x + size]
        cropped_image = cv2.resize(cropped_image, (28, 28))
        tiny_digits.append(Image.fromarray(cropped_image))

        plt.imshow(cropped_image, cmap="gray")
        #plt.show()

    return tiny_digits



if __name__ == "__main__":
    preprocessor = get_preprocessed_dataset(is_preprocessed=True) # change this to False if the dataset hasn't been processed and saved
    cnn = get_model(preprocessor, False)

    #if test_image:
        #predict_sudoku(cnn, test_image)
    tiny_digits = get_tiny_digits()
    if len(tiny_digits) > 0:
        predictions = cnn.predict(tiny_digits)
        for prediction in predictions:
            print(np.argmax(prediction))
    else:
        print("no digits found")