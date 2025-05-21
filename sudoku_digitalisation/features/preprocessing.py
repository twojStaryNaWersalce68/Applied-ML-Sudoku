import datasets
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image, ImageOps
from tqdm import tqdm
from collections import Counter
from datasets import load_dataset, DatasetDict, Dataset

import keras
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay


def load_data(path):
    """Loads a dataset from huggingface.com."""
    data = load_dataset(path)
    return data


def show_image(image):
    """Shows an image from the dataset."""
    plt.imshow(image, cmap="gray")
    plt.show()


def convert_to_grayscale(ds):
    """Converts the image to grayscale."""
    grayscale_image = ImageOps.grayscale(ds['image'])
    ds['image'] = grayscale_image
    return ds


def crop_to_bounding_box(ds, output_size=(450, 450)):
    """
    Crops the image to the bounding box. It does this by warping the image so it becomes a square.
    This also normalizes image sizes and makes images RGB.

    This is done using the "keypoints". They are a list of 8 values, which represent four coordinates.
    The keypoints are ordered: top-left, bottom-left, bottom-right, top-right.
    Some keypoints may be outside the bounds of the image.
    """
    keypoints = ds['keypoints']
    bounding_box = np.array([
        [keypoints[0], keypoints[1]],  # top left
        [keypoints[2], keypoints[3]],  # bottom left
        [keypoints[4], keypoints[5]],  # bottom right
        [keypoints[6], keypoints[7]]   # top right
    ], dtype=np.float32)

    desired_corner_points = np.array([
        [0, 0],  # top left
        [0, output_size[1] - 1],  # bottom left
        [output_size[0] - 1, output_size[1] - 1],  # bottom right
        [output_size[0] - 1, 0]  # top right
    ], dtype=np.float32)

    image = np.array(ds['image'].convert('RGB'))
    pers_transform_matrix = cv2.getPerspectiveTransform(bounding_box, desired_corner_points)
    warped_image = cv2.warpPerspective(image, pers_transform_matrix, output_size)
    ds['image'] = Image.fromarray(warped_image)

    return ds


def adaptive_histogram_equalization(ds):
    """
    Applies adaptive histogram equalization to the image.
    This is done using CLAHE algorithm.
    """
    image = np.array(ds['image'])
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized_image = clahe.apply(image)
    ds['image'] = Image.fromarray(equalized_image)
    return ds


def split_image(image):
    """
    Splits the image into 81 cells.
    """
    image = np.array(image)
    cells = []
    cell_size = (image.shape[0] // 9, image.shape[1] // 9)
    for i in range(9):
        for j in range(9):
            x_start = i * cell_size[0]
            x_end = (i + 1) * cell_size[0]
            y_start = j * cell_size[1]
            y_end = (j + 1) * cell_size[1]
            cell = image[x_start:x_end, y_start:y_end]
            cells.append(Image.fromarray(cell))
    return cells


def change_label(label):
    """Changes the label of a single cell from binary representation to the simple integer."""
    label[0] = abs(label[0] - 1)  # changes 0 to 1 and 1 to 0, such that below if the cell is unsolved it returns 0
    for idx, item in enumerate(label):
        if item == 1:
            return idx
    return 0


def split_labels(labels):
    """Splits the labels into a list of 81 labels."""
    new_labels = []
    for i in range(9):
        for j in range(9):
            new_label = change_label(labels[i][j])
            new_labels.append(new_label)
    return new_labels


def create_digit_ds(ds):
    """Sets up the datasets which we use for training."""
    new_ds = []
    for idx, row in enumerate(ds):
        digit_images = split_image(row['image'])
        digit_labels = split_labels(row['cells'])
        for j in range(len(digit_labels)):
            new_ds.append({"digit_img": digit_images[j], "label": digit_labels[j], "index": idx})
    return new_ds
    

def preprocess_dataset(ds_dict):
    """Preprocesses the dataset."""
    digit_ds_dict = DatasetDict()
    for split in ds_dict:
        print(split)
        ds_dict[split] = ds_dict[split].map(crop_to_bounding_box)
        ds_dict[split] = ds_dict[split].map(convert_to_grayscale)
        ds_dict[split] = ds_dict[split].map(adaptive_histogram_equalization)

        digit_ds = Dataset.from_list(create_digit_ds(ds_dict[split]))
        digit_ds_dict[split] = digit_ds

    return ds_dict, digit_ds_dict


# Preprocessing for CNN
def make_dataset(img_list):
    N = len(img_list)
    out = np.zeros((N, 28, 28, 1), dtype=np.float32)
    for i, img in enumerate(img_list):
        arr = np.array(img)
        resized = cv2.resize(arr, (28, 28), interpolation=cv2.INTER_AREA)
        out[i, :, :, 0] = resized.astype(np.float32)
    return out


if __name__ == '__main__':
    dataset_dict = load_data("Lexski/sudoku-image-recognition")
    dataset_dict, digit_dataset_dict = preprocess_dataset(dataset_dict)
    # digit_dataset_dict is what we use for training



    ### Convert labels for CNN ###
    train_labels = np.array(digit_dataset_dict["train"]["label"])
    val_labels = np.array(digit_dataset_dict["validation"]["label"])
    test_labels = np.array(digit_dataset_dict["test"]["label"])

    # grab each split separately
    train_imgs = digit_dataset_dict["train"]["digit_img"]
    val_imgs   = digit_dataset_dict["validation"]["digit_img"]
    test_imgs  = digit_dataset_dict["test"]["digit_img"]

    train_images = make_dataset(train_imgs)
    val_images   = make_dataset(val_imgs)
    test_images  = make_dataset(test_imgs)


    # One-hot encoding
    train_labels = keras.utils.to_categorical(train_labels, num_classes=10)
    val_labels = keras.utils.to_categorical(val_labels, num_classes=10)
    test_labels = keras.utils.to_categorical(test_labels, num_classes=10)

    # CNN model: 2 conv layers, 2 max pooling layers, 2 dense layers
    cnn = keras.models.Sequential()

    cnn.add(keras.layers.Conv2D(filters=32, kernel_size=(3, 3), input_shape=(28, 28, 1), activation='relu'))
    cnn.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

    cnn.add(keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    cnn.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

    cnn.add(keras.layers.Flatten())

    cnn.add(keras.layers.Dense(activation='relu', units=128))
    cnn.add(keras.layers.Dropout(0.2))

    cnn.add(keras.layers.Dense(units=128, activation='relu'))
    cnn.add(keras.layers.Dropout(0.2))

    # Output layer
    cnn.add(keras.layers.Dense(units=10, activation='softmax'))

    cnn.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall')
        ]
    )

    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    history = cnn.fit(
        train_images, train_labels,
        epochs=100,
        verbose=1,
        batch_size=128,
        validation_data=(val_images, val_labels),
        callbacks=[early_stopping]
    )

    # Convert one-hot encoded data back to normal labels
    y_pred_probs = cnn.predict(test_images)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(test_labels, axis=1)

    # Per class precision, recall and F1
    report = classification_report(y_true, y_pred, target_names=[f'Class {i}' for i in range(10)])
    print(report)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(len(cm)))
    disp.plot(cmap='viridis', values_format='d')
    plt.title('Confusion Matrix')
    plt.show()

    # Per class accuracy
    class_accuracies = cm.diagonal() / cm.sum(axis=1)
    for i, acc in enumerate(class_accuracies):
        print(f"Accuracy for Class {i}: {acc:.4f}")

    # Evaluate the cnn on the test set
    test_loss, test_accuracy, test_precision, test_recall = cnn.evaluate(test_images, test_labels, verbose=2)
    print(f"Test loss: {test_loss}")
    print(f"Test accuracy: {test_accuracy}")

    # Accuracy plot over time
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training vs Validation Accuracy')
    plt.show()

    # Loss plot over time
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training vs Validation Loss')
    plt.show()

    # Save the model
    cnn.save('sudoku_cnn_model.keras')