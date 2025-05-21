import numpy as np
import keras
import matplotlib.pyplot as plt
import cv2
from datasets import Dataset
from typing import Tuple
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

def reshape_image(img):
    arr = np.array(img)
    resized = cv2.resize(arr, (28, 28), interpolation=cv2.INTER_AREA)
    return resized

# Preprocessing for CNN
def make_dataset(img_list):
    N = len(img_list)
    out = np.zeros((N, 28, 28, 1), dtype=np.float32)
    for i, img in enumerate(img_list):
        resized = reshape_image(img)
        out[i, :, :, 0] = resized.astype(np.float32)
    return out

def train_cnn(train_data: Dataset, val_data: Dataset) -> Tuple[keras.models.Sequential, keras.callbacks.History]:
    ### Convert labels for CNN ###
    train_labels = np.array(train_data["label"])
    val_labels = np.array(val_data["label"])

    # grab each split separately
    train_imgs = train_data["image"]
    val_imgs   = val_data["image"]

    train_images = make_dataset(train_imgs)
    val_images   = make_dataset(val_imgs)

    # One-hot encoding
    train_labels = keras.utils.to_categorical(train_labels, num_classes=10)
    val_labels = keras.utils.to_categorical(val_labels, num_classes=10)

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

    return cnn, history

def evaluate_cnn(cnn: keras.models.Sequential,
                 history: keras.callbacks.History, 
                 test_data: Dataset) -> None:
    
    test_images = make_dataset(test_data['image'])
    y_true = np.array(test_data['label'])
    test_labels = keras.utils.to_categorical(y_true, num_classes=10)

    # Convert one-hot encoded data back to normal labels
    y_pred_probs = cnn.predict(test_images)
    y_pred = np.argmax(y_pred_probs, axis=1)

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
    