import numpy as np
import keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay


if __name__ == "__main__":
    (train_images, train_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()

    # Validation split
    train_images, val_images, train_labels, val_labels = train_test_split(
        train_images, train_labels, test_size=0.2, random_state=42
    )

    # Scales to [0, 1]
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    val_images = val_images / 255.0

    # Reshape images to (28, 28, 1) for CNN input
    train_images = train_images.reshape(-1, 28, 28, 1)
    val_images = val_images.reshape(-1, 28, 28, 1)
    test_images = test_images.reshape(-1, 28, 28, 1)

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

    cnn.add(keras.layers.Dense(filters=128, activation='relu'))
    cnn.add(keras.layers.Dropout(0.2))

    cnn.add(keras.layers.Dense(filters=64, activation='relu'))
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

    # cnn.save("fashion_mnist_cnn.h5")

    # cnn = keras.models.load_model("fashion_mnist_cnn.h5")
    # keras.utils.plot_model(cnn, show_dtype=True, show_shapes=True)

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
