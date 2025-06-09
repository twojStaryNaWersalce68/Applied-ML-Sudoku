from sklearn.model_selection import KFold
import numpy as np
import pandas as pd

def evaluate_model(model, preprocessor, k=5, val_split=0.1, is_cnn=False):
    """
    Perform k-fold cross-validation and return mean and std of metrics.
    """
    kf = KFold(n_splits=k, shuffle=True)
    results = []

    images = np.array(images)
    labels = np.array(labels)

    for train_index, test_index in kf.split(images):
        X_train_full, X_test = images[train_index], images[test_index]
        y_train_full, y_test = labels[train_index], labels[test_index]

        if is_cnn:
            # Split train into train/val
            val_size = int(len(X_train_full) * val_split)
            X_val = X_train_full[:val_size]
            y_val = y_train_full[:val_size]
            X_train = X_train_full[val_size:]
            y_train = y_train_full[val_size:]

            model.train(list(X_train), list(y_train), list(X_val), list(y_val))
        else:
            model.train(list(X_train_full), list(y_train_full))

        _, metrics = model.evaluate(list(X_test), list(y_test))
        results.append(metrics)

    df = pd.DataFrame(results)
    mean_metrics = df.mean().to_dict()
    std_metrics = df.std().to_dict()

    return mean_metrics, std_metrics



def evaluate_model(model, model_type, preprocessor):
    pass

def compare_models():
    pass
