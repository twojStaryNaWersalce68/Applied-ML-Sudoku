from fastAPI import predict
from sudoku_digitalisation.features.sudoku_preprocessing import get_preprocessor
from sudoku_digitalisation.run_training import get_model

# Run: "python -m sudoku_digitalisation.main" if you have ModuleNotFoundError
if __name__ == "__main__":
    IS_PREPROCESSED = True      # bool to see if the dataset is preprocessed already
    TRAIN_CNN = False           # change this to True if you want to train the CNN, otherwise it is loaded
    TRAIN_SVM = False           # change this to True if you want to train the SVM (baseline)
    LOAD_SVM = False            # change this to True if you want to load the SVM

    EVALUATE = False            # change to true if you want to run the evaluation
    PREDICT = False             # change to true if you want to run the prediction

    CNN_NAME = 'sudoku_cnn'     # name for saved CNN, None if you want training
    SVM_NAME = 'default'        # name for saved SVM, None if you want training

    # preprocessor for everything, clip_limit is for CLAHE, output_size for the output size of cropped images
    preprocessor = get_preprocessor(clip_limit=3, output_size=252, is_preprocessed=IS_PREPROCESSED)

    # load or train cnn or svm or both
    if TRAIN_CNN:
        cnn = get_model('cnn', CNN_NAME, preprocessor)
    else:
        pass  # load the CNN
    if TRAIN_SVM:
        svm = get_model('svm', SVM_NAME, preprocessor)
    elif LOAD_SVM:
        pass  # load the SVM

    # evaluate if we want
    if EVALUATE:
        # if you use an SVM we need to catch the possibility the var has not been assigned.
        pass

    # prediction if we want but that would be more for the API
    if PREDICT:
        pass