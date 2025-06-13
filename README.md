# Sudoku digitisation CNN

Our model takes an image of a sudoku as input, predicts the digits in each cell and returns a digitised version of the sudoku as a 9x9 matrix. For the prediction the sudoku grid is split into its individual cells such that each digit can be classified individually using a CNN

## Setup

The dependencies can be seen listed in the requirements.txt file. The python version has to be **python 3.10** in order for the dependencies to work correctly. In order to set up your virtual environment, run the following lines of code in your terminal, these lines of code assume you use venv as you virtual environment, if you wish to use something, else, you need to do it your self.

First create the environment in python3.10:

```bash
$ python3.10 -m venv .venv
```

Next upgrade the pip in your virtual environment so you don't run into potential issues in the future:

```bash
$ pip install --upgrade pip
```

Following that you can install all dependencies needed, these are listed in the requirements.txt with the versions that are needed:

```bash
$ pip install -r requirements.txt
```

Finally, you can activate your virtual environment:

```bash
$ source .venv/bin/activate
```

Currently, the virtual environment is called .venv, if you wish to change the name to something else replace .venv in the command to create the virtual environment as well as the one to activate it.

If you wish to leave the virtual environment at any point, just run the following code:

```bash
$ deactivate
```

## Running the code

### Running it via main

In order to run the code, run the following command line in your terminal:

'''bash
python -m sudoku_digitalisation.main
'''

Running it in a different way may result in a ModuleNotFoundError.

When running the file for the first time, you need to make sure that you import the dataset and preprocess it first, this is done by putting the IS_PREPROCESSED boolean in the main file to False. This will automatically fetch and locally save the dataset from hugging face (if you want to see the dataset for yourself, click [here](https://huggingface.co/datasets/Lexski/sudoku-image-recognition)).

In order to change what running the file does, open the file [sudoku_digitalisation/main.py](sudoku_digitalisation/main.py). Within this file, from lines 10 to 16, you will find booleans that you can set to true or false depending on what you want the code to run. What each boolean does

- IS_PREPROCESSED: If the dataset is not already on your computer (it does not come with the repo), set this to false and it will preprocess the data for you and save it locally, after that you can set it to true for the rest of your runs.
- TRAIN_CNN: This will train a new CNN using the preprocessed data. It will save the CNN under sudoku_digitalisation/models/saved under the name that is set in CNN_NAME on line 18.
- TRAIN_SVM: This will train a new SVM using the preprocessed data. It will save the SVM under sudoku_digitalisation/models/saved under the name that is set in SVM_NAME on line 19.
- GET_SVM: This loads a preexisting SVM and uses that instead of training a new one.
- COMPARE: This compares the CNN model to the SVM model by comparing the mean and variance
- EVALUATE: Evaluates the CNN model, either the one that was just trained or loads the one saved under sudoku_digitalisation/models/saved, if TRAIN_CNN is set to false and there is no saved model, it will throw an error.

### Running the streamlit prototype

In order to run the streamlit demo, you need to run the code:

'''bash
python -m streamlit run demo.py
'''

It will open the streamlit page automatically in your default browser, you can then follow the instructions on that page in order to digitise your sudoku.

### Launching the API

In order to run the API, run the following command line in your terminal:

'''bash
uvicorn fastAPI:app --reload
'''

Once you see the line **Application startup complete**, the API is launched successfully. You can go to [this website](http://127.0.0.1:8000/docs) in order to preview the API using fastAPI.

From that website, click **try it now** and upload a png, jpg or jpeg of an already cropped sudoku.
You can find an example to try it on in the repo, the file called **sudoku.png**

Once you click **execute**, a matrix of the sudoku will be returned.
