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



## Launching the API

In order to run the API, run the following command line in your terminal:

- uvicorn fastAPI:app --reload

Once you see the line **Application startup complete**, the API is launched successfully. You can go to [this website](http://127.0.0.1:8000/docs) in order to preview the API using fastAPI.

From that website, click **try it now** and upload a png, jpg or jpeg of an already cropped sudoku.
You can find an example to try it on in the repo, the file called **sudoku.png**

Once you click **execute**, a matrix of the sudoku will be returned.