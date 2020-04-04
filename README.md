# Kaggle's Give Me Some Credit

Competition: https://www.kaggle.com/c/GiveMeSomeCredit/overview

# Installation
Python 3.6 or later is required to run the codes in this repository. There are a few ways to install the required Python packages:

1. pip
```
pip install -r requirements.txt
```

2. docker
```
docker-compose up
```

# Notebooks
- EDA.ipynb
This notebook contains all the codes used to perform Exploratory Data Analysis (EDA) of the challenge's dataset. If docker is used for setting up, navigate to *localhost:8889*, password would be *password*, and click into *app/* folder.

# Scripts
- preprocess.py
Running this script will preprocess training dataset and saves essential items to ensure that the test dataset goes through the same preprocessing steps.

- train.py
Running this script will train a classic machine learning model on the preprocessed training dataset and saves the model.

- generate_predictions.py
Running this script will allow the trained model to generate predictions for the test dataset.

# Automate Running of All Scripts
Run the following commands in your terminal to generate predictions for the test dataset in the competition
```
chmod a+x ./run.sh
./run.sh
```