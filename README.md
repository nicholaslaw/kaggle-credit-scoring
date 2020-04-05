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

# Further Setup
Place all the data files (*.xls, *.csv) into data/ folder for the codes to run smoothly.

# Notebooks
- EDA.ipynb
This notebook contains all the codes used to perform Exploratory Data Analysis (EDA) of the challenge's dataset. If docker is used for setting up, navigate to *localhost:8889*, password would be *password* and the notebook should appear in your current directory.

# Scripts
- preprocess.py
Running this script will preprocess training dataset and saves essential items to ensure that the test dataset goes through the same preprocessing steps. Output items will be a pickled dictionary containing values for imputation (if activated), and preprocessed dataset in the data/ folder.

- train.py
Running this script will train a classic machine learning model on the preprocessed training dataset and saves the model. The model file can be found in model_output/ folder.

- generate_predictions.py
Running this script will allow the trained model to generate predictions for the test dataset. The predictions can be found in a csv contained within model_output/ folder.

# Automate Running of All Scripts
Run the following commands in your terminal to generate predictions for the test dataset in the competition
```
chmod a+x ./run.sh
./run.sh
```

# Best Parameters for XGBoost
- subsample: 0.6
- reg_lambda: 0
- reg_alpha: 0
- n_estimators: 550
- min_child_weight: 6
- max_depth: 8
- learning_rate: 0.007
- gamma: 0.4
- colsample_bytree: 0.6
- booster: gbtree
- random_state: 0