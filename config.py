import os

FOLDERS = {
    "preprocess": "./data/",
    "train": "./model_output/"
}

for folder in FOLDERS.values():
    if not os.path.exists(folder):
        os.makedirs(folder)

PREPROCESS_CONFIGS = {
    "load": {
        "raw_data": FOLDERS["preprocess"] + "cs-training.csv"
    },
    "save": {
        "preprocessed_data": FOLDERS["preprocess"] + "preprocessed.csv",
        "imputation": FOLDERS["preprocess"] + "impute.p"
    },
    "impute": False
}

TRAIN_CONFIGS = {
    "load": {
        "preprocessed_data": PREPROCESS_CONFIGS["save"]["preprocessed_data"],
    },
    "save": {
        "model": FOLDERS["train"] + "model.p"
    },
    "model_params": {
        'min_child_weight': list(range(1,10,2)),
        'gamma':[i/10.0 for i in range(0,5)],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree':[i/10.0 for i in range(6,10)],
        'max_depth':range(3,10,2),
        'reg_alpha': [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2],
        'reg_lambda': [0, 0.001, 0.005, 0.01, 0.05, 0.2, 0.4, 0.6, 0.8, 1],
        'learning_rate': [0.001, 0.002, 0.005, 0.006, 0.01, 0.02, 0.05, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.2],
        'n_estimators': [50, 100, 150, 200, 250, 300,350,400,450,500, 550, 600, 650, 700, 750],
        "booster": ["gbtree", "gblinear", "dart"]
        }
}

PRED_CONFIGS = {
    "load": {
        "test_data": FOLDERS["preprocess"] + "cs-test.csv",
        "model": TRAIN_CONFIGS["save"]["model"],
        "imputation": PREPROCESS_CONFIGS["save"]["imputation"]
    },
    "save": {
        "predictions": FOLDERS["train"] + "predictions.csv"
    },
    "impute": PREPROCESS_CONFIGS["impute"]
}