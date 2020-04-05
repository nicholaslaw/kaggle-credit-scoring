import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from xgboost import XGBClassifier
from config import TRAIN_CONFIGS, FOLDERS

LOAD = TRAIN_CONFIGS["load"]
SAVE = TRAIN_CONFIGS["save"]

# Import Preprocessed Data
print("Importing Preprocessed Data...")
df = pd.read_csv(LOAD["preprocessed_data"])
X = df.drop("SeriousDlqin2yrs", axis=1)
Y = df["SeriousDlqin2yrs"]
print("Done...\n")

# Train Model
print("Training Model...")
xgb = XGBClassifier(random_state=0)
model = RandomizedSearchCV(xgb, param_distributions=TRAIN_CONFIGS["model_params"], n_iter=400, scoring='roc_auc', n_jobs=-1, cv=StratifiedKFold(n_splits=5, shuffle = True, random_state = 0), verbose=3, random_state=0)
model.fit(X, Y, eval_metric="auc")
print("Done...\n")

# Save Model
print("Saving Model...")
joblib.dump(model, open(SAVE["model"], "wb"))
print("Completed!")