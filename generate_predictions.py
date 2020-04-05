import pandas as pd
import numpy as np
import joblib
from config import PRED_CONFIGS, FOLDERS

LOAD = PRED_CONFIGS["load"]
SAVE = PRED_CONFIGS["save"]

# Import Essential Items
print("Importing Essential Items...")
test = pd.read_csv(LOAD["test_data"])
model = joblib.load(open(LOAD["model"], "rb"))
print("Done...\n")

# Preprocess Test Data
print("Preprocessing Test Data...")
if PRED_CONFIGS["impute"]:
    print("IMPUTE!")
    impute_dic = joblib.load(open(LOAD["imputation"], "rb"))
    test["NumberOfDependents"] = test["NumberOfDependents"].fillna(impute_dic["NumberOfDependents"]["mode"])
    if "median" in impute_dic["MonthlyIncome"]:
        print("MEDIAN")
        test["MonthlyIncome"] = test["MonthlyIncome"].fillna(impute_dic["MonthlyIncome"]["median"])
    else:
        np.random.seed(0)
        test.loc[test["MonthlyIncome"].isnull()]["MonthlyIncome"] = np.random.normal(loc=impute_dic["MonthlyIncome"]["mean"], scale=impute_dic["MonthlyIncome"]["std"], size=len(test.loc[test["MonthlyIncome"].isnull()]))
print("Done...")

# Generate Predictions
print("Generating Predictions...")
test_X = test.drop(["Unnamed: 0", "SeriousDlqin2yrs"],axis=1)
predict_prob = model.predict_proba(test_X)[:, -1]
print("Done...\n")

# Save Predictions
print("Saving Predictions...")
result = pd.DataFrame({"Id": test["Unnamed: 0"], "Probability": predict_prob})
result["Id"] = result["Id"].astype(int)
result["Probability"] = result["Probability"].astype(float)
result.to_csv(SAVE["predictions"], index=False)
print("Completed!")