import pandas as pd
import numpy as np
import joblib
from config import PREPROCESS_CONFIGS, FOLDERS

LOAD = PREPROCESS_CONFIGS["load"]
SAVE = PREPROCESS_CONFIGS["save"]

# Import Data
print("Importing Raw Data...")
df = pd.read_csv(LOAD["raw_data"])
print("Done...\n")

# Preprocess Data
print("Preprocessing Data...")
df = pd.read_csv(LOAD["raw_data"])
df = df.drop("Unnamed: 0", axis=1) # drop id column
df = df.loc[df["DebtRatio"] <= df["DebtRatio"].quantile(0.975)]
df = df.loc[(df["RevolvingUtilizationOfUnsecuredLines"] >= 0) & (df["RevolvingUtilizationOfUnsecuredLines"] < 13)]
df = df.loc[df["NumberOfTimes90DaysLate"] <= 17]
impute_dic = None
if PREPROCESS_CONFIGS["impute"]:
    print("IMPUTE!")
    dependents_mode = df["NumberOfDependents"].mode()[0] # impute with mode
    df["NumberOfDependents"] = df["NumberOfDependents"].fillna(dependents_mode)

    #################################### Uncomment below if want median imputation for MonthlyIncome #################################### 
    income_median = df["MonthlyIncome"].median()
    df["MonthlyIncome"] = df["MonthlyIncome"].fillna(income_median)
    impute_dic = {
        "MonthlyIncome": {
            "median": income_median
        },
        "NumberOfDependents": {
            "mode": dependents_mode
        }
    }
    ################################################################################################################################################ 


    #################################### Uncomment below if want random normal imputation for Monthly Income #################################### 
    # mean = df["MonthlyIncome"].mean()
    # std = df["MonthlyIncome"].std()
    # np.random.seed(0)
    # df.loc[df["MonthlyIncome"].isnull()]["MonthlyIncome"] = np.random.normal(loc=mean, scale=std, size=len(df.loc[df["MonthlyIncome"].isnull()]))
    # impute_dic = {
    #     "MonthlyIncome": {
    #         "mean": mean,
    #         "std": std
    #     },
    #     "NumberOfDependents": {
    #         "mode": dependents_mode
    #     }
    # }
    ################################################################################################################################################ 
print("Done...\n")

# Save Essential Items
print("Saving Essential Items...\n")
if impute_dic:
    joblib.dump(impute_dic, open(SAVE["imputation"], "wb"))
df.to_csv(SAVE["preprocessed_data"], index=False)
print("Completed!")