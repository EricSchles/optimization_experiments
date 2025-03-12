import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def split_unevenly(df):
    half = len(df)/2
    if int(half) != half:
        return int(half)+1, int(half)
    return int(half), int(half)

def gen_training_data():
    df_cov = pd.DataFrame()

    df_cov["party"] = [random.choice(["republican", "democrat"])
                       for _ in range(2000)]
    democrat = df_cov[df_cov["party"] == "democrat"]
    republican = df_cov[df_cov["party"] == "republican"]
    first_half_republican, second_half_republican = split_unevenly(republican)
    first_half_democrat, second_half_democrat = split_unevenly(democrat)

    democrat["Age"] = np.random.normal(35, 19, size=len(democrat))
    democrat["Age"] = democrat["Age"].astype(int)
    republican["Age"] = np.random.normal(55, 10, size=len(republican))
    republican["Age"] = republican["Age"].astype(int)

    democrat["Salary"] = np.random.normal(65000, 15000, size=len(democrat))
    democrat["Salary"] = democrat["Salary"].apply(lambda x: round(x, 2))
    republican["Salary"] = np.concatenate([
        np.random.normal(25000, 1500, size=first_half_republican),
        np.random.normal(155000, 10000, size=second_half_republican)
    ])
    republican["Salary"] = republican["Salary"].apply(lambda x: round(x, 2))

    democrat["Latitude"] = np.concatenate([
        np.random.normal(40, 1, size=first_half_democrat),
        np.random.normal(34, 1, size=second_half_democrat)
    ])
    democrat["Latitude"] = democrat["Latitude"].apply(lambda x: round(x, 4))
    republican["Latitude"] = np.random.normal(39, 15, size=len(republican))
    republican["Latitude"] = republican["Latitude"].apply(lambda x: round(x, 4))

    democrat["Longitude"] = np.random.normal(94, 15, size=len(democrat))
    democrat["Longitude"] = democrat["Longitude"].apply(lambda x: round(x, 4))
    republican["Longitude"] = np.random.normal(94, 15, size=len(republican))
    republican["Longitude"] = republican["Longitude"].apply(lambda x: round(x, 4))

    df_cov = pd.concat([democrat, republican])
    df_cov["intelligence"] = df_cov["Salary"] * df_cov["Age"]

    df_cov["party"] = df_cov["party"].map({"republican": 0, "democrat": 1})
    X_cov = df_cov[["Age", "Salary", "Longitude", "Latitude", "party"]]
    y_cov = df_cov["intelligence"]
    return train_test_split(X_cov, y_cov, random_state=1), df_cov
