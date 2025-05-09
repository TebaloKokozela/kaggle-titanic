import pandas as pd


def load_data(file_path="../data/raw/train.csv",index="PassengerId"):
    df =  pd.read_csv(file_path, index_col = index)
    return df


