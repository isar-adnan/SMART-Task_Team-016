# This code loads the DBpedia dataset json file and convert it into dataframe 
import json
import os
import numpy as np
import pandas as pd


# Read the Dataset from json file and save it into Dataframe
def load_dataset(path:str) -> pd.DataFrame: 
    df = pd.read_json(path)
    return df

def load_data(path: str) -> dict:
    """Loads the data from the path.
    Args:
        path (str): relative path of the data files.
    Returns:
        dict: DBpedia ontology.
    """
    file = open(path, "r")
    data = json.load(file)
    file.close()
    return  data
   
# Remove Null values from the Dataset
def preprocessingDB(df: pd.DataFrame):
    df = df.dropna()
    return df

def UpdateCategroy(df: pd.DataFrame) -> pd.DataFrame:
    df["updated_category"] = np.where(df["category"] == "literal", df["type"].str[0], df["category"])
    return df

def get_dataframe():
    train_path = "data/smarttask_dbpedia_train.json"
    test_path = "data/smarttask_dbpedia_test.json"
    df_train = load_dataset(train_path)
    df_test = load_dataset(test_path)
    # Preprocessing
    df_train = preprocessingDB(df_train)
    df_test = preprocessingDB(df_test)
    return df_train, df_test

#train and test queries for the type prediction


if __name__ == "__main__":

    print("="*100)
    print("Loading data...")
    print("="*100) 
   
    #df = load_dataset(train_path)
    train, test = get_dataframe()
    print("Train set shape:",train.shape, "Test set shape:",test.shape)
