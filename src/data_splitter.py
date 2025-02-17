import pandas as pd
from sklearn.model_selection import train_test_split

def split_data(input_csv="data/processed_features.csv", test_size=0.2):
    df = pd.read_csv(input_csv)
    train, test = train_test_split(df, test_size=test_size, random_state=42, stratify=df["style"])
    train.to_csv("data/train.csv", index=False)
    test.to_csv("data/test.csv", index=False)

if __name__ == "__main__":
    split_data()
