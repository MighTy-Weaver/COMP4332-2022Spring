import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

if __name__ == "__main__":
    ans = pd.read_csv("data/valid.csv", usecols=["review_id", "stars"])
    pred = pd.read_csv("data/valid_pred.csv", usecols=["review_id", "stars"])
    df = pd.merge(ans, pred, how="left", on=["review_id"])
    df.fillna(0, inplace=True)
    acc = accuracy_score(df["stars_x"], df["stars_y"])
    p, r, f1, _ = precision_recall_fscore_support(df["stars_x"], df["stars_y"], average="macro")
    print("accuracy:", acc, "\tprecision:", p, "\trecall:", r, "\tf1:", f1)
