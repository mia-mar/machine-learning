import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score


def main():
    df = pd.read_csv("data/products.csv")

    df = df[["Product Title", "Category Label"]]
    df = df.dropna()

    X = df["Product Title"].astype(str)
    y = df["Category Label"].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english")),
        ("model", LogisticRegression(max_iter=2000))
    ])

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))

    os.makedirs("model", exist_ok=True)
    joblib.dump(pipeline, "model/product_category_model.pkl")
    print("Saved: model/product_category_model.pkl")


if __name__ == "__main__":
    main()
