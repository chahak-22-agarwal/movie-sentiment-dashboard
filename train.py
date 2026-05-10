import pandas as pd
import re
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

def preprocess(text):
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return text

df = pd.read_csv("IMDB_Dataset.csv")

df["label"] = df["sentiment"].map({"positive": 1, "negative": 0})
df["clean"] = df["review"].apply(preprocess)

X_train, X_test, y_train, y_test = train_test_split(df["clean"], df["label"], test_size=0.2)

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", LogisticRegression())
])

pipeline.fit(X_train, y_train)

with open("sentiment_pipeline.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("Done")