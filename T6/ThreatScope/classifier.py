import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from config import MODEL_DIR

CLF_PATH = MODEL_DIR / "threat_clf.joblib"

def train_baseline(texts, labels):
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1, 2), stop_words="english")),
        ("clf", LogisticRegression(max_iter=1000)),
    ])
    pipe.fit(texts, labels)
    joblib.dump(pipe, CLF_PATH)
    return pipe

def load_baseline():
    if CLF_PATH.exists():
        return joblib.load(CLF_PATH)
    return None

def predict(pipe, texts):
    if pipe is None:
        return [("other", 0.0)] * len(texts)
    probs = pipe.predict_proba(texts)
    classes = pipe.classes_
    out = []
    for p in probs:
        i = int(np.argmax(p))
        out.append((classes[i], float(p[i])))
    return out