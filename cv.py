import pickle
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

# Assuming 'data.csv' contains your text data
data = pd.read_csv("data.csv")

# Extract text data from DataFrame
x = data["tweet"]

# Instantiate CountVectorizer
cv = CountVectorizer()

# Fit CountVectorizer to text data
x_cv = cv.fit_transform(x)

# Save CountVectorizer using pickle
with open("count_vectorizer.pkl", "wb") as f:
    pickle.dump(cv, f)
