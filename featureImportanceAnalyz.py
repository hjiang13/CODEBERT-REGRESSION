
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd
from scipy.stats import pointbiserialr

# Load data
keywords_df = pd.read_csv('keyFeature.csv')
labels_df = pd.read_csv('PARIS_result.csv')
keywords_df = keywords_df.sort_values("Filename").reset_index(drop=True)
labels_df = labels_df.sort_values("BenchMark").reset_index(drop=True)

# Preprocessing
vectorizer = CountVectorizer(tokenizer=lambda x: x.split(','))
X = vectorizer.fit_transform(keywords_df['Keywords']).toarray()
vocab = vectorizer.get_feature_names_out()

# Compute point-biserial correlation
correlations = [pointbiserialr(X[:, i], labels_df['Actual_SDC'])[0] for i in range(X.shape[1])]

# Create DataFrame
df = pd.DataFrame({'Token': vocab, 'Correlation': correlations})
df['Importance'] = abs(df['Correlation'])
df = df.sort_values('Importance', ascending=False)
print(df)