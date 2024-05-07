import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import CountVectorizer

# Load data
keywords_df = pd.read_csv('keyFeature.csv')
labels_df = pd.read_csv('PARIS_result.csv')

# Preprocessing
vectorizer = CountVectorizer(tokenizer=lambda x: x.split(','))
X = vectorizer.fit_transform(keywords_df['Keywords'])

# Function to calculate token importance for a specific error type
def calculate_token_importance(X, y):
    logreg = LinearRegression(max_iter=1000)
    logreg.fit(X, y)
    importance_df = pd.DataFrame({
        'Token': vectorizer.get_feature_names_out(),
        'Coefficient': logreg.coef_[0]
    })
    importance_df['Importance'] = importance_df['Coefficient'].abs()
    return importance_df.sort_values(by='Importance', ascending=False)

# Calculate importance for each error type
benign_importance = calculate_token_importance(X, labels_df['Actual_benign'])
crash_importance = calculate_token_importance(X, labels_df['Actual_crash'])
SDC_importance = calculate_token_importance(X, labels_df['Actual_SDC'])

# Display results
print("Benign Token Importance:\n", benign_importance.head(10))
print("\nCrash Token Importance:\n", crash_importance.head(10))
print("\nSDC Token Importance:\n", SDC_importance.head(10))