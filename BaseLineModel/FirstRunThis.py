# Written By: Areeba Amjad (B20102022)

import pandas as pd
import numpy as np
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt


# Load the dataset
dataset = pd.read_csv('Airline-Sentiment-Count.csv')
dataset = pd.read_csv('Tweets.csv')
dataset = dataset[['text', 'airline_sentiment']]

# Preprocess the text
def preprocess_text(text):
    text = re.sub(r'@\w+', '', text)  # mentions
    text = re.sub(r'#\w+', '', text)  # hashtags
    text = re.sub(r'https?://\S+', '', text)  # URLs
    text = re.sub(r'@[a-zA-Z0-9_]+', '', text)  # twitter handles
    text = re.sub(r'[^\w\s]', '', text)  # special characters
    text = re.sub(r'\s+', ' ', text)  # whitespace
    text = text.lower()  
    return text.strip()


dataset['processed_text'] = dataset['text'].apply(preprocess_text)

# Map sentiment labels to numerical values
sentiment_mapping = {'positive': 2, 'neutral': 1, 'negative': 0}
dataset['airline_sentiment'] = dataset['airline_sentiment'].map(sentiment_mapping)

# Split the data
X = dataset['processed_text']
y = dataset['airline_sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize using TF-IDF
vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)

# Train the Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vectorized, y_train)

# Save the vectorizer and model
with open('tfidf_vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

with open('logistic_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

# Evaluate accuracy
X_test_vectorized = vectorizer.transform(X_test)
y_pred = model.predict(X_test_vectorized)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: ", accuracy)

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)


print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Negative', 'Neutral', 'Positive']))


# Visualize the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Neutral', 'Positive'], yticklabels=['Negative', 'Neutral', 'Positive'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()
