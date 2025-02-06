# Written By: Areeba Amjad (B20102022)

import streamlit as st
import pickle
import re

# Load TF-IDF Vectorizer
with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    tfidf_vectorizer = pickle.load(vectorizer_file)

# Load Logistic Regression model
with open('logistic_model.pkl', 'rb') as model_file:
    logistic_model = pickle.load(model_file)

def process_text(text):
    text = re.sub(r'@\w+', '', text)  # mentions
    text = re.sub(r'#\w+', '', text)  # hashtags
    text = re.sub(r'https?://\S+', '', text)  # URLs
    text = re.sub(r'@[a-zA-Z0-9_]+', '', text)  # twitter handles
    text = re.sub(r'[^\w\s]', '', text)  # special characters
    text = re.sub(r'\s+', ' ', text)  # whitespace
    return text.strip()

sentiment_mapping = {
    0: "Negative 😠",
    1: "Neutral 😐",
    2: "Positive 😊"
}

def main():
    col1, col2, col3, col4 = st.columns([1, 1, 3, 1])
    with col3:
        st.image("twitter.png", width=100)

    st.title("Twitter Sentiment Classifier")
    st.write("Enter a twitter text below:")

    input_text = st.text_area("Input Text", "")

    if st.button("Predict"):
        cleaned_text = process_text(input_text)
        vectorized_text = tfidf_vectorizer.transform([cleaned_text])
        sentiment_prediction = logistic_model.predict(vectorized_text)[0]

        predicted_sentiment = sentiment_mapping.get(sentiment_prediction, 'Unknown ❓')
        st.write(f"Predicted Sentiment: {predicted_sentiment}")

if __name__ == "__main__":
    main()
