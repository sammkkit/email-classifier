import streamlit as st
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Load resources
ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    y = [ps.stem(i) for i in y]
    return " ".join(y)

# Load vectorizer & model
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Text Classifier 🔮")

# User input
user_input = st.text_area("Enter text to classify:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text first!")
    else:
        # Preprocess
        transformed_text = transform_text(user_input)
        # Vectorize
        vector_input = vectorizer.transform([transformed_text])
        # Predict
        prediction = model.predict(vector_input)[0]

        st.success(f"Prediction: {prediction}")
        if prediction == 1:
            st.error("🚨 This email is **SPAM**")
        else:
            st.success("✅ This email is **NOT SPAM**")
