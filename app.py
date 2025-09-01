from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle


train_texts = [
    "Free prize! Click to win money!",
    "Meeting reminder for tomorrow",
    "URGENT: Verify your account",
    "Hi, how are you doing?"
]
train_labels = [1, 0, 1, 0]

# 1. Create and fit vectorizer
tfidf = TfidfVectorizer()
X_train = tfidf.fit_transform(train_texts)

# 2. Train the model
model = MultinomialNB()
model.fit(X_train, train_labels)


with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Both vectorizer and model trained and saved successfully!")



import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


nltk.download('stopwords')
nltk.download('punkt')
ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    words = nltk.word_tokenize(text)
    words = [ps.stem(w) for w in words if w.isalnum() and w not in stopwords.words('english')]
    return " ".join(words)



try:
    # Load vectorizer
    with open('vectorizer.pkl', 'rb') as f:
        tfidf = pickle.load(f)
        if not hasattr(tfidf, 'vocabulary_'):
            st.error("Vectorizer not fitted properly!")
            st.stop()

    # Load model
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
        if not hasattr(model, 'classes_'):
            st.error("Model not trained properly!")
            st.stop()

except Exception as e:
    st.error(f"Loading error: {str(e)}")
    st.stop()

# Streamlit UI
st.title(" SMS / Email Spam Classifier")
user_input = st.text_area("Enter message to check:")

if st.button("Check "):
    if not user_input.strip():
        st.warning("Please enter a message")
    else:
        try:
            # Preprocess
            clean_text = transform_text(user_input)

            # Vectorize
            features = tfidf.transform([clean_text])

            # Predict
            prediction = model.predict(features)[0]
            proba = model.predict_proba(features)[0]

            # Display results
            if prediction == 1:
                st.error(f"🚨 Spam ")
            else:
                st.success(f"✅ Not spam ")

        except Exception as e:
            st.error(f"Prediction error: {str(e)}")








