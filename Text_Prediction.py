import streamlit as st
import pandas as pd
import numpy as np
import re
import string
import contractions
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

model = load_model("Text Domain Project Model.h5")

tokenizer = pickle.load(open("tokenizer.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))


MAX_LEN = 100
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def expand_contractions(text):
    return contractions.fix(text)

def clean_content(text):
    text = expand_contractions(text)
    text = re.sub(r'@\w+\s?', '', text)
    return text

def lowercase_text(text):
    return text.lower()

def remove_punctuation(text):
    return ''.join([char for char in text if char not in string.punctuation])

def remove_numbers(text):
    return ''.join([char for char in text if not char.isdigit()])

def remove_stopwords(text):
    return ' '.join([word for word in text.split() if word.lower() not in stop_words])

def remove_urls(text):
    return re.sub(r'http\S+', '', text)

def remove_html_tags(text):
    return re.sub(r'<.*?>', '', text)

def remove_underscores(text):
    return text.replace("_", " ")

def tokenize_and_lemmatize(text):
    tokens = word_tokenize(text)
    return ' '.join([lemmatizer.lemmatize(token) for token in tokens])

def preprocess_text(text):
    text = clean_content(text)
    text = lowercase_text(text)
    text = remove_punctuation(text)
    text = remove_numbers(text)
    text = remove_stopwords(text)
    text = remove_urls(text)
    text = remove_html_tags(text)
    text = remove_underscores(text)
    text = tokenize_and_lemmatize(text)
    return text

def text_to_padded_sequence(text):
    sequences = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequences, maxlen=MAX_LEN, truncating='post', padding='post')
    return padded

def predict_emotion(text):
    processed_text = preprocess_text(text)
    padded_sequence = text_to_padded_sequence(processed_text)
    prediction = model.predict(padded_sequence)
    predicted_label = label_encoder.inverse_transform([prediction.argmax()])[0]
    return predicted_label, processed_text

st.title("Text Emotion Detection")
st.write("Type any text below and get the predicted emotion!")

text_input = st.text_area("Enter Text Here:")

save_data = st.checkbox("Save this input to dataset")

if st.button("Predict Emotion"):
    if text_input.strip() != "":
        emotion, processed = predict_emotion(text_input)
        st.success(f"Predicted Emotion: **{emotion}**")
        st.info(f"Processed Text: {processed}")
        
        if save_data:
            try:
                df = pd.read_csv("Domain_Pre_Processed_Dataset.csv")
            except FileNotFoundError:
                df = pd.DataFrame(columns=['Text','Emotion','Processed_Text','Author','Tweet_id'])
            
            new_row = {
                'Text': text_input,
                'Emotion': emotion,
                'Processed_Text': processed,
                'Author': 'Random Author',
                'Tweet_id': 'Random ID'
            }
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            df.to_csv("Domain_Pre_Processed_Dataset.csv", index=False)
            st.success("Input saved to dataset")
    else:
        st.error("Please enter some text!")
