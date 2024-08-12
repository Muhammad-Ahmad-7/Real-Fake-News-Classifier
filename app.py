import streamlit as st
import pickle
import pandas as pd
import numpy as np
import nltk
import re
import string

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

nltk.download('stopwords')
nltk.download('punkt')

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

def transform_text(text):
    y = []
    ## Lower Case
    text = text.lower()
    text = nltk.word_tokenize(text)

    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))
    
    return " ".join(y)

st.title('Real/Fake News Classifier')
input = st.text_area('Enter the news text')

if st.button('Predict'):
    
    # 1. Transforming the text
    transform_news = transform_text(input)
    
    # 2. Vectorize the given transform text
    vector_input = tfidf.transform([transform_news])
    
    # 3. Make the prediction
    
    result = model.predict(vector_input)
    
    if result == 1:
        st.header('Real News')
    else:
        st.header('Fake News')
    