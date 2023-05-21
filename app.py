from math import expm1
from flask import Flask
import joblib
import pandas as pd
import pickle
from flask import Flask, jsonify, request
from tensorflow import keras

from keras.utils import pad_sequences
from keras.preprocessing.text import Tokenizer

from utils import clean_text


app = Flask(__name__)

lstm_text_model = keras.models.load_model("models/kaggle_lstm_text.h5")
lr_title_model = joblib.load("models/wellfake_lr_title.joblib")

tfidf_vectorizer = pickle.load(open('models/wellfake_lr_title_vectorizer.pickle', 'rb'))


def tokenize(data):
  tokenizer = Tokenizer(num_words = 10000)
  tokenizer.fit_on_texts(data)
  sequences = tokenizer.texts_to_sequences(data)

  tokenized_data = pad_sequences(sequences, maxlen = 300)
  return tokenized_data

@app.route("/lstm-text", methods=["POST"], endpoint='get_text_prediction')
def get_text_prediction():
    data = request.json
    print("Received Input text", data)
    cleaned = clean_text(data)
    prediction =  lstm_text_model.predict(tokenize([cleaned]))
    print(str(prediction))
    return jsonify({"prediction": str(prediction)})
