import streamlit as st
import requests
import gdown
import pickle
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences 


# Utilityfunctions
# 1. Load model and tokens from drive
@st.cache_resource(ttl = 60*60 *24 *7, show_spinner="Fetching model from cloud...")
def load_model_from_google_drive(fileid_weights, fileid_tokens, weights_path, pickle_path):
    # Download and Save Model
    weights_url = f'https://drive.google.com/uc?id={fileid_weights}'
    gdown.download(weights_url, weights_path, quiet=True)
    # Load Model
    try:
        model_gd = tf.keras.models.load_model(weights_path)
    except Exception as e:
        st.error(f"Error loading the model weights: {e}")

    # Download and save tokens
    tokens_url = f'https://drive.google.com/uc?id={fileid_tokens}'
    gdown.download(tokens_url, pickle_path, quiet=True)
    # Load Tokens
    try:
        with open(pickle_path, 'rb') as handle:
            tokens = pickle.load(handle)
    except Exception as e:
        st.error(f"Error loading the model tokens: {e}")

    # Return model and tokens
    return model_gd, tokens

# 2. Predict next word
def predict_next_word(model, tokenizer, text):
    max_sequence_len = model.input_shape[1]+1
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len-1):]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)
    for word, index in tokenizer.word_index.items():
        if index==predicted_word_index:
            return word
    return None



## Load the Model from Google Drive
weights_path = 'next_word_LSTM.h5'
pickle_path = 'tokens.pickle'
weights_fileid_drive = st.secrets["WEIGHTS_FILE_ID"]
tokens_fileid_drive = st.secrets["TOKENS_FILE_ID"]
model, tokens = load_model_from_google_drive(weights_fileid_drive, tokens_fileid_drive, weights_path, pickle_path)
# model = tf.keras.models.load_model("src/next_word_LSTM.h5")
# with open("src/tokens.pickle", 'rb') as handle:
#     tokens = pickle.load(handle)




# Initialize global variables
input_text = None
output_label = None
n_next_words = 5

# Initialize the state variables
if "n_next_words" not in st.session_state:
    st.session_state.n_next_words = n_next_words


# Streamlit UI
st.title("Next Word Predict")
st.write("LSTM trained on USCIS-I485 instructions")
input_text = st.text_input("Enter the sequence of words", "USCIS will not accept")
st.session_state.n_next_words = st.slider("Number of next words to predict", min_value=1, max_value=10, value=n_next_words)


if st.button("Predict"):
    result = st.empty()
    result.text(input_text)
    for _ in range(st.session_state.n_next_words):
        next_word = predict_next_word(model, tokens, input_text)
        input_text += ' '+next_word
        result.text(input_text) 