import streamlit as st
import gdown
import pickle
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences 


# Utilityfunctions
# 1. Load model and tokens from drive
@st.cache_resource(ttl = 60*60 *24 *7, show_spinner="Fetching model from cloud...")
def load_model_from_google_drive(fileid_weights_LSTM, fileid_weights_GRU, fileid_tokens, path_weights_LSTM, path_weights_GRU, path_tokens):
    # Download and Save Model - LSTM
    weights_url = f'https://drive.google.com/uc?id={fileid_weights_LSTM}'
    gdown.download(weights_url, path_weights_LSTM, quiet=True)
    # Load Model - LSTM
    try:
        model_lstm = tf.keras.models.load_model(path_weights_LSTM)
    except Exception as e:
        st.error(f"Error loading the model weights - LSTM: {e}")

    # Download and Save Model - GRU
    weights_url = f'https://drive.google.com/uc?id={fileid_weights_GRU}'
    gdown.download(weights_url, path_weights_GRU, quiet=True)
    # Load Model - GRU
    try:
        model_gru = tf.keras.models.load_model(path_weights_GRU)
    except Exception as e:
        st.error(f"Error loading the model weights - GRU: {e}")

    # Download and save tokens
    tokens_url = f'https://drive.google.com/uc?id={fileid_tokens}'
    gdown.download(tokens_url, path_tokens, quiet=True)
    # Load Tokens
    try:
        with open(path_tokens, 'rb') as handle:
            tokens = pickle.load(handle)
    except Exception as e:
        st.error(f"Error loading the model tokens: {e}")

    # Return model and tokens
    return model_lstm, model_gru, tokens


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
weights_path_lstm = 'next_word_LSTM.h5'
weights_path_gru  = 'next_word_GRU.h5'
pickle_path       = 'tokens.pickle'
weights_fileid_lstm = st.secrets["WEIGHTS_LSTM"]
weights_fileid_gru  = st.secrets["WEIGHTS_GRU"]
tokens_fileid       = st.secrets["TOKENS"]
model_lstm, model_gru, tokens = load_model_from_google_drive(weights_fileid_lstm, weights_fileid_gru, tokens_fileid, 
                                                             weights_path_lstm,    weights_path_gru,   pickle_path)
# model_lstm = tf.keras.models.load_model("src/next_word_LSTM.h5")
# model_gru  = tf.keras.models.load_model("src/next_word_GRU.h5")
# with open("src/tokens.pickle", 'rb') as handle:
#     tokens = pickle.load(handle)




# Initialize global variables
input_text = None
n_next_words = 5
model_type = "LSTM"

# Initialize the state variables
if "n_next_words" not in st.session_state:
    st.session_state.n_next_words = n_next_words
if "model_type" not in st.session_state:
    st.session_state.model_type = model_type

# Streamlit UI
st.title("Next Word Predict")
st.write(f'<p style="font-size: medium">LSTM trained on USCIS-I485 instructions</p>', unsafe_allow_html=True)
input_text = st.text_input("Enter the sequence of words", "USCIS will not accept")
col1, col2 = st.columns(2)
with col1:
    st.session_state.model_type = st.selectbox("Select Model",("LSTM", "GRU"))
with col2:
    st.session_state.n_next_words = st.slider("Number of next words to predict", min_value=1, max_value=10, value=n_next_words)


if st.button("Predict"):
    st.write(f'<p style="font-size: medium">Predicting Next {st.session_state.n_next_words} Words By {st.session_state.model_type} Model</p>', unsafe_allow_html=True)
    model = model_lstm if st.session_state.model_type=='LSTM' else model_gru
    result = st.empty()
    result.text(input_text)
    for _ in range(st.session_state.n_next_words):
        next_word = predict_next_word(model, tokens, input_text)
        input_text += ' '+next_word
        result.text(input_text) 