from tensorflow.keras.preprocessing.text import Tokenizer # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
import numpy as np

# Read file
def read_file(filepath):#'i-485instr.txt'
    with open(filepath) as file:  
        return file.read().lower()


# Tokenization (Creating index for words)
def words_index(text):
    tokens = Tokenizer()
    tokens.fit_on_texts([text])
    return tokens
#Print created tokens
def print_words_index(tokens):
    total_words = len(tokens.word_index)+1
    print(f'Total Words: {total_words}')
    print(tokens.word_index)  
    print(tokens.index_word)  
#Get Total Words - Dictonary size
def get_total_words(tokens):
    return len(tokens.word_index)+1

    

# Create input sequence for LSTM
def input_seq(text, tokens):
    #Create every possible n gram sequence from first word per line
    input_sequence = []
    for line in text.split('\n'):
        token_list = tokens.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence=token_list[:i+1]
            input_sequence.append(n_gram_sequence)

    return input_sequence
#Print created input sequence for LSTM
def print_input_seq(inputseq, tokens):
    for seq in inputseq:
        word_seq = []
        for index in seq:
            word_seq.append(tokens.index_word.get(index,'-'))
        print(' '.join(word_seq))


def get_max_seq_len(input_sequence):
    return max(len(seq) for seq in input_sequence)

def add_padding_pre(input_sequence):
    padding = 'pre'
    longest_seq_length = get_max_seq_len(input_sequence)

    return np.array(pad_sequences(input_sequence,maxlen=longest_seq_length, padding=padding))


#Predict Next Word
def predict_next_word(model, tokenizer, text, max_sequence_len):
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