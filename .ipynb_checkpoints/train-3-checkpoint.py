import numpy as np
import random
import sys
import io
import sqlite3
import unicodedata
import string
import re
import tensorflow as tf


from keras.models import Model, Sequential
from keras.layers import Dense, Activation, Dropout, Input, LSTM, Reshape, Lambda, RepeatVector
from keras.initializers import glorot_uniform
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras import backend as K
from keras.callbacks import LambdaCallback


conn = sqlite3.connect('funk_crawler/songs.db')
cursor = conn.cursor()

cursor.execute('SELECT text from songs')
songs = cursor.fetchall()
text = '\n'.join([song[0] for song in songs])


def shave_marks_latin(txt):
    norm_text = unicodedata.normalize('NFD', txt)
    latin_base = False
    keepers = []
    for c in norm_text:
        if unicodedata.combining(c) and latin_base:
            continue
        keepers.append(c)
        if not unicodedata.combining(c):
            latin_base = c in string.ascii_letters
    shaved = ''.join(keepers)
    return unicodedata.normalize('NFC', shaved)

text = shave_marks_latin(text).lower()
letter_space_re = re.compile(r'[^a-z\s]')
text = letter_space_re.sub('', text)
sentences = text.split('\n')

sentences = set(sentences)

tokenizer = re.compile(r'\s+')
sentences_tokens = [tokenizer.split(s) for s in sentences]

sentences_tokens = [sentence for sentence in sentences_tokens
                    if len(sentence) < 10]

words = set([w for s in sentences_tokens for w in s])

word_indices = dict((c, i) for i, c in enumerate(words))
indices_word = dict((i, c) for i, c in enumerate(words))

Tx = 10
N_values = len(words)
m = len(sentences_tokens)
X = np.zeros((m, Tx, N_values), dtype=np.bool)
Y = np.zeros((m, Tx, N_values), dtype=np.bool)
for i in range(m):
    data = sentences_tokens[i]
    for j in range(Tx):
        try:
            idx = word_indices[data[j]]
        except IndexError:
            idx = word_indices['']
        if j != 0:
            X[i, j, idx] = 1
            Y[i, j-1, idx] = 1

Y = np.swapaxes(Y,0,1)

n_a = 64
reshapor = Reshape((1, N_values))
LSTM_cell = LSTM(n_a, return_state = True)
densor = Dense(N_values, activation='softmax')

def funkmodel():
    # Define the input of your model with a shape 
    X = Input(shape=(Tx, N_values))
    
    # Define s0, initial hidden state for the decoder LSTM
    a0 = Input(shape=(n_a,), name='a0')
    c0 = Input(shape=(n_a,), name='c0')
    a = a0
    c = c0
    
    ### START CODE HERE ### 
    # Step 1: Create empty list to append the outputs while you iterate (≈1 line)
    outputs = []
    
    # Step 2: Loop
    for t in range(Tx):

        # Step 2.A: select the "t"th time step vector from X. 
        x = Lambda(lambda x: X[:,t,:])(X)
        # Step 2.B: Use reshapor to reshape x to be (1, n_values) (≈1 line)
        x = reshapor(x)
        # Step 2.C: Perform one step of the LSTM_cell
        a, _, c = LSTM_cell(x, initial_state=[a, c])
        # Step 2.D: Apply densor to the hidden state output of LSTM_Cell
        out = densor(a)
        # Step 2.E: add the output to "outputs"
        outputs.append(out)
        
    # Step 3: Create model instance
    model = Model(inputs=[X, a0, c0], outputs=outputs)
    
    ### END CODE HERE ###
    
    return model

model = funkmodel()

opt = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0.01)

model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

a0 = np.zeros((m, n_a))
c0 = np.zeros((m, n_a))

model.fit([X, a0, c0], list(Y), epochs=10)

def one_hot(x):
    # THIS IS NOT SAMPLING, JUST PICKING THE MOST LIKELY
    x = K.argmax(x)
    x = tf.one_hot(x, N_values) 
    x = RepeatVector(1)(x)
    return x

def inference_model(LSTM_cell, densor, n_values, n_a, Ty = 20):
    """
    Uses the trained "LSTM_cell" and "densor" from model() to generate a sequence of values.
    
    Arguments:
    LSTM_cell -- the trained "LSTM_cell" from model(), Keras layer object
    densor -- the trained "densor" from model(), Keras layer object
    n_values -- integer, umber of unique values
    n_a -- number of units in the LSTM_cell
    Ty -- integer, number of time steps to generate
    
    Returns:
    inference_model -- Keras model instance
    """
    
    # Define the input of your model with a shape 
    x0 = Input(shape=(1, n_values))
    
    # Define s0, initial hidden state for the decoder LSTM
    a0 = Input(shape=(n_a,), name='a0')
    c0 = Input(shape=(n_a,), name='c0')
    a = a0
    c = c0
    x = x0

    ### START CODE HERE ###
    # Step 1: Create an empty list of "outputs" to later store your predicted values (≈1 line)
    outputs = []
    
    # Step 2: Loop over Ty and generate a value at every time step
    for t in range(Ty):
        print(x)
        print(a)
        print(c)
        # Step 2.A: Perform one step of LSTM_cell (≈1 line)
        a, _, c = LSTM_cell(x, initial_state=[a, c])
        
        # Step 2.B: Apply Dense layer to the hidden state output of the LSTM_cell (≈1 line)
        out = densor(a)

        # Step 2.C: Append the prediction "out" to "outputs". out.shape = (None, 78) (≈1 line)
        outputs.append(out)
        
        # Step 2.D: Select the next value according to "out", and set "x" to be the one-hot representation of the
        #           selected value, which will be passed as the input to LSTM_cell on the next step. We have provided 
        #           the line of code you need to do this. 
        x = Lambda(one_hot)(out)
        
    # Step 3: Create model instance with the correct "inputs" and "outputs" (≈1 line)
    inference_model = Model(inputs=[x0, a0, c0], outputs=outputs)
    
    ### END CODE HERE ###
    
    return inference_model

inference_model = inference_model(LSTM_cell, densor, n_values = N_values, n_a = n_a, Ty = 10)

x_initializer = np.zeros((1, 1, N_values))
a_initializer = np.zeros((1, n_a))
c_initializer = np.zeros((1, n_a))

def predict_and_sample(inference_model, x_initializer = x_initializer, a_initializer = a_initializer, c_initializer = c_initializer):
    
    ### START CODE HERE ###
    # Step 1: Use your inference model to predict an output sequence given x_initializer, a_initializer and c_initializer.
    pred = inference_model.predict([x_initializer, a_initializer, c_initializer])
    # Step 2: Convert "pred" into an np.array() of indices with the maximum probabilities
    indices = np.argmax(pred, axis=-1)
    # Step 3: Convert indices to one-hot vectors, the shape of the results should be (1, )
    results = to_categorical(indices, num_classes=N_values)
    ### END CODE HERE ###
    
    return results, indices
results, indices = predict_and_sample(inference_model, x_initializer, a_initializer, c_initializer)

print(indices)
print(' '.join([indices_word[i[0]] for i in indices]).strip())

model.save('model-4.h5')
