#!/usr/bin/env python
# coding: utf-8

# In[1]:


# # https://deeplearningcourses.com/c/deep-learning-advanced-nlp
# get the data at: http://www.manythings.org/anki/
from __future__ import print_function, division
from builtins import range, input
# Note: you may need to update your version of future
# sudo pip install -U future

import pickle
import os, sys
import random
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, GRU, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt

from keras.layers import CuDNNLSTM as LSTM
from keras.layers import CuDNNGRU as GRU

from tensorflow.keras.models import load_model
import keras.backend as K


# In[2]:


# some config
NUM_SAMPLES = 12800*40 # Number of samples to train on.
MAX_NUM_WORDS = 30000
MAX_LEN_INPUT = 50
MAX_LEN_TARGET = 30


# In[3]:


#path = r'D:\Dokumenty\DeepNLP\machine_learning_examples\nlp_class3\OwnScripts\Dane'
path = r"D:\Dokumenty\MaciexTranslate\data"


# In[4]:


with open(path + "/"+'polish.pkl', 'rb') as f:
    foreign_data = pickle.load(f)
random_indices = random.sample(range(0,len(foreign_data)),NUM_SAMPLES)
foreign_data = [foreign_data[i] for i in random_indices]
#foreign_data = foreign_data[0:10000]


# In[5]:


with open(path + "/"+'english.pkl', 'rb') as f:
    english_data = pickle.load(f)
english_data = [english_data[i] for i in random_indices]
#english_data = english_data[0:10000]


# In[6]:


foreign_data_string = " ".join(foreign_data)
english_data_string = " ".join(english_data)


# In[7]:


print("Liczba słów: ",round((len(foreign_data_string.split()) + len(english_data_string.split()))/1000000,1), "mln")


# In[8]:


with open (path+'\\list.txt','w',encoding = 'utf-8') as proc_seqf:
    for a, am in zip(english_data, foreign_data):
        proc_seqf.write("{}\t{}".format(a, am)+"\n")


# In[9]:


# Where we will store the data
input_texts = [] # sentence in original language
target_texts = [] # sentence in target language
target_texts_inputs = [] # sentence in target language offset by 1


# load in the data
# download the data at: http://www.manythings.org/anki/
t = 0
for line in open(path+'\\list.txt', encoding = 'utf-8'):
  # only keep a limited number of samples
  t += 1
  if t > NUM_SAMPLES:
    break

  # input and target are separated by tab
  if '\t' not in line:
    continue

  # split up the input and translation
  input_text, translation, *rest = line.rstrip().split('\t')

  # make the target input and output
  # recall we'll be using teacher forcing
  target_text = translation + ' <eos>'
  target_text_input = '<sos> ' + translation

  input_texts.append(input_text)
  target_texts.append(target_text)
  target_texts_inputs.append(target_text_input)
print("num samples:", len(input_texts))


# tokenize the inputs
tokenizer_inputs = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer_inputs.fit_on_texts(input_texts)
input_sequences = tokenizer_inputs.texts_to_sequences(input_texts)

# get the word to index mapping for input language
word2idx_inputs = tokenizer_inputs.word_index
print('Found %s unique input tokens.' % len(word2idx_inputs))

# tokenize the outputs
# don't filter out special characters
# otherwise <sos> and <eos> won't appear
tokenizer_outputs = Tokenizer(num_words=MAX_NUM_WORDS, filters='')
tokenizer_outputs.fit_on_texts(target_texts + target_texts_inputs) # inefficient, oh well
target_sequences = tokenizer_outputs.texts_to_sequences(target_texts)
target_sequences_inputs = tokenizer_outputs.texts_to_sequences(target_texts_inputs)

# get the word to index mapping for output language
word2idx_outputs = tokenizer_outputs.word_index
print('Found %s unique output tokens.' % len(word2idx_outputs))


# In[10]:


max_num_words = min(MAX_NUM_WORDS, len(word2idx_outputs), len(word2idx_inputs))


# In[11]:


# determine maximum length input sequence
max_len_input = max(len(s) for s in input_sequences)

# store number of output words for later
# remember to add 1 since indexing starts at 1
num_words_output = len(word2idx_outputs) + 1

# determine maximum length output sequence
max_len_target = max(len(s) for s in target_sequences)


# In[12]:


max_len_input_final = min(max_len_input, MAX_LEN_INPUT)
max_len_target_final = min(max_len_target, MAX_LEN_TARGET)


# In[13]:


def generate_batch(X = input_sequences,  y_input = target_sequences_inputs, y_output = target_sequences , batch_size = 128):

    while True:
        for j in range(0,len(X),batch_size):
            encoder_inputs = pad_sequences(X[j:j+batch_size], maxlen=max_len_input_final)
            #print("encoder_inputs.shape:", encoder_inputs.shape)
            #print("encoder_inputs[0]:", encoder_inputs[0])

            decoder_inputs = pad_sequences(y_input[j:j+batch_size], maxlen=max_len_target_final, padding='post')
            #print("decoder_inputs[0]:", decoder_inputs[0])
            #print("decoder_inputs.shape:", decoder_inputs.shape)

            decoder_targets = pad_sequences(y_output[j:j+batch_size], maxlen=max_len_target_final, padding='post')

            # pad the sequences
        
# create targets, since we cannot use sparse
# categorical cross entropy when we have sequences
            decoder_targets_one_hot = np.zeros(
                  (
                    batch_size,
                    max_len_target_final,
                    max_num_words
                  ),
                  dtype='float32'
                )
# assign the values
            for i, d in enumerate(decoder_targets):
                for t, word in enumerate(d):
                    if word != 0:
                        decoder_targets_one_hot[i, t, word] = 1
            #print("decoder_targets_one_hot.shape:", decoder_targets_one_hot.shape)
            yield([encoder_inputs, decoder_inputs], decoder_targets_one_hot)


# In[14]:


EMBEDDING_DIM = 100


# In[15]:


# store all the pre-trained word vectors
print('Loading word vectors...')
word2vec = {}
with open(os.path.join(path, 'glove.6B.%sd.txt' % EMBEDDING_DIM),encoding = 'utf-8' ) as f:
  # is just a space-separated text file in the format:
  # word vec[0] vec[1] vec[2] ...
  for line in f:
    values = line.split()
    word = values[0]
    vec = np.asarray(values[1:], dtype='float32')
    word2vec[word] = vec
print('Found %s word vectors.' % len(word2vec))

# prepare embedding matrix
print('Filling pre-trained embeddings...')
num_words = max_num_words
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word2idx_inputs.items():
  if i < MAX_NUM_WORDS:
    embedding_vector = word2vec.get(word)
    if embedding_vector is not None:
      # words not found in embedding index will be all zeros.
      embedding_matrix[i] = embedding_vector


# In[16]:


LATENT_DIM = 256  # Latent dimensionality of the encoding space.


# In[17]:


# create embedding layer
embedding_layer = Embedding(
  max_num_words,
  EMBEDDING_DIM,
  #weights=[embedding_matrix],
  input_length=max_len_input_final,
  trainable=True
)

##### build the model #####
encoder_inputs_placeholder = Input(shape=(max_len_input_final,))
x = embedding_layer(encoder_inputs_placeholder)
encoder = LSTM(
  LATENT_DIM,
  return_state=True,
  # dropout=0.5 # dropout not available on gpu
)
encoder_outputs, h, c = encoder(x)
# encoder_outputs, h = encoder(x) #gru

# keep only the states to pass into decoder
encoder_states = [h, c]
# encoder_states = [state_h] # gru

# Set up the decoder, using [h, c] as initial state.
decoder_inputs_placeholder = Input(shape=(max_len_target_final,))

# this word embedding will not use pre-trained vectors
# although you could
decoder_embedding = Embedding(max_num_words, EMBEDDING_DIM)
decoder_inputs_x = decoder_embedding(decoder_inputs_placeholder)

# since the decoder is a "to-many" model we want to have
# return_sequences=True
decoder_lstm = LSTM(
  LATENT_DIM,
  return_sequences=True,
  return_state=True,
  # dropout=0.5 # dropout not available on gpu
)
decoder_outputs, _, _ = decoder_lstm(
  decoder_inputs_x,
  initial_state=encoder_states
)

# decoder_outputs, _ = decoder_gru(
#   decoder_inputs_x,
#   initial_state=encoder_states
# )

# final dense layer for predictions
decoder_dense = Dense(max_num_words, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Create the model object
model = Model([encoder_inputs_placeholder, decoder_inputs_placeholder], decoder_outputs)


def custom_loss(y_true, y_pred):
  # both are of shape N x T x K
  mask = K.cast(y_true > 0, dtype='float32')
  out = mask * y_true * K.log(y_pred)
  return -K.sum(out) / K.sum(mask)


def acc(y_true, y_pred):
  # both are of shape N x T x K
  targ = K.argmax(y_true, axis=-1)
  pred = K.argmax(y_pred, axis=-1)
  correct = K.cast(K.equal(targ, pred), dtype='float32')

  # 0 is padding, don't include those
  mask = K.cast(K.greater(targ, 0), dtype='float32')
  n_correct = K.sum(mask * correct)
  n_total = K.sum(mask)
  return n_correct / n_total

model.compile(optimizer='adam', loss=custom_loss, metrics=[acc])

# Compile the model and train it
# model.compile(
#   optimizer='rmsprop',
#   loss='categorical_crossentropy',
#   metrics=['accuracy']
# )

# In[18]:


train_samples = 0.8*NUM_SAMPLES# Total Training samples
val_samples = 0.2*NUM_SAMPLES   # Total validation or test samples
batch_size = 64  # Batch size for training.
epochs = 1  # Number of epochs to train for.


# In[19]:


r = model.fit(
  generate_batch(X = input_sequences,  y_input = target_sequences_inputs, y_output = target_sequences , batch_size = batch_size),
  steps_per_epoch = train_samples//batch_size,  
  epochs=epochs
  #validation_data = generate_batch(X_test, y_test, batch_size = batch_size),
  #validation_steps = val_samples//batch_size
)


# In[20]:



##### Make predictions #####
# As with the poetry example, we need to create another model
# that can take in the RNN state and previous word as input
# and accept a T=1 sequence.

# The encoder will be stand-alone
# From this we will get our initial decoder hidden state
encoder_model = Model(encoder_inputs_placeholder, encoder_states)

decoder_state_input_h = Input(shape=(LATENT_DIM,))
decoder_state_input_c = Input(shape=(LATENT_DIM,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
# decoder_states_inputs = [decoder_state_input_h] # gru

decoder_inputs_single = Input(shape=(1,))
decoder_inputs_single_x = decoder_embedding(decoder_inputs_single)

# this time, we want to keep the states too, to be output
# by our sampling model
decoder_outputs, h, c = decoder_lstm(
  decoder_inputs_single_x,
  initial_state=decoder_states_inputs
)
# decoder_outputs, state_h = decoder_lstm(
#   decoder_inputs_single_x,
#   initial_state=decoder_states_inputs
# ) #gru
decoder_states = [h, c]
# decoder_states = [h] # gru
decoder_outputs = decoder_dense(decoder_outputs)

# The sampling model
# inputs: y(t-1), h(t-1), c(t-1)
# outputs: y(t), h(t), c(t)
decoder_model = Model(
  [decoder_inputs_single] + decoder_states_inputs, 
  [decoder_outputs] + decoder_states
)

# map indexes back into real words
# so we can view the results
idx2word_eng = {v:k for k, v in word2idx_inputs.items()}
idx2word_trans = {v:k for k, v in word2idx_outputs.items()}



# In[25]:


decoder_model.save('decoder_model.h5')


# In[26]:


encoder_model.save('encoder_model.h5')


# In[36]:


def decode_sequence(input_seq):
  # Encode the input as state vectors.
  states_value = encoder_model.predict(input_seq)

  # Generate empty target sequence of length 1.
  target_seq = np.zeros((1, 1))

  # Populate the first character of target sequence with the start character.
  # NOTE: tokenizer lower-cases all words
  target_seq[0, 0] = word2idx_outputs['<sos>']

  # if we get this we break
  eos = word2idx_outputs['<eos>']

  # Create the translation
  output_sentence = []
  for _ in range(max_len_target_final):
    output_tokens, h, c = decoder_model.predict(
      [target_seq] + states_value
    )
    # output_tokens, h = decoder_model.predict(
    #     [target_seq] + states_value
    # ) # gru

    # Get next word
    idx = np.argmax(output_tokens[0, 0, :])

    # End sentence of EOS
    if eos == idx:
      break

    word = ''
    if idx > 0:
      word = idx2word_trans[idx]
      output_sentence.append(word)

    # Update the decoder input
    # which is just the word just generated
    target_seq[0, 0] = idx

    # Update states
    states_value = [h, c]
    # states_value = [h] # gru

  return ' '.join(output_sentence)


# In[37]:


def str_to_tokens( sentence : str ):
    words = sentence.lower().split()
    tokens_list = list()
    for word in words:
        tokens_list.append( word2idx_inputs[ word ] ) 
    return pad_sequences( [tokens_list] , maxlen=max_len_input_final)


# In[ ]:


while True:
  try:
      # Do some test translations
      #i = np.random.choice(len(input_texts))
      input_seq = input("English text: ")
      input_seq = str_to_tokens(input_seq)
      translation = decode_sequence(input_seq)
      print('-')
      #print('Input:', input_texts[i])
      print('Translation:', translation)

      ans = input("Continue? [Y/n]")
      if ans and ans.lower().startswith('n'):
        break
  except:
      
        print("Sorry, I don't understand. Please try again.")
      


# In[ ]:


# plot some data
# plt.plot(r.history['loss'], label='loss')
# plt.plot(r.history['val_loss'], label='val_loss')
# plt.legend()
# plt.show()

# # accuracies
# plt.plot(r.history['accuracy'], label='acc')
# plt.plot(r.history['val_accuracy'], label='val_acc')
# plt.legend()
# plt.show()

