#!/usr/bin/env python
# coding: utf-8

# In[9]:


# importing necessary libraries and functions
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import json
#model = load_model("s2s.h5", compile=False)

max_len_input_final = 50
max_len_target_final = 30

app = Flask(__name__) #Initialize the flask App
path = r"D:\Dokumenty\DeepNLP\machine_learning_examples\nlp_class3\OwnScripts\Flask\model_spec"

encoder_model = load_model(path +"\\encoder_model.h5", compile=False)
decoder_model = load_model(path +"\\decoder_model.h5", compile=False)

with open(path + "\\word2idx_inputs.txt") as file:
    word2idx_inputs = json.loads(file.read())

with open(path + "\\word2idx_outputs.txt") as file:
    word2idx_outputs = json.loads(file.read())

idx2word_eng = {v:k for k, v in word2idx_inputs.items()}
idx2word_trans = {v:k for k, v in word2idx_outputs.items()}

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

def str_to_tokens( sentence : str ):
    words = sentence.lower().split()
    tokens_list = list()
    for word in words:
        tokens_list.append( word2idx_inputs[ word ] ) 
    return pad_sequences( [tokens_list] , maxlen=max_len_input_final)

@app.route('/') # Homepage
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    input_seq = str_to_tokens(request.form.getlist('english_text')[0])
    translation = decode_sequence(input_seq)
    #print('-')
    #print('Input:', input_texts[i])
    #print('Translation:', translation)

    return render_template('index.html', prediction_text='Translation: {}'.format(translation)) # rendering the predicted result

if __name__ == "__main__":
    app.run(debug=True)


# In[ ]:




