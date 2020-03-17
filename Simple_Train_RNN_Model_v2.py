# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 11:20:19 2020

@author: fijrin.j.roysh
"""

import time
import os
from collections import namedtuple
import codecs
import numpy as np
import tensorflow as tf

print("TensorFlow version: ", tf.__version__)



def lr_space(vocab, word):
    return vocab.replace(word, " "+word+" ")
    
    
with codecs.open(r'C:\Users\fijrin.j.roysh\Downloads\Ann\New folder\sample.txt', 'r',encoding='utf-8-sig') as f:
    text=f.read()
text_cleanse1 = lr_space(text,',')
text_cleanse2 = lr_space(text_cleanse1,'.')
text_cleanse3 = lr_space(text_cleanse2,'\'')
text_cleanse4 = lr_space(text_cleanse3,'=')
text_cleanse5 = lr_space(text_cleanse4,'(')
text_cleanse6 = lr_space(text_cleanse5,')')
text_cleanse7 = lr_space(text_cleanse6,';')
text_cleanse = text_cleanse7.split()

vocab = set(text_cleanse)
#print(vocab)
#print(enumerate(vocab))
vocab_to_int = {c: i for i, c in enumerate(vocab)}


#print(vocab_to_int)
int_to_vocab = dict(enumerate(vocab))
encoded = np.array([vocab_to_int[c] for c in text_cleanse], dtype=np.int32)


#print(text[:100])
#print(encoded[:100])
#print(len(vocab))
#print(encoded)

def get_batches(arr, n_seqs, n_steps):
    '''Create a generator that returns batches of size
       n_seqs x n_steps from arr.
       
       Arguments
       ---------
       arr: Array you want to make batches from
       n_seqs: Batch size, the number of sequences per batch
       n_steps: Number of sequence steps per batch
    '''
    # Get the number of characters per batch and number of batches we can make
    characters_per_batch = n_seqs*n_steps
    n_batches = len(arr)//characters_per_batch
    
    # Keep only enough characters to make full batches
    arr = arr[:characters_per_batch*n_batches]
    
    # Reshape into n_seqs rows
    arr = arr.reshape((n_seqs,-1))
    #print(arr)
    
    for n in range(0, arr.shape[1], n_steps):
        # The features
        print(n)
        x = arr[:, n:n+n_steps]
        # The targets, shifted by one
        y = np.zeros_like(x)
        y[:, :-1], y[:, -1] = x[:, 1:], x[:, 0]
        yield x, y





BATCH_SIZE = 10         # Sequences per batch
num_steps = 50          # Number of sequence steps per batch
     # Size of hidden layers in LSTMs
num_layers = 2          # Number of LSTM layers
learning_rate = 0.001    # Learning rate
keep_prob = 0.5         # Dropout keep probability        



epochs = 20

# Length of the vocabulary in chars
vocab_size = len(vocab)

# The embedding dimension
embedding_dim = 256

# Number of RNN units
#lstm_size = 512    
rnn_units = 512

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
  model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim,
                              batch_input_shape=[batch_size, None]),
    tf.keras.layers.GRU(rnn_units,
                        return_sequences=True,
                        stateful=True,
                        recurrent_initializer='glorot_uniform'),
    tf.keras.layers.Dense(vocab_size)
  ])
  return model


model = build_model(
  vocab_size = len(vocab),
  embedding_dim=embedding_dim,
  rnn_units=rnn_units,
  batch_size=BATCH_SIZE)


counter=0
#        new_state = model.initial_state
#        loss = 0


for input_example_batch, target_example_batch in get_batches(encoded, BATCH_SIZE, num_steps):
    example_batch_predictions = model(tf.convert_to_tensor(input_example_batch, np.float32))
    print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")
    counter += 1
    print(counter)
    
model.summary()

sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
sampled_indices = tf.squeeze(sampled_indices,axis=-1).numpy()



def loss(labels, logits):
  return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

example_batch_loss  = loss(target_example_batch, example_batch_predictions)
print("Prediction shape: ", example_batch_predictions.shape, " # (batch_size, sequence_length, vocab_size)")
print("scalar_loss:      ", example_batch_loss.numpy().mean())

model.compile(optimizer='adam', loss=loss)
# Directory where the checkpoints will be saved
checkpoint_dir = './training_checkpoints'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
print(os.path)
checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)


dataset = get_batches(encoded, BATCH_SIZE, num_steps)

history = model.fit(dataset, epochs=1, callbacks=[checkpoint_callback])

