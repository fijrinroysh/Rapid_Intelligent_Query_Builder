# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 10:11:15 2020

@author: fijrin.j.roysh
"""


# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 01:15:06 2020

@author: fijrin.j.roysh
"""
import tensorflow as tf
import numpy as np

import os
from Query_Preprocessing import vocab, int_to_vocab, vocab_to_int,encoded, text_cleanse
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
import pandas as pd
from pylab import rcParams



sns.set(style='whitegrid', palette='muted', font_scale=1.5)



## Creating a mapping from unique characters to indices
#char2idx = {u:i for i, u in enumerate(vocab)}
#idx2char = np.array(vocab)
#
#text_as_int = np.array([char2idx[c] for c in text])

#print('{')
#for char,_ in zip(int_to_vocab, range(20)):
#    print('  {:4s}: {:3d},'.format(repr(char), int_to_vocab[char]))
#print('  ...\n}')

# Show how the first 13 characters from the text are mapped to integers
print ('{} ---- characters mapped to int ---- > {}'.format(repr(text_cleanse[:13]), encoded[:13]))

# The maximum length sentence we want for a single input in characters
seq_length = 10
examples_per_epoch = len(text_cleanse)//(seq_length+1)

# Create training examples / targets
char_dataset = tf.data.Dataset.from_tensor_slices(encoded)


for i in char_dataset.take(5):
  print(int_to_vocab[i.numpy()])


sequences = char_dataset.batch(seq_length+1, drop_remainder=True)

for item in sequences.take(5):
    #print(repr(''.join(int_to_vocab[item.numpy()])))
    print(int_to_vocab[item.numpy()])


def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

dataset = sequences.map(split_input_target)

for input_example, target_example in  dataset.take(1):
  print ('Input data: ', repr(' '.join(int_to_vocab[input_example.numpy()])))
  print ('Target data:', repr(' '.join(int_to_vocab[target_example.numpy()])))

# Batch size
BATCH_SIZE = 10

# Buffer size to shuffle the dataset
# (TF data is designed to work with possibly infinite sequences,
# so it doesn't attempt to shuffle the entire sequence in memory. Instead,
# it maintains a buffer in which it shuffles elements).
BUFFER_SIZE = 10000

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
print(dataset)




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
    tf.keras.layers.LSTM(rnn_units,
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





for input_example_batch, target_example_batch in dataset.take(4):
    example_batch_predictions = model(input_example_batch, np.float32)
    print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")
    
    
model.summary()

sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
sampled_indices = tf.squeeze(sampled_indices,axis=-1).numpy()



def loss(labels, logits):
 return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

example_batch_loss  = loss(target_example_batch, example_batch_predictions)
print("Prediction shape: ", example_batch_predictions.shape, " # (batch_size, sequence_length, vocab_size)")
print("scalar_loss:      ", example_batch_loss.numpy().mean())

model.compile(optimizer='adam', loss=loss,metrics=['accuracy'])
# Directory where the checkpoints will be saved
checkpoint_dir = './training_checkpoints'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
print(checkpoint_prefix)
checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)




#history = model.fit(dataset, epochs=10, callbacks=[checkpoint_callback]).history
#model.save('keras_model.h5')
#pickle.dump(history, open("history.p", "wb"))
#loss_df=pd.DataFrame(history)
#loss_df.plot()






