# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 01:15:06 2020

@author: fijrin.j.roysh
"""
import tensorflow as tf
import numpy as np
import codecs
import os


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

# The unique characters in the file
vocab = sorted(set(text_cleanse))
print ('{} unique words'.format(len(vocab)))
print ('Number of words: {} characters'.format(len(text_cleanse)))
#print(vocab)
#print(enumerate(vocab))
vocab_to_int = {c: i for i, c in enumerate(vocab)}


#print(vocab_to_int)
#int_to_vocab = dict(enumerate(vocab))
int_to_vocab = np.array(vocab)


encoded = np.array([vocab_to_int[c] for c in text_cleanse], dtype=np.int32)


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
print ('{} ---- characters mapped to int ---- > {}'.format(repr(text[:13]), encoded[:13]))

# The maximum length sentence we want for a single input in characters
seq_length = 10
examples_per_epoch = len(text)//(seq_length+1)

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
print(checkpoint_prefix)
checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)




history = model.fit(dataset, epochs=10, callbacks=[checkpoint_callback])

tf.train.latest_checkpoint(checkpoint_dir)