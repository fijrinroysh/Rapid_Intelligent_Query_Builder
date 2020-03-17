# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 17:15:48 2020

@author: fijrin.j.roysh
"""

from Query_Preprocessing import int_to_vocab, vocab_to_int,encoded
from Simple_Train_RNN_Model_v4 import build_model, checkpoint_dir, vocab_size,embedding_dim,rnn_units 
import tensorflow as tf
import numpy as np
import heapq 

def sample(preds, top_n=3):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds)
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    
    return heapq.nlargest(top_n, range(len(preds)), preds.take)

def generate_text(model, start_string):
  # Evaluation step (generating text using the learned model)



  # Converting our start string to numbers (vectorizing)
  input_eval = [vocab_to_int[s] for s in start_string.split()]
  input_eval = tf.expand_dims(input_eval, 0)

  # Empty string to store our results
  text_generated = []

  # Low temperatures results in more predictable text.
  # Higher temperatures results in more surprising text.
  # Experiment to find the best setting.
  temperature = 0.1

  # Here batch size == 1
  model.reset_states()
  
  predictions = model(input_eval)
  print(predictions.shape, "# (batch_size, sequence_length, vocab_size)")
      # remove the batch dimension
  predictions = tf.squeeze(predictions, 0)

      # using a categorical distribution to predict the character returned by the model
  predictions = predictions / temperature
      
  print(predictions.shape, "# ( batch_size, vocab_size)")
  preds = model.predict(input_eval, verbose=0)[0]
  print(preds.shape)
  predicted_id = sample(preds, top_n=1)[0]
  print(predicted_id)
      # We pass the predicted character as the next input to the model
      # along with the previous hidden state
  #input_eval = tf.expand_dims([predicted_id], 0)

  text_generated.append(int_to_vocab[predicted_id])

  return (start_string + ' '.join(text_generated))



print("Latest checkpoint is:" +tf.train.latest_checkpoint(checkpoint_dir))

model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)

model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

model.build(tf.TensorShape([1, None]))

model.summary()
samp = tf.random.categorical(tf.math.log([[0.5, 0.5]]), 5)
print(samp)

print(generate_text(model, start_string=u"\n\nSELECT MBR_ID GRP_NM"))