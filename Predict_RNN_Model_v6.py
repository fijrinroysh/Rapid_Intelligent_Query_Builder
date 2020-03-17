# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 17:15:48 2020

@author: fijrin.j.roysh
"""

from Query_Preprocessing import int_to_vocab, vocab_to_int,encoded
from Simple_Train_RNN_Model_v4 import build_model, checkpoint_dir, vocab_size,embedding_dim,rnn_units 
import tensorflow as tf
import pandas as pd

import operator

from flask import Flask, render_template, request

app = Flask(__name__)

app.config['JSON_SORT_KEYS'] = False
#app.config['DEBUG'] = True


@app.route('/')

def student():

   return render_template('searchbar.html')


@app.route('/possiblelist',methods = ['POST', 'GET'])

def result():
    if request.method == 'POST':
        result = request.form
        print(result)
        #print(x.upper()) for x in list(result.values())
        item=list(result.values())
        print(item)
        item=[x.upper() for x in item[0].split()]
        print(item)
                
        def generate_text(model, start_list):
          # Evaluation step (generating text using the learned model)
                          
          # Converting our start string to numbers (vectorizing)
          input_eval = [vocab_to_int[s] for s in start_list]
          input_eval = tf.expand_dims(input_eval, 0)
        
          # Empty string to store our results
          
          # Low temperatures results in more predictable text.
          # Higher temperatures results in more surprising text.
          # Experiment to find the best setting.
          temperature = 0.1
          text_generated=[]
          
          # Here batch size == 1
          model.reset_states()
          
          predictions = model.predict(input_eval)
          # remove the batch dimension
          predictions = tf.squeeze(predictions, 0)
        
          # using a categorical distribution to predict the character returned by the model
          predictions = predictions / temperature
          
          #predicted_id = tf.random.categorical(predictions, num_samples=4)[-1,:].numpy()
          predicted_last_batch=predictions[-1,:].numpy()
          predicted_last_batch=pd.DataFrame(predicted_last_batch).reset_index()
          
          predicted_last_batch_top=predicted_last_batch.sort_values(by=0,ascending=False)['index'].head(4)
          
          # We pass the predicted character as the next input to the model
          # along with the previous hidden state
          #input_eval = tf.expand_dims([predicted_id[-1]], 0)
          
          text_generated=[" ".join(start_list)+" "+int_to_vocab[x]  for x in predicted_last_batch_top]
          #print(text_generated)
          return text_generated

        model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)
        print("Latest checkpoint is:" +tf.train.latest_checkpoint(checkpoint_dir))
        model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
        
        model.build(tf.TensorShape([1, None]))
        
        #model.summary()
        #item=['GRP_NM', 'MBR_ID']
        outupt_query=generate_text(model, item)
        
        print(outupt_query)
        
    return render_template('possiblelist.html', list_keys=outupt_query)  

if __name__ == '__main__':

   app.run()