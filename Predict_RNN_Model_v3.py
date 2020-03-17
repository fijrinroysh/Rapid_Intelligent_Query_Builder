# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 17:15:48 2020

@author: fijrin.j.roysh
"""

from Query_Preprocessing import int_to_vocab, vocab_to_int,encoded
from Simple_Train_RNN_Model_v4 import build_model, checkpoint_dir, vocab_size,embedding_dim,rnn_units 
import tensorflow as tf

from flask import Flask, render_template, request
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

@app.route('/')

def student():

   return render_template('searchbar.html')

@app.route('/possiblelist',methods = ['POST', 'GET'])
def result():
    if request.method == 'POST':
        result = request.form
        item=list(result.values())
        item=[x.upper() for x in item[0].split()]
        def generate_text(model, start_string):
          # Evaluation step (generating text using the learned model)
        
          # Number of characters to generate
          num_generate = 40
        
          # Converting our start string to numbers (vectorizing)
          input_eval = [vocab_to_int[s] for s in start_string]
          input_eval = tf.expand_dims(input_eval, 0)
        
          # Empty string to store our results
          text_generated = []
          text_generated.extend(start_string)
          # Low temperatures results in more predictable text.
          # Higher temperatures results in more surprising text.
          # Experiment to find the best setting.
          temperature = 0.1
        
          # Here batch size == 1
          model.reset_states()
          for i in range(num_generate) :
              predictions = model(input_eval)
              # remove the batch dimension
              predictions = tf.squeeze(predictions, 0)
        
              # using a categorical distribution to predict the character returned by the model
              predictions = predictions / temperature
              print(predictions[-1,0].numpy())
              predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()
    #          print(predicted_id)
              # We pass the predicted character as the next input to the model
              # along with the previous hidden state
              input_eval = tf.expand_dims([predicted_id], 0)
        
              text_generated.append(int_to_vocab[predicted_id])
          empty_space    = ' '
          print(empty_space.join(text_generated))    
          return (empty_space.join(text_generated))
                
        
        print("Latest checkpoint is:" +tf.train.latest_checkpoint(checkpoint_dir))
        
        model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)
        
        model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
        
        model.build(tf.TensorShape([1, None]))
        
        #model.summary()
        
        outupt_query=[]
        outupt_query.append(generate_text(model, start_string=item))
        print(outupt_query)
        
    return render_template('possiblelist.html', list_keys=outupt_query)  
if __name__ == '__main__':
   app.run()