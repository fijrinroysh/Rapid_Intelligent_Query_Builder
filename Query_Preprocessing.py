# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 10:29:57 2020

@author: fijrin.j.roysh
"""
import codecs
import numpy as np

def lr_space(txt):
    stop_words = [',','.','\'','=','(',')',';']
    for i in stop_words:
        txt= txt.replace(i, " "+i+" ")
        
    return txt
    
    
with codecs.open(r'C:\Users\fijrin.j.roysh\Rapid_query_builder_api\Query_Dump\sample_remove_alias.txt', 'r',encoding='utf-8-sig') as f:
    text=f.read()
    
text_cleanse = lr_space(text).upper().split()

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