# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 10:35:58 2020

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
text_cleanse = lr_space(text).split()
