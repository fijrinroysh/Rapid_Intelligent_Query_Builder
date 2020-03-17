import operator
from flask import Flask, render_template, request,jsonify
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False
@app.route('/')
def student():
   return render_template('interface.html')

@app.route('/',methods = ['POST', 'GET'])
def result():
    if request.method == 'POST':
      result = request.form
      item=list(result.values())
      class TrieNode(): 
       def __init__(self): 
          
        # Initialising one node for trie 
        self.children = {} 
        self.last = False
      class Trie(): 
       def __init__(self): 
          
        # Initialising the trie structure. 
        self.root = TrieNode() 
        self.word_list = [] 
  
       def formTrie(self, keys): 
          
        # Forms a trie structure with the given set of strings 
        # if it does not exists already else it merges the key 
        # into it by extending the structure as required 
        for key in keys: 
            self.insert(key) # inserting one key to the trie. 
  
       def insert(self, key): 
          
        # Inserts a key into trie if it does not exist already. 
        # And if the key is a prefix of the trie node, just  
        # marks it as leaf node. 
        node = self.root 
  
        for a in list(key): 
            if not node.children.get(a): 
                node.children[a] = TrieNode() 
  
            node = node.children[a] 
  
        node.last = True
  
       def search(self, key): 
          
        # Searches the given key in trie for a full match 
        # and returns True on success else returns False. 
        node = self.root 
        found = True
  
        for a in list(key): 
            if not node.children.get(a): 
                found = False
                break
  
            node = node.children[a] 
  
        return node and node.last and found 
  
       def suggestionsRec(self, node, word): 
          
        # Method to recursively traverse the trie 
        # and return a whole word.  
        if node.last: 
            self.word_list.append(word) 
  
        for a,n in node.children.items(): 
            self.suggestionsRec(n, word + a) 
  
       def printAutoSuggestions(self, key): 
          
        # Returns all the words in the trie whose common 
        # prefix is the given key thus listing out all  
        # the suggestions for autocomplete. 
        node = self.root 
        not_found = False
        temp_word = '' 
  
        for a in list(key): 
            if not node.children.get(a): 
                not_found = True
                break
  
            temp_word += a 
            node = node.children[a] 
  
        if not_found: 
            return 0
        elif node.last and not node.children: 
            return -1
  
        self.suggestionsRec(node, temp_word) 
  
        for x in  self.word_list :
                       for y in keys:
                             if(x==y):
                                   filtered.update({y:keys[y]})                      
        sorted_d = dict( sorted(filtered.items(), key=operator.itemgetter(1),reverse=True)) 
        #sorted_l=jsonify(sorted_d)                    
        return sorted_d
      for i in item:
           key=i.lower()
      keys = {'computer':90,'computer keyboard':80,'computer networks':84,'computer science':85,'computer application':89,'bookmyshow':65,'boot':70,'bsnl':80,'bsnl recharge':90,'bsnl recharge plans':100} # keys to form the trie structure. 
      filtered={}
      list_keys=[]
      t = Trie() 
      t.formTrie(keys) 
      comp = t.printAutoSuggestions(key)
      list_keys = [ k for k in comp ]
      if comp == -1: 
          print("No other strings found with this prefix\n") 
      elif comp == 0: 
          print("No string found with this prefix\n") 
      return render_template('interface.html', list_keys=list_keys)  
if __name__ == '__main__':
   app.run()

# -*- coding: utf-8 -*-

