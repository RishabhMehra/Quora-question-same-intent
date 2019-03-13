# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 14:54:58 2018

@author: SAGAR
"""
import pickle
import nltk
from api_client import ApiClient

my_auth_key = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJleHAiOjE1NTQ2MjY5ODAsImlhdCI6MTUzOTA3NDk4MCwibmJmIjoxNTM5MDc0OTgwLCJpZGVudGl0eSI6MTF9.kbHzSmkCqCu3akW7bMM6Vj4flZ-CVGuRYixh-k2x0Ps"

client = ApiClient(auth_key = my_auth_key)

#Load the dataset that was obtained from API and pickeled beforehand
with open('dataset.pkl', 'rb') as handle:
    dataset = pickle.load(handle)
    
#Extract all questions tokenized as sentences from the dataset
sentences = []
for i in dataset:
    sentences+=nltk.sent_tokenize(i['question1'])
    sentences+=nltk.sent_tokenize(i['question2'])
    
#Tokenize as words and find their frequency and select top 10000
word_dist = nltk.FreqDist()
for s in sentences:
    word_dist.update([i.lower() for i in nltk.word_tokenize(s)])
    
word_dist = word_dist.most_common(10000)

#obtain the glove vectors for the 10000 words from the web service
embeddings_list = []
for i in range(10,100):
      embeddings_list += client.w2v([i[0] for i in word_dist[i*100:i*100+100]])
     
#with the word as the key and the glove vector as the value and pickle it
embeddings_index = {}
for i in embeddings_list:
    embeddings_index[i['word']] = i['vec']
    
with open('embeddings_index1.pkl', 'wb') as handle:
        pickle.dump(embeddings_index, handle, protocol=pickle.HIGHEST_PROTOCOL)