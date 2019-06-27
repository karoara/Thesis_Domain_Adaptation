# DATA UTILITIES --------------------------------------------------------------
# 
# Functions for loading, preparing data for training & testing models.


# imports
import numpy as np
import nltk
from train_utils import write


# function for getting data
def DA_load_data(filename, domains_wanted=None):
  
  f = open(filename, "r")
  data = f.readlines()
  data = [l.strip() for l in data]
  
  sents = []
  labels = []
  domains = []
  
  for l in data:
    label, sent, domain = l.split('\t')
    if domains_wanted == None:
      labels.append(float(label))
      sents.append(nltk.word_tokenize(sent))
      domains.append(int(domain))
    else:
      if int(domain) in domains_wanted:
        labels.append(float(label))
        sents.append(nltk.word_tokenize(sent))
        domains.append(int(domain))
  
  num_domains, counts = np.unique(domains, return_counts=True)
  num_per_domain = dict(zip(num_domains, counts))
  
  return sents, labels, domains, num_per_domain


# function for getting embeddings for words in the datasets
def get_data_embeddings(all_data, words, vectors, lookup):
  
  dist_words = distinct_words(all_data)
  data_embedding_words = []
  dist_vectors = np.zeros((len(dist_words), vectors.shape[1]))

  for i in range(len(dist_words)):
    if (dist_words[i] in words): data_embedding_words.append(dist_words[i])
  dist_words = np.asarray(data_embedding_words)
  
  for i in range(len(dist_words)):
    dist_vectors[i] = vectors[lookup[dist_words[i]]]
  
  dist_lookup = dict(zip(dist_words, np.arange(len(dist_vectors))))
  
  return dist_words, dist_vectors, dist_lookup


# function that retrieves words in data for which we have embeddings
def get_rel_words(all_data, words, vectors, lookup, mds):
  
  if (mds == False):
    dist_words = distinct_words(all_data)
  else:
    dist_words = set()
    for i in range(len(all_data)):
      dist_words.add(all_data[i])
    dist_words = np.asarray(list(dist_words))
  
  dist_emb_words = []
  for i in range(len(dist_words)):
    if (dist_words[i] in words): dist_emb_words.append(dist_words[i])
  
  return dist_emb_words


# function that replaces oov words in data with oov token
def replace_oov(data, domains, rel_words, oov):

  new_data = []
  new_domains = []

  for i in range(len(data)):
    sentence = data[i]
    new_sentence = []
    oov_count = 0

    for w in range(len(sentence)):
      word = sentence[w]
      if word in rel_words: 
        new_sentence.append(word)
      else: 
        new_sentence.append(oov)
        oov_count += 1
    
    if oov_count < len(sentence) and len(new_sentence) > 1:
      new_data.append(new_sentence)
      new_domains.append(domains[i])
  
  return new_data, new_domains


# function that gets vector indices for words in data
def get_sent_vect_indices(x_data_sents, y_data, d_data, rel_words, vectors, lookup, trans_words):
  
  sentences = []
  sentence_vector_indices = []
  labels = []
  domains = []
  if trans_words != None: sentence_vector_trans = []
  else: sentence_vector_trans = None
  
  for i in range(len(x_data_sents)):
    
    sentence = x_data_sents[i]
    sentence_vect_indices = []
    if trans_words != None: sentence_vect_trans = []
    
    for w in range(len(sentence)):
      if sentence[w] in rel_words: 
        sentence_vect_indices.append(lookup[sentence[w]])
        if trans_words != None:
          if sentence[w] in trans_words: sentence_vect_trans.append(1)
          else: sentence_vect_trans.append(0)
    
    if (len(sentence_vect_indices) != 0):
      sentences.append(sentence)
      sentence_vector_indices.append(sentence_vect_indices)
      if trans_words != None: sentence_vector_trans.append(sentence_vect_trans)
      labels.append(y_data[i])
      domains.append(d_data[i])
  
  return sentences, labels, domains, sentence_vector_indices, sentence_vector_trans


# function that applies given matrix transformation to sentences (sequences of word vectors)
def transform_sent_vects(sentences, sentence_vect_trans, domains, transforms):
  
  new_sentences = []
  
  for i in range(len(sentences)):
    old_sent = sentences[i]
    new_sent = np.zeros((old_sent.shape[0], old_sent.shape[1]))
    sentence_trans = sentence_vect_trans[i]
    
    for w in range(len(old_sent)):
      if sentence_trans[w] == 1: new_sent[w] = np.matmul(old_sent[w], transforms[int(domains[i])])
      else: new_sent[w] = old_sent[w]
    
    new_sentences.append(new_sent)    
  
  return new_sentences


# function that gets vector indices for words in data (doesn't check for no-embedding sentences)
def get_sent_vect_indices_2(x_data_sents, rel_words, vectors, lookup):
  
  sentences = []
  sentence_vector_indices = []
  
  for i in range(len(x_data_sents)):
    sentence_vect_indices = []
    
    for w in range(len(x_data_sents[i])):
      if(x_data_sents[i][w] in rel_words): 
        sentence_vect_indices.append(lookup[x_data_sents[i][w]])
    
    sentences.append(x_data_sents[i])
    sentence_vector_indices.append(sentence_vect_indices)
  
  return sentences, sentence_vector_indices


# function that gets vector indices for words in data (for language modeling)
def get_sent_vect_indices_lang(x_data_sents, vectors, lookup, test=False, only_test=None):
  
  X, Y = [], []
  
  for i in range(len(x_data_sents)):
    sentence = x_data_sents[i]
    sentence_vect_indices = np.zeros(len(sentence))
    
    for w in range(len(sentence)):
      word = sentence[w]
      sentence_vect_indices[w] = lookup[word]
    
    X.append(sentence_vect_indices[:len(sentence)-1])
    Y.append(sentence_vect_indices[1:len(sentence)])
  
  return X, Y


# function that retrieves sentence vectors for a dataset
def get_sentence_vectors(sentence_vector_indices, vectors):
  
  sentence_vectors = []
  
  for i in range(len(sentence_vector_indices)):
    sentence = np.zeros((len(sentence_vector_indices[i]), vectors.shape[1]))
    
    for w in range(len(sentence_vector_indices[i])):
      sentence[w] = vectors[int(sentence_vector_indices[i][w])]
    
    sentence_vectors.append(sentence)
  
  return sentence_vectors


# function that averages sentence words for LR/NN with fixed input size
def avg_for_LR_NN(x_data):
  
  x_data_avgd = []
  
  for i in range(len(x_data)):
    x_data_avgd.append(np.mean(x_data[i], axis=0))
  
  return x_data_avgd


# function that returns list of distinct words in a dataset
def distinct_words(data):
  
  words = set()
  
  for i in range(len(data)):
    for w in range(len(data[i])):
      words.add(data[i][w])
  words = np.asarray(list(words))
  
  return words


# function that returns % of distinct words from dataset that we have embeddings for
def in_embeddings(words, distinct_words):
  
  in_embeddings = 0
  
  for i in range(len(distinct_words)):
    if (distinct_words[i] in words): in_embeddings+=1
  percentage = (in_embeddings/len(distinct_words)) * 100
  
  return percentage


# function that balances data
def balance(sentences, labels, domains):
  
  pos_s = [sentences[i] for i in range(len(sentences)) if labels[i] == 1]
  pos_l = [label for label in labels if label == 1]
  pos_d = [domains[i] for i in range(len(sentences)) if labels[i] == 1]
  neg_s = [sentences[i] for i in range(len(sentences)) if labels[i] == 0]
  neg_l = [label for label in labels if label == 0]
  neg_d = [domains[i] for i in range(len(sentences)) if labels[i] == 0]
  
  if (len(pos_s) > len(neg_s)):
    
    new_neg_s = []
    new_neg_l = []
    new_neg_d = []
    
    q = len(pos_s) // len(neg_s)
    r = len(pos_s) % len(neg_s)
    
    for i in range(q):
      new_neg_s += neg_s
      new_neg_l += neg_l
      new_neg_d += neg_d
    
    rands = np.random.permutation(len(neg_s))
    new_neg_s += list(np.asarray(neg_s)[rands[:r]])
    new_neg_l += list(np.asarray(neg_l)[rands[:r]])
    new_neg_d += list(np.asarray(neg_d)[rands[:r]])
    
    sentences = pos_s + new_neg_s
    labels = pos_l + new_neg_l
    domains = pos_d + new_neg_d
  
  elif (len(pos_s) < len(neg_s)):
    
    new_pos_s = []
    new_pos_l = []
    new_pos_d = []
    
    q = len(neg_s) // len(pos_s)
    r = len(neg_s) % len(pos_s)
    
    for i in range(q):
      new_pos_s += pos_s
      new_pos_l += pos_l
      new_pos_d += pos_d
    
    rands = np.random.permutation(len(pos_s))
    new_pos_s += list(np.asarray(pos_s)[rands[:r]])
    new_pos_l += list(np.asarray(pos_l)[rands[:r]])
    new_pos_d += list(np.asarray(pos_d)[rands[:r]])
    
    sentences = new_pos_s + neg_s
    labels = new_pos_l + neg_l
    domains = new_pos_d + neg_d
  
  return sentences, labels, domains
