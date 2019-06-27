# EMBEDDING MODIFYING METHODS -------------------------------------------------
# 
# Methods for modifying word embeddings and searching for certain words. 


# imports
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from embedding_helper_methods import normalize_embeddings
from skipdict import SkipDict
from nltk import pos_tag


# function that projects embeddings onto subspace that captures the meaning of certain words
def word_salad_PCA(vectors, lookup, imp_words, k, diff=False):
  
  # if invalid # of components
  if (k > vectors.shape[1]): k = vectors.shape[1]
  
  # get vectors for meaning-significant words
  imp_vectors = np.zeros((len(imp_words), vectors.shape[1]))
  for i in range(len(imp_words)):
      imp_vectors[i] = vectors[lookup[imp_words[i]]]

  # use PCA to get basis that captures meaning for given words, project embeddings
  pca = PCA(n_components=k, svd_solver='full')
  pca.fit(imp_vectors)
  projected_vectors = pca.transform(vectors)
  
  return projected_vectors, pca.explained_variance_ratio_, pca.components_, pca.mean_


# function for getting discriminatory (+ive/-ive) words in data to do PCA on
def get_pca_words(words, stopwords, sentences, labels):
  
  # get entropy of set of sentences
  p_init = np.sum(labels)/len(labels)
  e_init = entropy_1att(p_init)
  word_probs = {}
  
  # get counts for non-stopwords that we have embeddings for
  for i in range(len(sentences)):
    sent_words = []
    for j in range(len(sentences[i])):
      if sentences[i][j] in words and sentences[i][j] not in stopwords \
        and sentences[i][j] not in sent_words:
        if sentences[i][j] in word_probs:
          word_probs[sentences[i][j]][0] += 1
          word_probs[sentences[i][j]][1] += 1/len(sentences)
        else:
          word_probs[sentences[i][j]] = [1, 1/len(sentences)]
        sent_words.append(sentences[i][j])
  
  word_p_probs = {}
  word_np_probs = {}
  words = list(word_probs.keys())
  
  # set probabilities of +ive sentence based on occurrence/lack of occurrence of word to zero
  for i in range(len(words)):
    word_p_probs[words[i]] = 0
    word_np_probs[words[i]] = 0
  
  # compute probabilities of +ive sentence based on occurrence/lack of occurrence of word
  for i in range(len(words)):
    for j in range(len(sentences)):
      if labels[j] == 1:
        if words[i] in sentences[j]:
          word_p_probs[words[i]] += 1/word_probs[words[i]][0]
        else:
          word_np_probs[words[i]] += 1/(len(sentences) - word_probs[words[i]][0])
  word_scores = {}
  
  # compute information gain scores for words
  for i in range(len(words)):
    att_probs = [word_probs[words[i]][1], 1 - word_probs[words[i]][1]]
    pos_probs = [word_p_probs[words[i]], word_np_probs[words[i]]]
    word_scores[words[i]] = e_init - entropy(words[i], att_probs, pos_probs)
  word_scores = SkipDict(word_scores)

  return word_scores


# function for getting words with certain tags
def filter_for_tags(words, sentences, tags=['JJ', 'JJS']):
  
  sent_words = []
  for word in words:
    for sentence in sentences:
      if word in sentence:
        tagged_sentence = pos_tag(sentence)
        for sentence_word, tag in tagged_sentence:
          if sentence_word == word and tag in tags:
            sent_words.append(word)
            break
        break
  
  return sent_words


# function for k-means clustering to measure "noisiness" of different words
def kmeans(words, vectors, lookup, k, seed=0, pca=0, pca_dims=10):
  
  # get vectors for words, set up k-means clustering
  imp_vectors = np.zeros((len(words), vectors.shape[1]))
  for i in range(len(words)):
    imp_vectors[i] = vectors[lookup[words[i]]]
  kmeans = KMeans(n_clusters=k, random_state=seed)
  
  # do pca either on all vectors, or on vectors for important words
  if pca == 1: 
    pca = PCA(n_components=pca_dims, svd_solver='full').fit(vectors)
    imp_vectors = pca.transform(imp_vectors)
  elif pca == 2:
    pca = PCA(n_components=pca_dims, svd_solver='full').fit(imp_vectors)
    imp_vectors = pca.transform(imp_vectors)
  
  # do clustering, get centers & labels
  kmeans.fit(imp_vectors)
  centers = kmeans.cluster_centers_
  labels = kmeans.labels_
  
  # get sets of words in each cluster, distance of each word from cluster center
  clusters = [[] for i in range(k)]
  cluster_mapping = dict(zip(words, labels))
  word_dists = {}
  for i in range(k):
    for j in range(len(words)):
      if cluster_mapping[words[j]] == i:
        clusters[i].append(words[j])
        word_dists[words[j]] = np.linalg.norm(centers[i] - imp_vectors[j])
  word_dists = SkipDict(word_dists)

  return clusters, word_dists


# function for filtering out "noisy" words with k-means clustering
def kmeans_filter(words, vectors, lookup, trials, cluster_nums):
  
  # get vectors for words
  imp_vectors = np.zeros((len(words), vectors.shape[1]))
  for i in range(len(words)):
    imp_vectors[i] = vectors[lookup[words[i]]]
  
  # set up distance test
  word_dists = {}
  for i in range(len(words)):
    word_dists[words[i]] = 0
  
  # iterate over and perform clustering tests
  for j in cluster_nums:
    for i in range(trials):
      clusters, trial_word_dists = kmeans(words, vectors, lookup, j, seed=i)
      for word in words:
        word_dists[word] += trial_word_dists[word]
  
  return SkipDict(word_dists)


# UTILITIES -------------------------------------------------------------------

# helper entropy function 1
def entropy_1att(p):
  
  if np.isclose(p, 0): return 0
  elif np.isclose(p, 1): return 0
  else: return -p * np.log2(p) - (1-p) * np.log2(1-p)
  
# helper entropy function 2
def entropy(word, att_probs, pos_probs):

  entropy = 0
  for i in range(len(att_probs)):
    entropy += att_probs[i]*entropy_1att(pos_probs[i])
  
  return entropy

# cosine similarity
def cosine(vect1, vect2):

  return np.dot(vect1, vect2)/(np.linalg.norm(vect1) * np.linalg.norm(vect2))

# method that computes "subspace alignment"
def alignment(ss1, ss2, axis=0, cos_or_dot=1):
  
  if axis == 0:
    alignments = np.zeros(ss1.shape[0])
    for i in range(ss1.shape[0]):
      if cos_or_dot == 1: alignments[i] = cosine(ss1[0], ss2[0])
      else: alignments[i] = np.dot(ss1[0], ss2[0])
  
  else:
    alignments = np.zeros(ss1.shape[1])
    for i in range(ss1.shape[1]):
      if cos_or_dot == 1: alignments[i] = cosine(ss1[:,0], ss2[:,0])
      else: alignments[i] = np.dot(ss1[:,0], ss2[:,0])
  
  return alignments

# function for printing cosine similarities of word pairs
def similarities(words, vectors, lookup):
  
  for i in range(len(words)):
    for j in range(len(words)):
      if i == j: continue
      print("Similarity between ", words[i], " and ", words[j], ": ", 
            cosine(vectors[lookup[words[i]]], vectors[lookup[words[j]]]))
    print("\n")