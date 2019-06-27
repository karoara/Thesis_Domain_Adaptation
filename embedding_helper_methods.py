# EMBEDDING HELPER METHODS ----------------------------------------------------
# 
# Methods for getting word embeddings from file & manipulating them. 


# imports
import numpy as np


# function for getting word embeddings from file
def read_embeddings(fn='glove.6B.200d.txt', d=200):
  with open(fn, 'r') as f:
    embeddings = f.readlines()
    
    # set up word & vector arrays
    words = np.empty(shape=(len(embeddings)), dtype=object)
    vectors = np.zeros((len(embeddings), d))
    
    # parse lines from embedding file to get words, vectors
    for i in range(len(embeddings)): 
      words[i], vectors[i] = parse_embedding(embeddings[i], fn[:5])
    
    return words, vectors


# function for parsing lines into words and vectors
def parse_embedding(line, type):
  
  # split line string into values, create line array
  if type != "glove": line = line.rstrip("\n ")
  split_line = line.split(' ')
  line_array = np.empty(shape = (len(split_line)), dtype = object)
  
  # make vector values in the line array floats
  for i in range(len(split_line)):
    if (i == 0): line_array[i] = split_line[i]
    else: line_array[i] = float(split_line[i])
  
  return line_array[0], line_array[1:]


# function for normalizing embeddings to have unit length
def normalize_embeddings(vectors):

  # iterate over embeddings and normalize them
  normed_vectors = np.zeros(np.shape(vectors))
  for i in range(len(vectors)):
    normed_vectors[i] = vectors[i]/np.linalg.norm(vectors[i])
  
  return normed_vectors


