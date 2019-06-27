import sys
import numpy as np
import nltk
import math
from DA_experiments import experiments
np.random.seed(10)


# helper function for writing data to file
def write_data(filename, data):

  f = open(filename, "w")
  for i in range(len(data)): 
    if i == (len(data)-1): f.write(data[i])
    else: f.write(data[i] + '\n')
  f.close()


# helper function for flipping a label
def flip_label(label):

  new_label = str(int(math.fabs(int(label) - 1)))
  return new_label


# helper function for splitting domains into training, validation set
def train_val_split(domains_data, train_split):

  train_data, val_data = [], []
  for domain_data in domains_data:
    if len(domain_data) > 0:
      domain_data = np.asarray(domain_data)
      perm = np.random.permutation(len(domain_data))
      train_data += domain_data[perm[:round(train_split*len(domain_data))]].tolist()
      val_data += domain_data[perm[round(train_split*len(domain_data)):]].tolist()
  return train_data, val_data

