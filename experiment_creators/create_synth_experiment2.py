import sys
import numpy as np
import nltk
import math
from DA_experiments import synth2_experiments
from create_experiment_utilities import write_data, flip_label
np.random.seed(10)


# get synthetic experiment to set up datasets for
exp = sys.argv[1]
train_split = float(sys.argv[2])
flip_domains = synth2_experiments[exp]

# get data from file
f = open(sys.argv[3], "r")
data = f.readlines()
data = [l.strip() for l in data]

# set up lists of data
norm_data, flip_data = [], []
train_data, val_data, test_data = [], [], []

# get data below threshold length, separate out into lists
for l in data:
  label, sentence, domain = l.split('\t')
  if len(nltk.word_tokenize(sentence)) < 128 and int(domain) in flip_domains:
    norm_data.append(l)
    flip_data.append(flip_label(label) + '\t' + sentence + '\t' + str(int(domain)+1))

# make the train, validation, test datasets
norm_data, flip_data = np.asarray(norm_data), np.asarray(flip_data)
perm = np.random.permutation(len(norm_data))
train_data += norm_data[perm[:round(train_split*len(norm_data))]].tolist()
train_data += flip_data[perm[:round(train_split*len(norm_data))]].tolist()
val_data += norm_data[perm[round(train_split*len(norm_data)):]].tolist()
val_data += flip_data[perm[round(train_split*len(norm_data)):]].tolist()
test_data += norm_data[perm[round(train_split*len(norm_data)):]].tolist()
test_data += flip_data[perm[round(train_split*len(norm_data)):]].tolist()

# write train, validation, & test data to appropriate files
write_data(exp + "_train.txt", train_data)
write_data(exp + "_val.txt", val_data)
write_data(exp + "_test.txt", test_data)

