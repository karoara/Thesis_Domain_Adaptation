import sys
import numpy as np
import nltk
import math
from DA_experiments import synth1_experiments
from create_experiment_utilities import write_data, flip_label, train_val_split
np.random.seed(10)


# get synthetic experiment to set up datasets for
exp = sys.argv[1]
train_split = float(sys.argv[2])
d_train_norm = synth1_experiments[exp][0]
d_train_tech = synth1_experiments[exp][1]
d_test_norm = synth1_experiments[exp][2]
d_test_tech = synth1_experiments[exp][3]

# get data from file
f = open(sys.argv[3], "r")
data = f.readlines()
data = [l.strip() for l in data]

# set up lists of data
test_data_norm, test_data_tech = [], []
train_data, val_data, test_data = [], [], []
domains_data_norm, domains_data_tech = [[] for i in range(22)], [[] for i in range(22)]

# divide data below threshold length into separate lists, flip labels
for l in data:
  label, sentence, domain = l.split('\t')
  if len(nltk.word_tokenize(sentence)) < 128:
    if int(domain) in d_test_norm: test_data_norm.append(l)
    elif int(domain) in d_test_tech: test_data_tech.append(flip_label(label) + '\t' + sentence + '\t' + domain)
    elif int(domain) in d_train_norm: domains_data_norm[int(domain)].append(l)
    else: domains_data_tech[int(domain)].append(flip_label(label) + '\t' + sentence + '\t' + domain)

# make the test dataset
test_data = test_data_norm + test_data_tech

# separate out norm and tech domains data into train and validation
train_data_norm, val_data_norm = train_val_split(domains_data_norm, train_split)
train_data_tech, val_data_tech = train_val_split(domains_data_tech, train_split)
train_data = train_data_norm + train_data_tech
val_data = val_data_norm + val_data_tech

# write train, validation, & test data to appropriate files
write_data(exp + "_train.txt", train_data)
write_data(exp + "_val.txt", val_data)
write_data(exp + "_test.txt", test_data)

