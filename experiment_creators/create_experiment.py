import sys
import numpy as np
import nltk
from DA_experiments import experiments
from create_experiment_utilities import write_data, train_val_split
np.random.seed(10)


# get experiment to set up datasets for
exp = sys.argv[1]
train_split = float(sys.argv[2])
d_train = experiments[exp][0]
d_test = experiments[exp][1]

# get data from file
f = open(sys.argv[3], "r")
data = f.readlines()
data = [l.strip() for l in data]

# set up lists of data
train_data, val_data, test_data = [], [], []
domains_data = [[] for i in range(len(d_train + d_test))]

# divide data below threshold length into test_data & domains_data
for l in data:
  if len(l.split('\t')) == 3:
    label, sentence, domain = l.split('\t')
    if len(nltk.word_tokenize(sentence)) < 128:
      if int(domain) in d_test: test_data.append(l)
      else: domains_data[int(domain)].append(l)

# separate out domains_data into train and validation
train_data, val_data = train_val_split(domains_data, train_split)

# write data to appropriate files
write_data(exp + "_train.txt", train_data)
write_data(exp + "_val.txt", val_data)
write_data(exp + "_test.txt", test_data)

