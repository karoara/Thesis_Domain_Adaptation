import sys
import numpy as np
import nltk
import math
from DA_experiments import synth3_experiments
from create_experiment_utilities import write_data, flip_label, train_val_split
np.random.seed(10)


# get synthetic experiment to set up datasets for
exp = sys.argv[1]
train_split = float(sys.argv[2])

# get info on experiment
exp_info = synth3_experiments[exp]
s_per_domain = int(exp_info[0])
num_domains = int(exp_info[1])
num_test_domains = int(exp_info[2])

# training and testing domains
d_train = [i for i in range(num_domains - num_test_domains)]
d_test = [i for i in range((num_domains - num_test_domains), num_domains)]

# get data from file
f = open(sys.argv[3], "r")
data = f.readlines()
data = [l.strip() for l in data]

# randomly permute the data
perm = np.random.permutation(len(data))
data = np.asarray(data)[perm].tolist()

# set up experiment data
exp_data = []

# iterate over number of domains, set up "domain" data
for l in range(num_domains * s_per_domain):
    label, sentence, domain = data[l].split('\t')
    exp_data.append(label + '\t' + sentence + '\t' + str(l//s_per_domain))

# set up lists of data
train_data, val_data, test_data = [], [], []
domains_data = [[] for i in range(num_domains)]

# separate experiment data into test, domains data
for l in exp_data:
  label, sentence, domain = l.split('\t')
  if int(domain) in d_test: test_data.append(l)
  else: domains_data[int(domain)].append(l)

# separate out domains data into train, validation
train_data, val_data = train_val_split(domains_data, train_split)

# write train, validation, & test data to appropriate files
write_data(exp + "_train.txt", train_data)
write_data(exp + "_val.txt", val_data)
write_data(exp + "_test.txt", test_data)

