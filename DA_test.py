# DA MODEL EXPERIMENTS --------------------------------------------------------
# 
# Script for training models on multiple domains and testing their performance
# on new domains (Zero-Shot Domain Adaptation) for the sentiment analysis task.


# IMPORTS, ETC ----------------------------------------------------------------

import timeit
import nltk
import numpy as np
import torch
from modifying_methods import get_pca_words
from embedding_helper_methods import read_embeddings, normalize_embeddings
from data_utilities import DA_load_data, get_rel_words, get_sent_vect_indices
from data_utilities import get_sent_vect_indices_2, get_sentence_vectors, balance
from data_utilities import transform_sent_vects
from models import DA_B, DA_TRANS
from train_utils import train_DA_B, train_DA_Con, write, write_history
from domain_info import amazon_domains, reddit_domains, synth3_domain, stopwords
from DA_experiments import amazon_experiments, reddit_experiments
from DA_experiments import synth1_experiments, synth2_experiments, synth3_experiments
import sys
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

np.random.seed(10)
torch.manual_seed(10)

write("\nImports complete. \n")


# EMBEDDINGS, TEST INFO -------------------------------------------------------

# set up embeddings
words, vectors = read_embeddings()
vectors = normalize_embeddings(vectors)
vectors = vectors * 3
lookup = dict(zip(words, np.arange(len(vectors))))

# get job information
exp = sys.argv[1]
mod_type = sys.argv[2]
DK_type = sys.argv[3]
trials = int(sys.argv[4])
write_hist = sys.argv[5]
exp_type, exp_no = exp.split("/")

# get training & testing domains
if exp_type == "amazon": 
  d_train, d_test = amazon_experiments[exp_no]
  domains = amazon_domains
elif exp_type == "yelp":
  dom_file = open("experiments/" + exp_type + "/yelp_DK.txt", "r")
  domains = dom_file.readlines()
  dom_file.close()
elif exp_type == "reddit":
  d_train, d_test = reddit_experiments[exp_no]
  domains = reddit_domains
elif exp_type == "synth1": 
  d_train_norm, d_train_tech, d_test_norm, d_test_tech = synth1_experiments[exp_no]
  d_train, d_test = d_train_norm + d_train_tech, d_test_norm + d_test_tech
  domains = amazon_domains
elif exp_type == "synth2":
  d_train, d_test = synth2_experiments[exp_no][0], synth2_experiments[exp_no][0]
  domains = amazon_domains
else: # synth3
  num_words = int(sys.argv[6])
  _, num_domains, num_test_domains = synth3_experiments[exp_no]
  d_train = [i for i in range(num_domains - num_test_domains)]
  d_test = [i for i in range((num_domains - num_test_domains), num_domains)]
  domains = [synth3_domain]

write("Embeddings, test info set up. \n")


# DATA ------------------------------------------------------------------------

# set up training data
tr_sents, tr_labels, tr_domains = [], [], []

# fetch training data, validation, test data (not yelp)
if exp_type != "yelp":
  for d in d_train:
    c_sents, c_labels, c_domains, _ = DA_load_data("experiments/" + exp + "/" + exp_no + "_train.txt", [d])
    c_sents, c_labels, c_domains = balance(c_sents, c_labels, c_domains)
    tr_sents += c_sents
    tr_labels += c_labels
    tr_domains += c_domains
  sd_te_sents, sd_te_labels, sd_te_domains, _ = DA_load_data("experiments/" + exp + "/" + exp_no + "_val.txt", d_train)
  te_sents, te_labels, te_domains, _ = DA_load_data("experiments/" + exp + "/" + exp_no + "_test.txt", d_test)

# if yelp (no need to balance)
else:
  tr_sents, tr_labels, tr_domains, _ = DA_load_data("experiments/" + exp + "/" + exp_no + "_train.txt")
  sd_te_sents, sd_te_labels, sd_te_domains, _ = DA_load_data("experiments/" + exp + "/" + exp_no + "_val.txt")
  te_sents, te_labels, te_domains, _ = DA_load_data("experiments/" + exp + "/" + exp_no + "_test.txt")

# get all data
all_sents = tr_sents + sd_te_sents + te_sents
if exp_type == "synth3": all_labels = tr_labels + sd_te_labels + te_labels

# get domain data
domains_tok = []

if DK_type == "Norm": 
  for domain in domains: domains_tok.append(nltk.word_tokenize(domain))
else:
  for domain in domains: domains_tok.append([word for word in nltk.word_tokenize(domain) if word not in stopwords])

# if synth3, get DK knowledge to transform
if exp_type == "synth3": domain_tok_for_trans = [word for word in nltk.word_tokenize(synth3_domain) if word not in stopwords]

write("Data set up. \n")


# PREPARE VECTORS, GET LENGTHS ------------------------------------------------

# get words in the data
all_data = np.concatenate((all_sents + domains_tok), axis=0)
rel_words = get_rel_words(all_data, words, vectors, lookup, mds=True)

# get words to be domain-specific if synth3
if exp_type == "synth3":
  words_dict = get_pca_words(rel_words, stopwords, all_sents, all_labels)
  trans_words = list(words_dict.keys())[-num_words:]
  trans_words += domain_tok_for_trans
else:
  trans_words = None

# get training, same-domain testing, testing vector indices
x_tr_sents, y_tr, d_tr, x_tr_indices, x_tr_trans = get_sent_vect_indices(tr_sents, 
                                                                        tr_labels, 
                                                                        tr_domains, 
                                                                        rel_words,
                                                                        vectors, 
                                                                        lookup,
                                                                        trans_words)

x_sd_te_sents, y_sd_te, d_sd_te, x_sd_te_indices, x_sd_te_trans = get_sent_vect_indices(sd_te_sents, 
                                                                                        sd_te_labels, 
                                                                                        sd_te_domains, 
                                                                                        rel_words, 
                                                                                        vectors, 
                                                                                        lookup,
                                                                                        trans_words)

x_te_sents, y_te, d_te, x_te_indices, x_te_trans = get_sent_vect_indices(te_sents, 
                                                                        te_labels, 
                                                                        te_domains, 
                                                                        rel_words, 
                                                                        vectors, 
                                                                        lookup,
                                                                        trans_words)

# get vector sequences
x_tr = get_sentence_vectors(x_tr_indices, vectors)
x_sd_te = get_sentence_vectors(x_sd_te_indices, vectors)
x_te = get_sentence_vectors(x_te_indices, vectors)

# if synth3, get transformations for each domain and transform data
if exp_type == "synth3":
  transforms = np.asarray([np.random.rand(vectors.shape[1]) for i in range(num_domains)])
  x_tr = transform_sent_vects(x_tr, x_tr_trans, d_tr, transforms)
  x_sd_te = transform_sent_vects(x_sd_te, x_sd_te_trans, d_sd_te, transforms)
  x_te = transform_sent_vects(x_te, x_te_trans, d_te, transforms)

# get lengths
l_tr = [len(x_tr_sent) for x_tr_sent in x_tr]
l_sd_te = [len(x_sd_te_sent) for x_sd_te_sent in x_sd_te]
l_te = [len(x_te_sent) for x_te_sent in x_te]

# shift to numpy
x_tr, y_tr = np.asarray(x_tr), np.asarray(y_tr) 
d_tr, l_tr = np.asarray(d_tr), np.asarray(l_tr)
x_sd_te, y_sd_te = np.asarray(x_sd_te), np.asarray(y_sd_te)
d_sd_te, l_sd_te = np.asarray(d_sd_te), np.asarray(l_sd_te)
x_te, y_te = np.asarray(x_te), np.asarray(y_te)
d_te, l_te = np.asarray(d_te), np.asarray(l_te) 

# domain vector setup
dom_sents, dom_indices = get_sent_vect_indices_2(domains_tok,
                                                 rel_words,
                                                 vectors,
                                                 lookup)
doms = get_sentence_vectors(dom_indices, vectors)

# transform domain knowledge (and add as new domain knowledge) if synth3
if exp_type == "synth3": doms = [np.matmul(doms[0], transforms[i]) for i in range(num_domains)]

# get domain knowledge in final, appropriate form
l_doms = [len(dom) for dom in doms]
doms, l_doms = np.asarray(doms), np.asarray(l_doms)

if DK_type == "Avg":
  new_doms = np.zeros((len(doms), vectors.shape[1]))
  for i in range(len(doms)): new_doms[i] = np.mean(doms[i], axis=0)
  doms, l_doms = new_doms, None

write("Preparing vectors done. \n")


# SET UP MODEL ----------------------------------------------------------------

# hyperparameters for each trial
epochs = 50
learn_rate = 0.003
batch_size = 128
loss = torch.nn.BCELoss()

write("Setting up model done. \n \n")


# TRAIN -----------------------------------------------------------------------

# accuracies, parameters for model that gives best validation accuracy across trials
all_time_val_acc = 0
all_time_train_acc = 0
all_time_test_acc = 0
all_time_m_dict = None

# accuracies across trials (for calculating averages)
val_accs = []
train_accs = []
test_accs = []

# test the model "trials" # of times
for t in range(trials):

  start = timeit.default_timer()

  # setup, training for baseline model
  if mod_type == "B":

    model = DA_B(in_d=vectors.shape[1], h_d=100, layers=2, dropout=0.5, bi=True)

    val_acc, train_acc, test_acc, model_dict, history = train_DA_B(model,
                                                                   x_tr, y_tr, l_tr,
                                                                   x_sd_te, y_sd_te, l_sd_te,
                                                                   x_te, y_te, l_te,
                                                                   epochs, learn_rate, loss, batch_size,
                                                                   report_acc=5, weaken_lr=0)
  
  # setup, training for conditioning model
  else:
    
    if DK_type == "Norm": model = DA_TRANS(in_d=vectors.shape[1], attn_width=20, dom_seq=True,
                                           s_h_d=100, s_layers=2, s_dropout=0.5, s_bi=True,
                                           d_h_d=20, d_layers=1, d_dropout=0, d_bi=True)
    
    else: model = DA_TRANS(in_d=vectors.shape[1], attn_width=20, dom_seq=False,
                           s_h_d=100, s_layers=2, s_dropout=0.5, s_bi=True,
                           d_h_d=None, d_layers=None, d_dropout=None, d_bi=None)
    
    val_acc, train_acc, test_acc, model_dict, history = train_DA_Con(model,
                                                                     doms, l_doms,
                                                                     x_tr, y_tr, l_tr, d_tr,
                                                                     x_sd_te, y_sd_te, l_sd_te, d_sd_te,
                                                                     x_te, y_te, l_te, d_te,
                                                                     epochs, learn_rate, loss, batch_size,
                                                                     report_acc=5, weaken_lr=0)
  
  end = timeit.default_timer()

  time_taken = end - start

  # print history if on
  if write_hist == "Hist":
    write("Full trial history: ")
    write_history(history)

  # print results of current trial
  write("Training done.")
  write("Best training accuracy: " + str(train_acc))
  write("Best validation accuracy: " + str(val_acc))
  write("Best test accuracy: " + str(test_acc))
  write("Time taken: " + str(time_taken) + "\n")
  
  # append current accuracies to lists
  val_accs.append(val_acc)
  train_accs.append(train_acc)
  test_accs.append(test_acc)

  # update best accuracies, model if needed
  if val_acc > all_time_val_acc:
    all_time_val_acc = val_acc
    all_time_train_acc = train_acc
    all_time_test_acc = test_acc
    all_time_m_dict = {}
    all_time_m_dict[exp_type + "_" + exp_no + "_" + mod_type + "_" + DK_type] = model_dict

# save best performing model from trials
torch.save(all_time_m_dict, "models/" + exp_type + "_" + exp_no + "_" + mod_type + "_" + DK_type + ".pt")

# print best accuracies from trials
write("\nBest training accuracy over trials: " + str(all_time_train_acc))
write("Best validation accuracy over trials: " + str(all_time_val_acc))
write("Best test accuracy over trials: " + str(all_time_test_acc) + "\n")

# calculate and print average accuracies from trials
write("Average training accuracy over trials: " + str(np.mean(np.asarray(train_accs))))
write("Average validation accuracy over trials: " + str(np.mean(np.asarray(val_accs))))
write("Average test accuracy over trials: " + str(np.mean(np.asarray(test_accs))) + "\n")

