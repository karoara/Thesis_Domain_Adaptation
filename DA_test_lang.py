# DA MODEL EXPERIMENTS --------------------------------------------------------
# 
# Script for training models on multiple domains and testing their performance
# on new domains (Zero-Shot Domain Adaptation) for the language modeling task.


# IMPORTS, ETC ----------------------------------------------------------------

import timeit
import nltk
import numpy as np
import torch
from scipy.stats import ortho_group
from modifying_methods import get_pca_words
from embedding_helper_methods import read_embeddings, normalize_embeddings
from data_utilities import DA_load_data, get_rel_words
from data_utilities import get_sent_vect_indices_2, get_sentence_vectors
from data_utilities import replace_oov, get_sent_vect_indices_lang
from models_lang import DA_B_lang, DA_TRANS_lang
from train_utils_lang import train_DA_B_lang, train_DA_Con_lang
from train_utils import write, write_history
from domain_info import amazon_domains, reddit_domains, stopwords
from DA_experiments import amazon_experiments, reddit_experiments
from sklearn.decomposition import PCA
import sys
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

np.random.seed(10)
torch.manual_seed(10)

write("\nImports complete. \n")


# EMBEDDINGS, TEST INFO -------------------------------------------------------

# set up embeddings
words, vectors = read_embeddings()
vectors = normalize_embeddings(vectors)
lookup = dict(zip(words, np.arange(len(vectors))))

# get job information
exp = sys.argv[1]
mod_type = sys.argv[2]
DK_type = sys.argv[3]
trials = int(sys.argv[4])
write_hist = sys.argv[5]
out_style = sys.argv[6]
small_or_nah = sys.argv[7]
mb_size = int(sys.argv[8])
lr = float(sys.argv[9])
check = sys.argv[10]
exp_type, exp_no = exp.split("/")

# get training & testing domains, domain knowledge
if exp_type == "amazon": 
  d_train, d_test = amazon_experiments[exp_no]
  domains = amazon_domains
else: # yelp
  dom_file = open("experiments/" + exp_type + "/yelp_DK.txt", "r")
  domains = dom_file.readlines()
  dom_file.close()

write("Embeddings, test info set up. \n")


# DATA ------------------------------------------------------------------------

# fetch training data, validation, test data
if exp_type == "amazon":
  tr_sents, _, tr_domains, _ = DA_load_data("experiments/" + exp + "/" + exp_no + "_train.txt", d_train)
  sd_te_sents, _, sd_te_domains, _ = DA_load_data("experiments/" + exp + "/" + exp_no + "_val.txt", d_train)
  te_sents, _, te_domains, _ = DA_load_data("experiments/" + exp + "/" + exp_no + "_test.txt", d_test)
else:
  tr_sents, _, tr_domains, _ = DA_load_data("experiments/" + exp + "/" + exp_no + "_train.txt")
  sd_te_sents, _, sd_te_domains, _ = DA_load_data("experiments/" + exp + "/" + exp_no + "_val.txt")
  te_sents, _, te_domains, _ = DA_load_data("experiments/" + exp + "/" + exp_no + "_test.txt")

# get all data (for vocab), domain knowledge
all_sents = tr_sents + sd_te_sents + te_sents
domains_tok = []

if DK_type == "Norm": 
  for domain in domains: domains_tok.append(nltk.word_tokenize(domain))
else:
  for domain in domains: domains_tok.append([word for word in nltk.word_tokenize(domain) if word not in stopwords])

write("Data set up. \n")


# PREPARE VECTORS, GET LENGTHS ------------------------------------------------

# get words in the data
all_data = np.concatenate((all_sents + domains_tok), axis=0)
rel_words = get_rel_words(all_data, words, vectors, lookup, mds=True)

# create out of vocabulary token, 0's vector, index
oov_tok = "oov"
oov_vect = np.zeros(200)

# iterate over data, replace oov words with oov token
tr_sents, d_tr = replace_oov(tr_sents, tr_domains, rel_words, oov_tok)
sd_te_sents, d_sd_te = replace_oov(sd_te_sents, sd_te_domains, rel_words, oov_tok)
te_sents, d_te = replace_oov(te_sents, te_domains, rel_words, oov_tok)

# new set of words, vectors based on what's relevant
new_vectors = np.zeros((len(rel_words) + 1, vectors.shape[1]))
for i in range(len(rel_words)): new_vectors[i] = vectors[lookup[rel_words[i]]]

# add oov word to this and reset words, vectors, lookup
new_vectors[len(rel_words)] = oov_vect
rel_words.append(oov_tok)
vectors, words = new_vectors, rel_words
lookup = dict(zip(words, np.arange(len(vectors))))

# vocab size
write("Vocabulary size for experiment: ")
write(len(words))
write("\n")

# get training, same-domain testing, testing vector indices
x_tr, y_tr = get_sent_vect_indices_lang(tr_sents, vectors, lookup)
x_sd_te, y_sd_te = get_sent_vect_indices_lang(sd_te_sents, vectors, lookup)
x_te, y_te = get_sent_vect_indices_lang(te_sents, vectors, lookup)

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
dom_sents, dom_indices = get_sent_vect_indices_2(domains_tok, words, vectors, lookup)
doms = get_sentence_vectors(dom_indices, vectors)

# get domain knowledge in final, appropriate form
l_doms = [len(dom) for dom in doms]
doms, l_doms = np.asarray(doms), np.asarray(l_doms)

if DK_type == "Avg":
  new_doms = np.zeros((len(doms), vectors.shape[1]))
  for i in range(len(doms)): new_doms[i] = np.mean(doms[i], axis=0)
  doms, l_doms = new_doms, None

  if check == "control":
    doms = np.ones(doms.shape)

# prep output layer
pca = PCA(n_components=100)
output_vectors = pca.fit_transform(vectors)

write("Preparing vectors done. \n")


# SET UP MODEL ----------------------------------------------------------------

# if testing on a batch of datapoints
if small_or_nah == "yes":
  x_tr, y_tr = x_tr[:128], y_tr[:128]
  d_tr, l_tr = d_tr[:128], l_tr[:128]

# hyperparameters for each trial
epochs = 200
learn_rate = lr
batch_size = mb_size

write("Setting up model done. \n \n")


# TRAIN -----------------------------------------------------------------------

# accuracies, parameters for model that gives best validation accuracy across trials
all_time_val_perp = 100000
all_time_train_perp = 100000
all_time_test_perp = 100000
all_time_m_dict = None

# accuracies across trials (for calculating averages)
val_perps = []
train_perps = []
test_perps = []

# test the model "trials" # of times
for t in range(trials):

  start = timeit.default_timer()

  # setup, training for baseline model
  if mod_type == "B":

    model = DA_B_lang(in_d=vectors.shape[1], h_d=2048, layers=2, dropout=0.5, bi=False,
                      vocab_size=len(words), vectors=vectors, output_vectors=output_vectors)

    val_perp, train_perp, test_perp, model_dict, history = train_DA_B_lang(model,
                                                                           x_tr, y_tr, l_tr,
                                                                           x_sd_te, y_sd_te, l_sd_te,
                                                                           x_te, y_te, l_te,
                                                                           epochs, learn_rate, batch_size,
                                                                           report_acc=5, weaken_lr=0.8)
  
  # setup, training for conditioning model
  else:
    
    model = DA_TRANS_lang(in_d=vectors.shape[1], attn_width=20, dom_seq=False, 
                          vocab_size=len(words), out_style=out_style, vectors=vectors, output_vectors=output_vectors,
                          s_h_d=2048, s_layers=2, s_dropout=0.5, s_bi=False,
                          d_h_d=None, d_layers=None, d_dropout=None, d_bi=None)
    
    val_perp, train_perp, test_perp, model_dict, history = train_DA_Con_lang(model,
                                                                             doms, l_doms,
                                                                             x_tr, y_tr, l_tr, d_tr,
                                                                             x_sd_te, y_sd_te, l_sd_te, d_sd_te,
                                                                             x_te, y_te, l_te, d_te,
                                                                             epochs, learn_rate, batch_size,
                                                                             report_acc=5, weaken_lr=0.8)
  
  end = timeit.default_timer()

  time_taken = end - start

  # print history if on
  if write_hist == "Hist":
    write("Full trial history: ")
    write_history(history)

  # print results of current trial
  write("Training done.")
  write("Best training perplexity: " + str(train_perp))
  write("Best validation perplexity: " + str(val_perp))
  write("Best test perplexity: " + str(test_perp))
  write("Time taken: " + str(time_taken) + "\n")
  
  # append current accuracies to lists
  val_perps.append(val_perp)
  train_perps.append(train_perp)
  test_perps.append(test_perp)

  # update best accuracies, model if needed
  if val_perp < all_time_val_perp:
    all_time_val_perp = val_perp
    all_time_train_perp = train_perp
    all_time_test_perp = test_perp
    all_time_m_dict = {}
    all_time_m_dict[exp_type + "_" + exp_no + "_" + mod_type + "_" + out_style] = model_dict

# save best performing model from trials
torch.save(all_time_m_dict, "models_lang/" + exp_type + "_" + exp_no + "_" + mod_type + "_" + out_style + ".pt")

# print best accuracies from trials
write("\nBest training perplexity over trials: " + str(all_time_train_perp))
write("Best validation perplexity over trials: " + str(all_time_val_perp))
write("Best test perplexity over trials: " + str(all_time_test_perp) + "\n")

# calculate and print average accuracies from trials
write("Average training perplexity over trials: " + str(np.mean(np.asarray(train_perps))))
write("Average validation perplexity over trials: " + str(np.mean(np.asarray(val_perps))))
write("Average test perplexity over trials: " + str(np.mean(np.asarray(test_perps))) + "\n")

