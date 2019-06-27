# DA TRAINING UTILITIES/FUNCTIONS ---------------------------------------------
# 
# Methods for training models for Zero-Shot Domain Adaptation for language modeling.


# imports 
import sys
import torch
import numpy as np
from train_utils import write, write_history
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# function for training baseline model
def train_DA_B_lang(model,
                    tr_inputs, tr_targets, tr_lengths,
                    sd_te_inputs, sd_te_targets, sd_te_lengths,
                    te_inputs, te_targets, te_lengths,
                    epochs, learn_rate, batch_size, 
                    report_acc=0, weaken_lr=0):
  
  # optimizer, model --> GPU
  optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)
  model.to(device)
  loss = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction='sum')
  acc_loss = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction='sum')
  
  # best validation perplexity variable, corresponding training/test perplexity variables
  best_sd_te_perp = 100000
  corr_tr_perp = 100000
  corr_te_perp = 100000
  model_dict = None

  # history
  history = []
  
  # iterate over epochs
  for epoch in range(epochs):
    
    # decay learning rate if on
    if weaken_lr != 0 and (epoch+1) % 5 == 0:
      for param_group in optimizer.param_groups: param_group["lr"] *= weaken_lr
    
    # permute datapoints
    perm = np.random.permutation(len(tr_inputs))
    
    # iterate over minibatches
    if len(tr_inputs) % batch_size == 0: iterations = len(tr_inputs)//batch_size
    else: iterations = len(tr_inputs)//batch_size + 1
    for batch in range(iterations):
      
      # get batch, forward pass, backward pass, update
      b_inputs = tr_inputs[perm[batch*batch_size : (batch+1)*batch_size]]
      b_targets = tr_targets[perm[batch*batch_size : (batch+1)*batch_size]]
      b_lengths = tr_lengths[perm[batch*batch_size : (batch+1)*batch_size]]
      b_outputs, b_targets = model.forward(b_inputs, b_targets, b_lengths)
      
      optimizer.zero_grad()
      curr_loss = loss(b_outputs, b_targets)
      curr_loss.backward()
      optimizer.step()
    
    # compute perplexity after epoch, add to history
    tr_cross, tr_perp = model.accuracy(tr_inputs, tr_targets, tr_lengths, batch_size, acc_loss)
    sd_te_cross, sd_te_perp = model.accuracy(sd_te_inputs, sd_te_targets, sd_te_lengths, batch_size, acc_loss)
    te_cross, te_perp = model.accuracy(te_inputs, te_targets, te_lengths, batch_size, acc_loss)
    history.append([tr_perp, sd_te_perp, te_perp])
    
    # update best perplexities, model dicts
    if sd_te_perp < best_sd_te_perp:
      best_sd_te_perp = sd_te_perp
      corr_tr_perp = tr_perp
      corr_te_perp = te_perp
      model_dict = model.state_dict()
    
    # report perplexities and crossentropies if on
    if report_acc != 0 and ((epoch+1) % report_acc == 0 or epoch == 0):
      write("Training stuff after epoch " + str(epoch) + ": " + str(tr_cross) + " " + str(tr_perp))
      write("Same-domain test stuff after epoch " + str(epoch) + ": " + str(sd_te_cross) + " " + str(sd_te_perp))
      write("Test stuff after epoch " + str(epoch) + ": " + str(te_cross) + " " + str(te_perp) + "\n")
  
  return best_sd_te_perp, corr_tr_perp, corr_te_perp, model_dict, history


# function for training transformation-based model
def train_DA_Con_lang(model,
                      doms, dom_lengths,
                      tr_inputs, tr_targets, tr_lengths, tr_domains,
                      sd_te_inputs, sd_te_targets, sd_te_lengths, sd_te_domains,
                      te_inputs, te_targets, te_lengths, te_domains,
                      epochs, learn_rate, batch_size,
                      report_acc=0, weaken_lr=0):
  
  # set up optimizer, model --> GPU, loss
  optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)
  model.to(device)
  loss = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction='sum')
  acc_loss = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction='sum')

  # best validation perplexity variable, corresponding training/test variables
  best_sd_te_perp = 100000
  corr_tr_perp = 100000
  corr_te_perp = 100000
  model_dict = None

  # history
  history = []
  
  # iterate over epochs
  for epoch in range(epochs):
    
    # decay learning rate if on
    if weaken_lr != 0 and (epoch+1) % 10 == 0:
      for param_group in optimizer.param_groups: param_group["lr"] *= weaken_lr
    
    # permute datapoints
    perm = np.random.permutation(len(tr_inputs))
    
    # iterate over minibatches
    if len(tr_inputs) % batch_size == 0: iterations = len(tr_inputs)//batch_size
    else: iterations = len(tr_inputs)//batch_size + 1
    for batch in range(iterations):

      # get batch
      b_inputs = tr_inputs[perm[batch*batch_size : (batch+1)*batch_size]]
      b_targets = tr_targets[perm[batch*batch_size : (batch+1)*batch_size]]
      b_lengths = tr_lengths[perm[batch*batch_size : (batch+1)*batch_size]]
      b_domains = tr_domains[perm[batch*batch_size : (batch+1)*batch_size]]

      # get domain information for batch domains
      b_doms = doms[b_domains]
      if dom_lengths != None: b_dom_lengths = dom_lengths[b_domains]
      else: b_dom_lengths = None
      
      # forward pass, backward pass, update
      b_outputs, b_targets = model.forward(b_inputs, b_targets, b_lengths,
                                           b_doms, b_dom_lengths)
      
      optimizer.zero_grad()
      curr_loss = loss(b_outputs, b_targets)
      curr_loss.backward()
      optimizer.step()
    
    # compute perplexity/crossentropy after epoch, add to history
    tr_cross, tr_perp = model.accuracy(tr_inputs, tr_targets, tr_lengths, \
      tr_domains, doms, dom_lengths, batch_size, acc_loss)
    sd_te_cross, sd_te_perp = model.accuracy(sd_te_inputs, sd_te_targets, sd_te_lengths, \
      sd_te_domains, doms, dom_lengths, batch_size, acc_loss)
    te_cross, te_perp = model.accuracy(te_inputs, te_targets, te_lengths, \
      te_domains, doms, dom_lengths, batch_size, acc_loss)
    history.append([tr_perp, sd_te_perp, te_perp])
    
    # update best accuracies, model dicts
    if sd_te_perp < best_sd_te_perp:
      best_sd_te_perp = sd_te_perp
      corr_tr_acc = tr_perp
      corr_te_acc = te_perp
      model_dict = model.state_dict()
    
    # report accuracies if on
    if report_acc != 0 and ((epoch+1) % report_acc == 0 or epoch == 0):
      write("Training stuff after epoch " + str(epoch) + ": " + str(tr_cross) + " " + str(tr_perp))
      write("Same-domain test stuff after epoch " + str(epoch) + ": " + str(sd_te_cross) + " " + str(sd_te_perp))
      write("Test stuff after epoch " + str(epoch) + ": " + str(te_cross) + " " + str(te_perp) + "\n")
  
  return best_sd_te_perp, corr_tr_perp, corr_te_perp, model_dict, history

